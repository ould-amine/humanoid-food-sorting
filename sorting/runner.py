"""Rollout orchestration against an injected SortingEnvironment; no auto-run."""

import numpy as np
import json

from .recording import EpisodeRecorder
from .contracts import Prediction, CATEGORIES
from .retargeting import WaypointController, make_plan


def collect_episode(
    env,
    classifier,
    metadata,
    output,
    *,
    demo=None,
    policy=None,
    max_steps=600,
    execute_chunk_steps=1,
    classification_frames=5,
    classification_stride=3,
):
    """No oracle category is passed to the classifier/policy.

    Simulator positions may be used by the deterministic planner and are labelled
    explicitly. Learned policy inputs contain only image/state/predicted category.
    """
    recorder = None
    try:
        obs = env.reset(metadata["seed"])
        if min(classification_frames, classification_stride) < 1:
            raise ValueError("Classification window and stride must be positive")
        # Gather an explicitly pre-grasp window: hold position with fingers open.
        predictions, classification_crops, perception_frames, perception_metadata = (
            [],
            [],
            [],
            [],
        )
        for index in range(classification_frames):
            crop = env.food_crop(obs)
            classification_crops.append(crop.copy())
            if hasattr(env, "last_perception_frame"):
                perception_frames.append(env.last_perception_frame.copy())
                perception_metadata.append(dict(env.last_perception_metadata))
            predictions.append(classifier.predict(crop))
            if index < classification_frames - 1:
                for _ in range(classification_stride):
                    obs, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        scene = env.scene()
        scores = {
            c: float(np.mean([p.probabilities[c] for p in predictions]))
            for c in CATEGORIES
        }
        prediction = Prediction(
            max(scores, key=scores.get), scores, predictions[0].method
        )
        meta = {
            **metadata,
            "predicted_category": prediction.category,
            "position_source": scene.position_source,
            "classifier": prediction.method,
            "classifier_scores": prediction.probabilities,
            "classification_frames": classification_frames,
            "classification_stride_control_steps": classification_stride,
        }
        if demo is not None:
            meta["human_video_ids"] = sorted(
                set(meta.get("human_video_ids", []) + [demo.video_id])
            )
        controller = policy or WaypointController(
            make_plan(scene, prediction.category, demo), scene.eef_down_quat_xyzw
        )
        controller.reset()
        recorder = EpisodeRecorder(output, meta, obs)
        import cv2

        for index, crop in enumerate(classification_crops):
            cv2.imwrite(
                str(recorder.output / f"classification_crop_{index:02d}.png"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            )
        for index, frame in enumerate(perception_frames):
            cv2.imwrite(
                str(recorder.output / f"perception_frame_{index:02d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        if perception_metadata:
            (recorder.output / "perception.json").write_text(
                json.dumps(perception_metadata, indent=2)
            )
            recorder.metadata["perception_camera"] = perception_metadata[0]["camera"]
            recorder.metadata["classification_per_crop"] = [
                {"index": i, "category": p.category, "probabilities": p.probabilities}
                for i, p in enumerate(predictions)
            ]
        # Instruction must not encode the ground-truth category/basket.
        instruction = "Sort the food into its category basket."
        recorder.metadata["instruction"] = instruction
        failure = "step_limit"
        completed = False
        steps = 0
        while steps < max_steps:
            try:
                chunk = np.asarray(
                    controller.act(obs, prediction.category, instruction)
                )
            except StopIteration:
                completed, failure = True, None
                break
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0:
                raise ValueError("Policy must return a non-empty H,7 action chunk")
            if execute_chunk_steps < 1:
                raise ValueError("execute_chunk_steps must be positive")
            for action in chunk[: min(execute_chunk_steps, max_steps - steps)]:
                obs, events = env.step(action)
                recorder.append(action, obs, events)
                steps += 1
                if events.terminated:
                    completed, failure = True, None
                    break
            if completed:
                break
        if completed and recorder.events and not recorder.events[-1].task_success:
            failure = (
                "wrong_basket"
                if recorder.events[-1].placed_basket
                else "no_stable_placement"
            )
        if hasattr(env, "physical_state"):
            physical = env.physical_state()
            recorder.metadata["final_object_position"] = physical.object_center.tolist()
            recorder.metadata["final_object_half_extents"] = (
                physical.object_half_extents.tolist()
            )
            recorder.metadata["final_object_speed_m_s"] = physical.object_speed_m_s
        return recorder.finish(completed=completed, failure_reason=failure)
    except (TimeoutError, RuntimeError, ValueError) as error:
        if recorder and recorder.actions and not recorder.saved:
            recorder.finish(completed=False, failure_reason=str(error))
        raise
    finally:
        env.close()
