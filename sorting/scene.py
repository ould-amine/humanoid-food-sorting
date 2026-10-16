"""Custom robosuite task using the same MuJoCo/Panda stack as LIBERO.

Food is a graspable cuboid proxy, optionally textured with a real training crop.
This is not a standard LIBERO benchmark or a realistic food mesh.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.controllers import load_controller_config

from .contracts import RobotObservation, Scene
from .environment import BasketRegion, PhysicalState, SuccessMonitor


def basket_sign(label):
    """Render a text asset for a physical, camera-facing destination sign."""
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(".cache/basket_signs")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{label.lower()}_v1.png"
    if not path.exists():
        image = Image.new("RGB", (768, 160), "white")
        draw = ImageDraw.Draw(image)
        font = None
        for candidate in (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            if Path(candidate).exists():
                font = ImageFont.truetype(candidate, 82)
                break
        if font is None:
            font = ImageFont.load_default(size=82)
        draw.rectangle((3, 3, 764, 156), outline="black", width=6)
        draw.text((384, 80), label, font=font, fill="black", anchor="mm")
        image.save(path)
    return str(path.resolve())


class FoodSorting(Lift):
    def __init__(
        self,
        *,
        category,
        swap=False,
        texture=None,
        randomize=False,
        image_size=256,
        perception_fovy=24,
        **kwargs,
    ):
        self.category = category
        self.swap = swap
        self.food_texture = None
        if texture:
            import hashlib
            from PIL import Image

            source = Path(texture)
            cache = Path(".cache/food_textures")
            cache.mkdir(parents=True, exist_ok=True)
            destination = cache / (
                hashlib.sha256(source.read_bytes()).hexdigest() + ".png"
            )
            if not destination.exists():
                with Image.open(source) as image:
                    image.convert("RGB").resize((256, 256)).save(destination)
            self.food_texture = str(destination.resolve())
        self.randomize = randomize
        self.perception_fovy = perception_fovy
        self.centers = {
            "fruit_basket": np.array([0.12, -0.18 if swap else 0.18, 0.808]),
            "vegetable_basket": np.array([0.12, 0.18 if swap else -0.18, 0.808]),
        }
        self.regions = {
            name: BasketRegion(p + [-0.075, -0.075, 0], p + [0.075, 0.075, 0.10])
            for name, p in self.centers.items()
        }
        self.monitor = SuccessMonitor(category, self.regions)
        super().__init__(
            robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            initialization_noise=None,
            camera_names="agentview",
            camera_heights=image_size,
            camera_widths=image_size,
            camera_segmentations="element",
            control_freq=20,
            horizon=800,
            hard_reset=False,
            use_object_obs=False,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        camera = next(
            c
            for c in self.model.worldbody.iter("camera")
            if c.get("name") == "agentview"
        )
        camera.set("fovy", "60")  # Keep both destinations and signs within frame.
        # Rendered on demand only during pre-grasp classification; pose is aimed
        # at the object at reset, with a lateral offset to avoid gripper occlusion.
        ET.SubElement(
            self.model.worldbody,
            "camera",
            name="foodview",
            mode="fixed",
            pos="-.12 0 1.08",
            quat="1 0 0 0",
            fovy=str(self.perception_fovy),
        )
        # Destination identity remains visible to learned policies when swapped.
        # Food appearance never depends on its category in this proxy task.
        for name, p in self.centers.items():
            body = ET.SubElement(
                self.model.worldbody, "body", name=name, pos=" ".join(map(str, p))
            )
            label = "FRUIT" if name == "fruit_basket" else "VEGETABLE"
            ET.SubElement(
                self.model.asset,
                "texture",
                name=f"{name}_label_tex",
                type="2d",
                file=basket_sign(label),
            )
            ET.SubElement(
                self.model.asset,
                "material",
                name=f"{name}_label_mat",
                texture=f"{name}_label_tex",
                texrepeat="1 1",
                texuniform="false",
                emission="0.4",
                specular="0",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_sign",
                type="plane",
                size=".083 .018 .001",
                pos=".105 0 .065",
                quat=camera.get("quat"),
                material=f"{name}_label_mat",
                group="1",
                contype="0",
                conaffinity="0",
            )
            pieces = [("floor", [0, 0, -0.004], [0.085, 0.085, 0.004])]
            for sign in (-1, 1):
                pieces += [
                    (f"x{sign}", [sign * 0.08, 0, 0.03], [0.005, 0.085, 0.03]),
                    (f"y{sign}", [0, sign * 0.08, 0.03], [0.075, 0.005, 0.03]),
                ]
            for label, pos, size in pieces:
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{name}_{label}",
                    type="box",
                    pos=" ".join(map(str, pos)),
                    size=" ".join(map(str, size)),
                    rgba="0.2 0.5 1 1" if name == "fruit_basket" else "1 .5 .15 1",
                    group="1",
                    friction="1 0.005 0.0001",
                )
        if self.food_texture:
            # Reuse Lift's material with a personally collected image.
            for texture in self.model.asset.findall("texture"):
                if "redwood" in texture.get("name", ""):
                    name = texture.get("name")
                    texture.attrib.clear()
                    texture.attrib.update(
                        name=name, type="cube", file=self.food_texture
                    )
            for material in self.model.asset.findall("material"):
                if "redwood" in material.get("name", ""):
                    material.set("texrepeat", "1 1")
            for geom in self.model.worldbody.iter("geom"):
                if geom.get("material", "").endswith("redwood_mat"):
                    geom.set("rgba", "1 1 1 1")

    def _reset_internal(self):
        super()._reset_internal()
        xy = np.array([-0.12, 0.0])
        if self.randomize:
            xy += np.random.uniform(-0.025, 0.025, 2)
        z = self.table_offset[2] + self.cube.size[2] + 0.001
        self.sim.data.set_joint_qpos(
            self.cube.joints[0], np.r_[xy, z, 1.0, 0.0, 0.0, 0.0]
        )
        self.sim.forward()

    def _check_success(self):
        # Reward never advances the monitor; update it exactly once per control step.
        return False


class SortingScene:
    """SortingEnvironment implementation, with oracle geometry/segmentation labelled."""

    def __init__(
        self, category, seed=0, perception_view="closeup", perception_size=512, **kwargs
    ):
        if perception_view not in ("closeup", "agentview") or perception_size < 64:
            raise ValueError("Invalid perception camera configuration")
        self.perception_view, self.perception_size = perception_view, perception_size
        self.seed_value = seed
        np.random.seed(seed)
        self.env = FoodSorting(category=category, **kwargs)
        self.steps = 0
        self.latest_raw = None

    def _obs(self, raw):
        self.latest_raw = raw
        return RobotObservation(
            np.ascontiguousarray(raw["agentview_image"][::-1]),
            np.r_[
                raw["robot0_eef_pos"],
                raw["robot0_eef_quat"],
                raw["robot0_gripper_qpos"],
            ],
            self.steps / 20,
        )

    def reset(self, seed):
        np.random.seed(seed)
        self.steps = 0
        raw = self.env.reset()
        camera_id = self.env.sim.model.camera_name2id("foodview")
        center = self.env.sim.data.body_xpos[self.env.cube_body_id]
        # Explicit oracle camera localization; fixed throughout this episode.
        from scipy.spatial.transform import Rotation

        offset = np.array([0.0, -0.16, 0.25])
        self.env.sim.model.cam_pos[camera_id] = center + offset
        z_axis = offset / np.linalg.norm(offset)
        x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        quat_xyzw = Rotation.from_matrix(
            np.column_stack((x_axis, y_axis, z_axis))
        ).as_quat()
        self.env.sim.model.cam_quat[camera_id] = quat_xyzw[[3, 0, 1, 2]]
        self.env.sim.forward()
        self.down_quat = raw["robot0_eef_quat"].copy()
        self.env.monitor.reset(self.physical_state())
        return self._obs(raw)

    def physical_state(self):
        env = self.env
        center = env.sim.data.body_xpos[env.cube_body_id].copy()
        rotation = env.sim.data.body_xmat[env.cube_body_id].reshape(3, 3)
        extent = np.abs(rotation) @ np.asarray(env.cube.size)
        held = env._check_grasp(env.robots[0].gripper, env.cube.contact_geoms)
        fingers = env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]
        return PhysicalState(
            center,
            extent,
            float(np.linalg.norm(env.sim.data.get_body_xvelp(env.cube.root_body))),
            bool(held),
            bool(np.abs(fingers).sum() > 0.06),
        )

    def scene(self):
        return Scene(
            self.physical_state().object_center,
            {
                name: p + [0, 0, self.env.cube.size[2] + 0.002]
                for name, p in self.env.centers.items()
            },
            np.zeros(3),
            np.array([0.0, 0.0, 0.025]),
            self.down_quat,
            "simulator_oracle",
        )

    def food_crop(self, observation):
        if self.perception_view == "closeup":
            size = self.perception_size
            rgb = np.ascontiguousarray(
                self.env.sim.render(camera_name="foodview", height=size, width=size)[
                    ::-1
                ]
            )
            raw_seg = self.env.sim.render(
                camera_name="foodview", height=size, width=size, segmentation=True
            )[::-1]
            segmentation = raw_seg[..., 1]  # MuJoCo returns (object type, object ID).
            camera_name = "foodview"
        else:
            rgb = observation.image
            segmentation = self.latest_raw["agentview_segmentation_element"][
                ::-1
            ].squeeze()
            camera_name = "agentview"
        geom_ids = [
            self.env.sim.model.geom_name2id(name) for name in self.env.cube.visual_geoms
        ]
        ys, xs = np.where(np.isin(segmentation, geom_ids))
        if not len(xs):
            raise ValueError(f"Food proxy not visible in {camera_name} segmentation")
        pad = 8
        h, w = rgb.shape[:2]
        bounds = [
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + pad + 1),
            min(h, int(ys.max()) + pad + 1),
        ]
        x1, y1, x2, y2 = bounds
        crop = rgb[y1:y2, x1:x2].copy()
        self.last_perception_frame = rgb.copy()
        self.last_perception_metadata = {
            "camera": camera_name,
            "frame_size_wh": [w, h],
            "bbox_xyxy": bounds,
            "object_bbox_xyxy": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ],
            "object_visible_pixels": int(len(xs)),
            "crop_size_wh": [x2 - x1, y2 - y1],
            "localization_source": "simulator_segmentation",
            "camera_target_source": "simulator_object_position_at_reset"
            if camera_name == "foodview"
            else "fixed_camera",
            "camera_position": self.env.sim.model.cam_pos[
                self.env.sim.model.camera_name2id(camera_name)
            ].tolist(),
            "camera_quat_wxyz": self.env.sim.model.cam_quat[
                self.env.sim.model.camera_name2id(camera_name)
            ].tolist(),
            "camera_fovy": float(
                self.env.sim.model.cam_fovy[
                    self.env.sim.model.camera_name2id(camera_name)
                ]
            ),
            "time_s": observation.time_s,
        }
        self.last_crop = crop
        return crop

    def step(self, action):
        raw, _, done, _ = self.env.step(action)
        self.steps += 1
        events = self.env.monitor.update(self.physical_state())
        events.terminated |= bool(done)
        return self._obs(raw), events

    def close(self):
        self.env.close()
