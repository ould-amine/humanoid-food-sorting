"""Click points and draw food boxes in extracted frames (OpenCV GUI).

Left clicks cycle food -> hand -> target -> food on the same frame.
1/2/3 select food/hand/target. Tab clears the selected point and advances.
Right-drag draws a food
box. a/g/t/r label APPROACH/GRASP/TRANSPORT/RELEASE. n/p navigate, s saves,
c clears selected point, q saves and exits. No automatic interpolation.
"""

import argparse
import csv
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New CSV; input is never overwritten"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.annotations.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError("No extracted frames to annotate")
    for field in ("food_x1", "food_y1", "food_x2", "food_y2"):
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in fields:
            row.setdefault(field, "")
    root = args.frames_dir or args.annotations.parent
    index, selected, dragging = 0, "food", None
    point_order = ("food", "hand", "target")
    window = "Annotate: click food > hand > target; Tab skip; n next; s save; q quit"

    def advance_point():
        nonlocal selected
        selected = point_order[(point_order.index(selected) + 1) % len(point_order)]

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(args.output)

    def mouse(event, x, y, flags, userdata):
        nonlocal dragging
        row = rows[index]
        if event == cv2.EVENT_LBUTTONDOWN:
            row[f"{selected}_u"], row[f"{selected}_v"] = str(x), str(y)
            advance_point()
        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and dragging is not None:
            x1, x2 = sorted((dragging[0], x))
            y1, y2 = sorted((dragging[1], y))
            if x2 > x1 and y2 > y1:
                for field, value in zip(
                    ("food_x1", "food_y1", "food_x2", "food_y2"), (x1, y1, x2, y2)
                ):
                    row[field] = str(value)
            dragging = None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            row = rows[index]
            image_path = root / (
                row.get("image_path") or f"frame_{int(row['frame_index']):06d}.jpg"
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot open {image_path}")
            for name, color in (
                ("food", (0, 255, 0)),
                ("hand", (255, 0, 0)),
                ("target", (0, 0, 255)),
            ):
                if row.get(f"{name}_u") and row.get(f"{name}_v"):
                    point = (int(float(row[f"{name}_u"])), int(float(row[f"{name}_v"])))
                    cv2.circle(image, point, 5, color, -1)
                    cv2.putText(
                        image, name, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )
            if all(row.get(f"food_{k}") for k in ("x1", "y1", "x2", "y2")):
                box = [int(float(row[f"food_{k}"])) for k in ("x1", "y1", "x2", "y2")]
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)
            cv2.putText(
                image,
                f"{index + 1}/{len(rows)} Next: {selected} {row.get('phase', '')}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window, image)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("1"), ord("2"), ord("3")):
                selected = {ord("1"): "food", ord("2"): "hand", ord("3"): "target"}[key]
            elif (
                key == 9
            ):  # Tab: mark the selected point missing, even if previously placed.
                row[f"{selected}_u"] = row[f"{selected}_v"] = ""
                advance_point()
            elif key == ord("n"):
                index = min(index + 1, len(rows) - 1)
                selected, dragging = "food", None
            elif key == ord("p"):
                index = max(index - 1, 0)
                selected, dragging = "food", None
            elif key == ord("c"):
                row[f"{selected}_u"] = row[f"{selected}_v"] = ""
            elif key in (ord("a"), ord("g"), ord("t"), ord("r")):
                row["phase"] = {
                    ord("a"): "APPROACH",
                    ord("g"): "GRASP",
                    ord("t"): "TRANSPORT",
                    ord("r"): "RELEASE",
                }[key]
            elif key == ord("s"):
                save()
            elif (
                key == ord("q")
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                save()
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
