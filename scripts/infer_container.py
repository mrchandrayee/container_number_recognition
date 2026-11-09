from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
DEFAULT_WEIGHTS = "weights/best.pt"
DEFAULT_DATA = "data/container.yaml"
DEFAULT_SOURCE = "sample.mp4"
DEFAULT_OUTPUT = "runs/container_inference"
OCR_ENGINES = ("tesseract", "easyocr")
OCR_ENGINE_ALIASES = {"tess": "tesseract", "easy": "easyocr"}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 960
DEFAULT_OCR_CONF = 25.0
DEFAULT_CROP_PADDING = 0.62
DEFAULT_PANEL_X = 12
DEFAULT_PANEL_WIDTH = 310
DEFAULT_TRAIL_LENGTH = 30
DEFAULT_MAX_CARDS = 6
BBOX_OCR_FONT_SCALE = 0.48
BBOX_OCR_FONT_THICKNESS = 1


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None


@dataclass
class PipelineState:
    tracker: str
    trail_length: int
    max_cards: int
    ocr_polygon: Optional[np.ndarray]
    hud_panel_x: int
    hud_panel_width: int


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for container detector validation and inference."""
    parser = argparse.ArgumentParser(
        prog="python scripts/infer_container.py",
        description="Validate or infer with a trained container YOLO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained weights (best.pt)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML (used with --val)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image, folder, or video for visual inference (ignored with --val)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output folder for saved predictions")
    parser.add_argument("--val", action="store_true", help="Run Ultralytics validation instead of inference")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=None, help="Device such as 0, cpu, or mps")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--no-track", action="store_true", help="Disable object tracking")
    parser.add_argument("--trail-length", type=int, default=DEFAULT_TRAIL_LENGTH)
    parser.add_argument("--max-cards", type=int, default=DEFAULT_MAX_CARDS, help="Maximum number of HUD records")
    parser.add_argument("--ocr-engine", default="tesseract", help="OCR engine: tesseract or easyocr")
    parser.add_argument("--ocr-conf", type=float, default=DEFAULT_OCR_CONF, help="Minimum OCR confidence (0-100)")
    parser.add_argument("--ocr-zone", default="", help="OCR polygon as x1,y1,x2,y2,... (full-frame coords). Empty = all detections")
    parser.add_argument("--no-ocr-zone", action="store_true", help="Disable OCR polygon filter (OCR every detection)")
    parser.add_argument("--crop", action="store_true", help="Crop source frame before inference")
    parser.add_argument("--crop-rect", default="", help="Crop rectangle x1,y1,x2,y2")
    parser.add_argument("--hud-x", type=int, default=DEFAULT_PANEL_X, help="HUD panel left edge")
    parser.add_argument("--hud-width", type=int, default=DEFAULT_PANEL_WIDTH, help="HUD panel width")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video output")
    parser.add_argument("--show", action="store_true", help="Display annotated frames")
    parser.add_argument("--dry-run", action="store_true", help="Log config and exit")
    return parser


def normalize_ocr_engine(value: str) -> str:
    """Normalize OCR engine CLI value."""
    engine = OCR_ENGINE_ALIASES.get(value.lower(), value.lower())
    if engine not in OCR_ENGINES:
        raise argparse.ArgumentTypeError(
            f"ocr-engine must be one of: {', '.join(OCR_ENGINES)}; aliases: {', '.join(OCR_ENGINE_ALIASES)}"
        )
    return engine


def is_video_source(source: Path) -> bool:
    """Return True when source is a video file."""
    return source.is_file() and source.suffix.lower() in VIDEO_EXTS


def parse_crop_rect(value: str) -> tuple[int, int, int, int]:
    """Parse 'x1,y1,x2,y2' crop rectangle."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(part) for part in parts)
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("crop requires x2>x1 and y2>y1")
    return x1, y1, x2, y2


def parse_polygon_points(value: str) -> tuple[tuple[int, int], ...]:
    """Parse 'x1,y1,x2,y2,...' into polygon vertices."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise argparse.ArgumentTypeError("ocr-zone must have at least 3 points as x,y pairs")
    points = []
    for index in range(0, len(parts), 2):
        points.append((int(parts[index]), int(parts[index + 1])))
    return tuple(points)


def polygon_for_frame(
    points: tuple[tuple[int, int], ...],
    *,
    crop_rect: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Map full-frame polygon points into the current inference frame."""
    offset_x = crop_rect[0] if crop_rect is not None else 0
    offset_y = crop_rect[1] if crop_rect is not None else 0
    return np.array([(x - offset_x, y - offset_y) for x, y in points], dtype=np.int32)


def bbox_center_in_polygon(
    bbox: tuple[float, float, float, float],
    polygon: Optional[np.ndarray],
) -> bool:
    """Return True when bbox center lies inside polygon."""
    if polygon is None:
        return True
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0


def filter_detections_in_ocr_zone(
    detections: list[Detection],
    polygon: Optional[np.ndarray],
) -> list[Detection]:
    """Keep detections whose center is inside the OCR polygon."""
    if polygon is None:
        return detections
    return [det for det in detections if bbox_center_in_polygon(det.bbox, polygon)]


def hud_panel_origin(
    frame_width: int,
    *,
    hud_panel_x: int,
    hud_panel_width: int,
    crop_rect: Optional[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Resolve HUD panel (x, width) in the current frame, clamped to fit."""
    offset_x = crop_rect[0] if crop_rect is not None else 0
    panel_x = hud_panel_x - offset_x
    panel_width = min(hud_panel_width, frame_width)
    panel_x = max(0, min(panel_x, max(0, frame_width - panel_width)))
    return panel_x, panel_width


def build_pipeline_state(
    *,
    enable_track: bool,
    trail_length: int,
    hud_max_cards: int,
    hud_panel_x: int,
    hud_panel_width: int,
    ocr_polygon_points: Optional[tuple[tuple[int, int], ...]],
    crop_rect: Optional[tuple[int, int, int, int]],
    use_crop: bool,
) -> PipelineState:
    """Create pipeline state with OCR zone and HUD layout for one run."""
    crop_for_coords = crop_rect if use_crop else None
    polygon = (
        polygon_for_frame(ocr_polygon_points, crop_rect=crop_for_coords)
        if ocr_polygon_points is not None
        else None
    )
    return PipelineState(
        tracker="bytetrack.yaml" if enable_track else "",
        trail_length=trail_length,
        max_cards=hud_max_cards,
        ocr_polygon=polygon,
        hud_panel_x=hud_panel_x,
        hud_panel_width=hud_panel_width,
    )


def draw_hud_field(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str,
    value: str,
    label_color: tuple[int, int, int],
    value_color: tuple[int, int, int],
    label_scale: float,
    value_scale: float,
) -> None:
    """Draw one HUD row with a small label and larger value text."""
    label_text = f"{label}:"
    cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                label_scale, label_color, 1, cv2.LINE_AA)
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
    cv2.putText(frame, value, (x + label_size[0][0] + 6, y),
                cv2.FONT_HERSHEY_SIMPLEX, value_scale, value_color, 1, cv2.LINE_AA)


def draw_ocr_bbox_label(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    text: str,
    *,
    font_scale: float = BBOX_OCR_FONT_SCALE,
    thickness: int = BBOX_OCR_FONT_THICKNESS,
) -> None:
    """Draw a high-contrast OCR label above a box."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad_x, pad_y = 5, 4
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_h = text_h + baseline + pad_y * 2
    box_w = text_w + pad_x * 2
    y_top = max(0, y1 - box_h - 4)
    x_left = max(0, x1)
    x_right = min(frame.shape[1] - 1, x_left + box_w)
    y_bottom = min(frame.shape[0] - 1, y_top + box_h)
    cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (0, 0, 0), -1)
    cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (255, 255, 255), 1)
    text_x = x_left + pad_x
    text_y = y_bottom - pad_y - baseline
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def crop_rect(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a BGR image to (x1,y1,x2,y2), clamped to image bounds."""
    x1, y1, x2, y2 = rect
    height, width = img.shape[:2]
    left = max(0, min(x1, width))
    right = max(0, min(x2, width))
    top = max(0, min(y1, height))
    bottom = max(0, min(y2, height))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop rect {rect} for image size {width}x{height}")
    return img[top:bottom, left:right].copy()


def crop_container_region(
    frame: np.ndarray,
    detection: Detection,
    *,
    padding: float = DEFAULT_CROP_PADDING,
) -> tuple[np.ndarray, int, int]:
    """Crop a container bbox with padding. Returns crop, left offset, top offset."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    box_w = max(2.0, x2 - x1)
    box_h = max(2.0, y2 - y1)
    pad_x = box_w * padding
    pad_y = box_h * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    crop = frame[top:bottom, left:right]
    return crop, left, top


def normalize_container_code(text: str) -> str:
    """Normalize OCR text to ISO-style container-code characters."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    # Preserve alphanumerics; final ISO 6346 validation is intentionally not forced.
    return cleaned[:16]


def assemble_text(detections: list[Detection]) -> str:
    """Join detected character classes left-to-right by bbox center x."""
    ordered = sorted(detections, key=lambda det: det.center[0])
    return normalize_container_code("".join(det.class_name for det in ordered))


def group_ocr_lines(results: list[OCRResult], min_conf: float) -> list[list[OCRResult]]:
    """Group OCR results into horizontal lines (top to bottom)."""
    kept = [result for result in results if result.confidence >= min_conf and result.text.strip()]
    with_bbox = [result for result in kept if result.bbox is not None]
    if not with_bbox:
        return [kept] if kept else []
    heights = [max(1, result.bbox[3] - result.bbox[1]) for result in with_bbox]
    line_tol = max(12.0, float(np.median(heights)) * 0.8)
    buckets: list[list[OCRResult]] = []
    for result in sorted(with_bbox, key=lambda item: (item.bbox[1] + item.bbox[3]) / 2):
        center_y = (result.bbox[1] + result.bbox[3]) / 2
        target = None
        for bucket in buckets:
            bucket_y = np.mean([(r.bbox[1] + r.bbox[3]) / 2 for r in bucket])
            if abs(center_y - bucket_y) <= line_tol:
                target = bucket
                break
        if target is None:
            buckets.append([result])
        else:
            target.append(result)
    return [sorted(bucket, key=lambda item: item.bbox[0]) for bucket in buckets]


class OCRReader:
    def __init__(self, engine: str):
        self.engine = engine
        self.reader = None
        if engine == "tesseract":
            try:
                import pytesseract  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Install pytesseract and the Tesseract OCR system package.") from exc
        elif engine == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(["en"], gpu=False)
            except ImportError as exc:
                raise RuntimeError("Install easyocr with: pip install easyocr") from exc

    def read(self, image: np.ndarray) -> list[OCRResult]:
        if image.size == 0:
            return []
        if self.engine == "tesseract":
            import pytesseract
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(
                rgb, config="--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                output_type=pytesseract.Output.DICT,
            )
            results = []
            for i, raw in enumerate(data["text"]):
                text = normalize_container_code(raw)
                try:
                    confidence = float(data["conf"][i])
                except (TypeError, ValueError):
                    confidence = -1
                if text and confidence >= 0:
                    x, y = int(data["left"][i]), int(data["top"][i])
                    w, h = int(data["width"][i]), int(data["height"][i])
                    results.append(OCRResult(text, confidence, (x, y, x + w, y + h)))
            return results

        raw_results = self.reader.readtext(image)
        results = []
        for polygon, text, confidence in raw_results:
            text = normalize_container_code(text)
            if not text:
                continue
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
            results.append(OCRResult(text, float(confidence) * 100,
                                     (min(xs), min(ys), max(xs), max(ys))))
        return results


def read_container_number(reader: OCRReader, frame: np.ndarray, detection: Detection,
                          min_conf: float) -> tuple[str, float]:
    crop, _, _ = crop_container_region(frame, detection)
    if crop.size == 0:
        return "", 0.0
    # Enlarge small text and improve local contrast.
    scale = 2 if max(crop.shape[:2]) < 1200 else 1
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lines = group_ocr_lines(reader.read(processed), min_conf)
    candidates = []
    for line in lines:
        value = normalize_container_code("".join(item.text for item in line))
        if len(value) >= 4:
            confidence = float(np.mean([item.confidence for item in line]))
            candidates.append((value, confidence))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (len(item[0]) >= 10, item[1], len(item[0])))


def extract_detections(result) -> list[Detection]:
    detections = []
    if result.boxes is None:
        return detections
    names = result.names
    for box in result.boxes:
        coords = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().item())
        class_id = int(box.cls[0].cpu().item())
        track_id = int(box.id[0].cpu().item()) if box.id is not None else None
        class_name = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])
        detections.append(Detection(tuple(float(v) for v in coords), conf,
                                    class_id, class_name, track_id))
    return detections


def draw_hud(frame: np.ndarray, records: list[tuple[str, float]], fps: float,
             panel_x: int, panel_width: int, max_cards: int) -> None:
    h, w = frame.shape[:2]
    panel_width = min(panel_width, w)
    x = max(0, min(panel_x, w - panel_width))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, 0), (x + panel_width, min(h, 44 + max_cards * 31)),
                  (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"CONTAINERS  |  FPS {fps:.1f}", (x + 9, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
    for idx, (code, conf) in enumerate(records[:max_cards]):
        label = code if code else "Reading..."
        cv2.putText(frame, f"{idx + 1}. {label}  {conf:.0f}%",
                    (x + 9, 48 + idx * 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (90, 255, 130), 1, cv2.LINE_AA)


def process_frame(frame: np.ndarray, model: YOLO, reader: OCRReader, args,
                  state: PipelineState, frame_index: int):
    inference_frame = crop_rect(frame, args.crop_rect) if args.crop else frame.copy()
    predict_kwargs = dict(source=inference_frame, imgsz=args.imgsz, conf=args.conf,
                          iou=args.iou, device=args.device, verbose=False)
    if args.no_track:
        result = model.predict(**predict_kwargs)[0]
    else:
        result = model.track(**predict_kwargs, persist=True, tracker=args.tracker)[0]

    detections = extract_detections(result)
    detections = filter_detections_in_ocr_zone(detections, state.ocr_polygon)
    annotated = result.plot()
    if args.crop:
        # Place crop annotation back onto a copy of the original frame.
        x1, y1, x2, y2 = args.crop_rect
        x1, y1 = max(0, x1), max(0, y1)
        canvas = frame.copy()
        hh, ww = annotated.shape[:2]
        x2, y2 = min(frame.shape[1], x1 + ww), min(frame.shape[0], y1 + hh)
        canvas[y1:y2, x1:x2] = annotated[:y2-y1, :x2-x1]
        annotated = canvas
        offset_x, offset_y = x1, y1
    else:
        offset_x = offset_y = 0

    records = []
    for det in detections:
        code, ocr_conf = read_container_number(reader, inference_frame, det, args.ocr_conf)
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        x1, x2 = x1 + offset_x, x2 + offset_x
        y1, y2 = y1 + offset_y, y2 + offset_y
        color = (0, 220, 80) if code else (0, 190, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if code:
            draw_ocr_bbox_label(annotated, x1, y1, x2, code)
        records.append((code, ocr_conf))
    fps = 0.0
    draw_hud(annotated, records, fps, args.hud_x, args.hud_width, state.max_cards)
    return annotated, records


def iter_image_files(source: Path):
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        yield source
    else:
        raise FileNotFoundError(f"Image source not found or unsupported: {source}")


def run_inference(args) -> None:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Place your trained YOLO best.pt at weights/best.pt or pass --weights /path/to/best.pt."
        )
    model = YOLO(str(weights))
    reader = OCRReader(args.ocr_engine)
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    crop_rect_value = args.crop_rect
    if args.crop and not crop_rect_value:
        raise ValueError("--crop requires --crop-rect x1,y1,x2,y2")
    args.crop_rect = parse_crop_rect(crop_rect_value) if crop_rect_value else None

    if args.no_ocr_zone or not args.ocr_zone:
        polygon_points = None
    else:
        polygon_points = parse_polygon_points(args.ocr_zone)
    state = build_pipeline_state(
        enable_track=not args.no_track,
        trail_length=args.trail_length,
        hud_max_cards=args.max_cards,
        hud_panel_x=args.hud_x,
        hud_panel_width=args.hud_width,
        ocr_polygon_points=polygon_points,
        crop_rect=args.crop_rect,
        use_crop=args.crop,
    )

    if source.is_file() and source.suffix.lower() in IMAGE_EXTS or source.is_dir():
        for index, image_path in enumerate(iter_image_files(source)):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[WARN] Could not read {image_path}")
                continue
            annotated, records = process_frame(frame, model, reader, args, state, index)
            target = output / f"{image_path.stem}_recognized.jpg"
            cv2.imwrite(str(target), annotated)
            print(f"[{image_path.name}] {records} -> {target}")
            if args.show:
                cv2.imshow("Container Number Recognition", annotated)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
        return

    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
    elif str(args.source).isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        raise FileNotFoundError(f"Unsupported source: {source}")

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/camera source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save_video:
        target = output / "annotated_output.mp4"
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
    frame_index = 0
    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, records = process_frame(frame, model, reader, args, state, frame_index)
            elapsed = max(time.time() - start, 1e-6)
            cv2.putText(annotated, f"FPS: {(frame_index + 1) / elapsed:.1f}",
                        (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Container Number Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames. Output: {output}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.ocr_engine = normalize_ocr_engine(args.ocr_engine)
        if args.dry_run:
            print("Container Number Recognition configuration")
            for key, value in vars(args).items():
                print(f"{key}: {value}")
            return 0
        if args.val:
            if not Path(args.weights).exists():
                raise FileNotFoundError(f"Weights not found: {args.weights}")
            model = YOLO(args.weights)
            metrics = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=args.device)
            print(metrics)
        else:
            run_inference(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
