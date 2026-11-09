import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.perception.lane_detection import ROAD_LINE_TAG, ROAD_TAG, LaneDetector, fit_lanes

CAMERA_HEIGHT_M = 1.7


def pixels_per_meter_at_bottom(width: int, height: int, fov_deg: float,
                               camera_height_m: float = CAMERA_HEIGHT_M) -> float:
    focal_px = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    ray_angle = math.atan((height / 2.0) / focal_px)
    lookahead_m = camera_height_m / math.tan(ray_angle)
    return focal_px / lookahead_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lane detection against semantic seg")
    parser.add_argument("--session", required=True, nargs="+")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-overlay", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="include sessions aborted early (diagnostics only; not an M1 verdict)")
    parser.add_argument("--roi-top", type=float, default=0.55)
    return parser.parse_args()


def draw_fit(image: np.ndarray, fit: np.ndarray, y0: int, roi_height: int,
             scale_x: float, scale_y: float, color: tuple[int, int, int]) -> None:
    ys = np.linspace(0, roi_height - 1, 30)
    xs = np.polyval(fit, ys)
    points = np.stack([xs * scale_x, (ys + y0) * scale_y], axis=1).astype(np.int32)
    cv2.polylines(image, [points], False, color, 2)


def evaluate_session(session: str, args: argparse.Namespace) -> dict:
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text())
    rgb_w = meta["rgb_width"]
    rgb_h = meta["rgb_height"]
    aux_w = meta["aux_width"]
    aux_h = meta["aux_height"]
    scale_x = rgb_w / aux_w
    scale_y = rgb_h / aux_h

    ego_rows = {}
    has_lane_gt = False
    csv_path = session / "ego.csv"
    if csv_path.exists():
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            has_lane_gt = "lane_offset_m" in (reader.fieldnames or [])
            has_lane_gt_valid = "lane_gt_valid" in (reader.fieldnames or [])
            for row in reader:
                ego_rows[int(row["index"])] = row

    frame_paths = sorted((session / "rgb").glob("*.png"))
    frame_paths = frame_paths[args.start:]
    if args.limit:
        frame_paths = frame_paths[: args.limit]
    if not frame_paths:
        raise SystemExit(f"no frames in {session / 'rgb'}")

    detector = LaneDetector(roi_top=args.roi_top)
    overlay_dir = session / "eval"
    if args.save_overlay or args.save_video:
        overlay_dir.mkdir(exist_ok=True)
    video = None
    if args.save_video:
        fps = 1.0 / float(meta.get("fixed_delta", 0.05))
        video = cv2.VideoWriter(
            str(overlay_dir / "overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (rgb_w, rgb_h),
        )
        if not video.isOpened():
            raise RuntimeError(f"could not create {overlay_dir / 'overlay.mp4'}")

    y0_aux = int(aux_h * args.roi_top)
    roi_h_aux = aux_h - y0_aux
    y0_rgb = int(rgb_h * args.roi_top)
    roi_h_rgb = rgb_h - y0_rgb
    aux_scale = aux_w / 192
    aux_fit = {
        "n_windows": 6,
        "margin": max(1, round(20 * aux_scale)),
        "min_pixels": max(1, round(5 * aux_scale)),
        "min_base_pixels": max(1, round(3 * aux_scale)),
    }

    tp = fp = fn = 0
    frame_count = 0
    pred_detected = 0
    gt_valid = 0
    compared = 0
    errors: list[float] = []
    seg_within15 = 0
    map_errors: list[float] = []
    map_gt_frames = 0
    map_within15 = 0
    map_px_per_meter = pixels_per_meter_at_bottom(
        rgb_w, rgb_h, float(meta.get("fov", 90)), CAMERA_HEIGHT_M
    )

    for i, rgb_path in enumerate(frame_paths):
        bgr = cv2.imread(str(rgb_path))
        tags = cv2.imread(str(session / "seg" / rgb_path.name), cv2.IMREAD_GRAYSCALE)
        if bgr is None or tags is None:
            continue
        frame_count += 1

        road = ((tags == ROAD_TAG) | (tags == ROAD_LINE_TAG)).astype(np.uint8) * 255
        road = cv2.resize(road, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        road = cv2.dilate(road, np.ones((5, 5), np.uint8))
        state = detector.detect(bgr, road_mask=road)
        pred_detected += int(state.detected)

        if has_lane_gt:
            row = ego_rows.get(int(rgb_path.stem))
            valid = row is not None
            if valid and has_lane_gt_valid:
                valid = row.get("lane_gt_valid", "0") == "1"
            if valid and row.get("lane_offset_m", "") not in ("", "nan"):
                map_gt_frames += 1
                gt_px = float(row["lane_offset_m"]) * map_px_per_meter
                if state.detected:
                    error = abs(state.offset_px - gt_px)
                    map_errors.append(error)
                    map_within15 += int(error < 15.0)

        pred_aux = cv2.resize(state.mask, (aux_w, aux_h), interpolation=cv2.INTER_AREA) > 64
        gt_aux = tags == ROAD_LINE_TAG
        pred_roi = pred_aux[y0_aux:]
        gt_roi = gt_aux[y0_aux:]
        tp += int(np.logical_and(pred_roi, gt_roi).sum())
        fp += int(np.logical_and(pred_roi, ~gt_roi).sum())
        fn += int(np.logical_and(~pred_roi, gt_roi).sum())

        left_gt, right_gt = fit_lanes(
            gt_aux.astype(np.uint8) * 255, args.roi_top, **aux_fit
        )
        if left_gt is not None and right_gt is not None:
            gt_valid += 1
            if state.left_fit is not None and state.right_fit is not None:
                compared += 1
                gt_center = 0.5 * (
                    float(np.polyval(left_gt, roi_h_aux - 1))
                    + float(np.polyval(right_gt, roi_h_aux - 1))
                )
                pred_center_rgb = 0.5 * (
                    float(np.polyval(state.left_fit, roi_h_rgb - 1))
                    + float(np.polyval(state.right_fit, roi_h_rgb - 1))
                )
                pred_center = pred_center_rgb / scale_x
                error = abs(pred_center - gt_center) * scale_x
                errors.append(error)
                seg_within15 += int(error < 15.0)
                if args.verbose:
                    print(f"  {rgb_path.stem} pred=True offset={state.offset_px:+.1f}px err={error:.1f}px")
            elif args.verbose:
                print(f"  {rgb_path.stem} pred=False gt_valid=True")

        if i < args.save_overlay or video is not None:
            visual = bgr.copy()
            visual[cv2.resize(gt_aux.astype(np.uint8) * 255, (rgb_w, rgb_h),
                              interpolation=cv2.INTER_NEAREST) > 0] = (0, 0, 255)
            visual[state.mask > 0] = (0, 255, 0)
            if state.left_fit is not None:
                draw_fit(visual, state.left_fit, y0_rgb, roi_h_rgb, 1.0, 1.0, (255, 0, 0))
            if state.right_fit is not None:
                draw_fit(visual, state.right_fit, y0_rgb, roi_h_rgb, 1.0, 1.0, (255, 0, 0))
            if left_gt is not None:
                draw_fit(visual, left_gt, y0_aux, roi_h_aux, scale_x, scale_y, (255, 255, 0))
            if right_gt is not None:
                draw_fit(visual, right_gt, y0_aux, roi_h_aux, scale_x, scale_y, (255, 255, 0))
            cv2.putText(visual, f"idx {rgb_path.stem} offset {state.offset_px:+.1f}px", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if i < args.save_overlay:
                cv2.imwrite(str(overlay_dir / rgb_path.name), visual)
            if video is not None:
                video.write(visual)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(frame_paths)} frames", flush=True)

    if video is not None:
        video.release()

    total = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / total if total else 0.0
    frame_count = max(frame_count, 1)

    print(f"{session}: frames={frame_count} pred_detected={pred_detected} "
          f"({pred_detected / frame_count:.1%}) gt_fit_valid={gt_valid} "
          f"compared={compared} ({compared / frame_count:.1%})")
    print(f"mask precision={precision:.3f} recall={recall:.3f} IoU={iou:.3f}")
    verdict = "FAIL"
    if errors:
        values = np.asarray(errors)
        within = float((values < 15.0).mean() * 100.0)
        print(f"seg center error px (n={len(values)}): mean={values.mean():.1f} "
              f"median={np.median(values):.1f} p90={np.percentile(values, 90):.1f} "
              f"within15px among compared={within:.1f}%")
    else:
        print("seg center error: no frames with both predictions and ground truth")
    seg_coverage = gt_valid / frame_count
    detector_coverage = compared / gt_valid if gt_valid else 0.0
    seg_score = seg_within15 / gt_valid if gt_valid else 0.0
    print(f"seg GT valid={gt_valid}/{frame_count} ({seg_coverage:.1%}) | "
          f"predicted centerline on valid GT={compared}/{gt_valid} "
          f"({detector_coverage:.1%}) | within15 including misses={seg_score:.1%}")
    verdict = "PASS" if gt_valid > 0 and seg_score >= 0.80 else "FAIL"
    if has_lane_gt:
        gt_coverage = map_gt_frames / frame_count * 100.0
        if map_errors:
            values = np.asarray(map_errors)
            within = float((values < 15.0).mean() * 100.0)
            print(f"map GT center error px (n={len(values)}): mean={values.mean():.1f} "
                  f"median={np.median(values):.1f} p90={np.percentile(values, 90):.1f} "
                  f"within15px={within:.1f}% detected coverage={len(values) / frame_count:.1%}")
        else:
            print("map GT center error: detector produced no centerline")
        print(f"map GT valid={map_gt_frames}/{frame_count} ({gt_coverage:.1f}%) | "
              f"within15 over all frames={map_within15 / frame_count:.1%}")
    print("M1 seg-GT target (>=80% valid-GT frames <15px, misses fail): " + verdict)
    return {
        "session": str(session),
        "frames": frame_count,
        "pred_detected": pred_detected,
        "gt_valid": gt_valid,
        "compared": compared,
        "seg_within15": seg_within15,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "center_error_mean": float(np.mean(errors)) if errors else None,
        "center_within15": seg_within15 / gt_valid if gt_valid else 0.0,
        "gt_coverage": gt_valid / frame_count,
        "detector_coverage": compared / gt_valid if gt_valid else 0.0,
        "seg_errors": errors,
        "has_lane_gt": has_lane_gt,
        "map_gt_frames": map_gt_frames,
        "map_within15": map_within15,
        "map_errors": map_errors,
    }


def main() -> None:
    args = parse_args()
    sessions = []
    for session in args.session:
        meta = json.loads((Path(session) / "meta.json").read_text())
        requested = meta.get("requested_frames")
        complete = meta.get("complete")
        if complete is None:
            complete = not meta.get("aborted") and (
                requested is None or meta.get("frames", 0) >= requested
            )
        if not complete and not args.include_incomplete:
            reason = ",".join(meta.get("incomplete_reasons", [])) or meta.get("aborted") or "incomplete"
            print(f"SKIP incomplete session (reason={reason}, "
                  f"frames={meta.get('frames')}/{requested}): {session}")
            continue
        sessions.append(session)
    if not sessions:
        raise SystemExit("no complete sessions to evaluate; record a full session or use --include-incomplete for diagnostics")

    results = [evaluate_session(session, args) for session in sessions]
    frames = sum(result["frames"] for result in results)
    tp = sum(result["tp"] for result in results)
    fp = sum(result["fp"] for result in results)
    fn = sum(result["fn"] for result in results)
    pred_detected = sum(result["pred_detected"] for result in results)
    map_errors = [error for result in results for error in result["map_errors"]]
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"COMBINED sessions={len(results)} frames={frames} "
          f"pred_detected={pred_detected} ({pred_detected / max(frames, 1):.1%})")
    print(f"COMBINED mask precision={precision:.3f} recall={recall:.3f} IoU={iou:.3f}")

    gt_valid = sum(result["gt_valid"] for result in results)
    compared = sum(result["compared"] for result in results)
    within15 = sum(result["seg_within15"] for result in results)
    seg_score = within15 / gt_valid if gt_valid else 0.0
    print(f"COMBINED seg GT valid={gt_valid}/{frames} ({gt_valid / max(frames, 1):.1%}) | "
          f"predicted centerline on valid GT={compared}/{gt_valid} "
          f"({compared / gt_valid if gt_valid else 0.0:.1%}) | "
          f"within15 including misses={within15}/{gt_valid} ({seg_score:.1%})")
    if map_errors:
        values = np.asarray(map_errors)
        map_gt_frames = sum(result["map_gt_frames"] for result in results)
        map_within15 = sum(result["map_within15"] for result in results)
        print(f"COMBINED map GT diagnostic: valid={map_gt_frames}/{frames}, "
              f"detected comparisons={len(values)}, within15px="
              f"{float((values < 15).mean()):.1%} conditional, "
              f"within15 over all frames={map_within15 / max(frames, 1):.1%}")
    verdict = "PASS" if gt_valid > 0 and seg_score >= 0.80 else "FAIL"
    print("COMBINED M1 seg-GT target (>=80% valid-GT frames <15px, misses fail): " + verdict)
    print("\nMetric table (per session)")
    print("session | precision | recall | IoU | center MAE px | <15px incl misses | GT coverage | detector coverage")
    for result in results:
        name = Path(result["session"]).name
        mae = "n/a" if result["center_error_mean"] is None else f"{result['center_error_mean']:.1f}"
        print(f"{name} | {result['precision']:.3f} | {result['recall']:.3f} | "
              f"{result['iou']:.3f} | {mae} | {result['center_within15']:.1%} | "
              f"{result['gt_coverage']:.1%} | {result['detector_coverage']:.1%}")
    print(f"ALL ({len(results)} sessions) | {precision:.3f} | {recall:.3f} | "
          f"{iou:.3f} | n/a | {seg_score:.1%} | {gt_valid / max(frames, 1):.1%} | "
          f"{compared / gt_valid if gt_valid else 0.0:.1%}")


if __name__ == "__main__":
    main()
