import argparse
import time
import carla
import cv2
import numpy as np

from configs.config import CFG
from src.sensors import SensorManager
from src.vehicle import EgoVehicle

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA Phase 1 harness")
    parser.add_argument("--host", default=CFG.host)
    parser.add_argument("--port", type=int, default=CFG.port)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = chạy đến khi Ctrl+C")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def build_view(frame, fps: float) -> np.ndarray:
    rgb = frame.rgb
    seg = cv2.resize(frame.seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    view = np.hstack([rgb, seg])
    text = (f"fps {fps:4.1f} | speed {frame.speed_kmh:5.1f} km/h | "
            f"collisions {frame.collisions} | lane_inv {frame.lane_invasions}")
    cv2.putText(view, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def main() -> None:
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(CFG.timeout)
    world = client.get_world()
    print(f"connected {args.host}:{args.port} | map {world.get_map().name}")

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = CFG.fixed_delta
    world.apply_settings(settings)

    ego = None
    sensors = None
    frames = 0
    start = time.monotonic()
    window_start = start
    window_frames = 0

    try:
        ego = EgoVehicle(world, CFG.vehicle_filter)
        sensors = SensorManager(world, ego.actor, CFG)
        print(f"ego {ego.actor.type_id} | target {1.0 / CFG.fixed_delta:.0f} Hz | "
              f"rgb {CFG.rgb_width}x{CFG.rgb_height} | aux {CFG.aux_width}x{CFG.aux_height}")
        start = time.monotonic()
        window_start = start
        window_frames = 0

        while True:
            if args.duration and time.monotonic() - start >= args.duration:
                break

            loop_start = time.perf_counter()
            previous_ids = sensors.frame_ids()
            world.tick()
            frame = sensors.wait_for_frame(previous_ids)

            if frame is None:
                print("WARN: sensor timeout, bỏ qua frame", flush=True)
            else:
                frames += 1
                window_frames += 1

                if not args.no_display:
                    now = time.monotonic()
                    fps = window_frames / max(now - window_start, 1e-6)
                    cv2.imshow("CARLA", build_view(frame, fps))
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break

                now = time.monotonic()
                if now - window_start >= 1.0:
                    fps = window_frames / (now - window_start)
                    print(f"fps={fps:4.1f} speed={frame.speed_kmh:5.1f} km/h | "
                          f"collisions={frame.collisions} lane_inv={frame.lane_invasions}", flush=True)
                    window_start, window_frames = now, 0

            elapsed = time.perf_counter() - loop_start
            if elapsed < CFG.fixed_delta:
                time.sleep(CFG.fixed_delta - elapsed)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if sensors is not None:
            sensors.close()
        if ego is not None:
            ego.destroy()
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = original.fixed_delta_seconds
            world.apply_settings(settings)
        except RuntimeError as exc:
            print(f"WARN: không khôi phục được settings: {exc}")
        if not args.no_display:
            cv2.destroyAllWindows()
        elapsed = time.monotonic() - start
        print(f"DONE frames={frames} elapsed={elapsed:.1f}s avg_fps={frames / max(elapsed, 1e-6):.1f}")


if __name__ == "__main__":
    main()
