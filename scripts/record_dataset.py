import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

import src.agents_compat  # noqa: F401
from agents.navigation.local_planner import LocalPlanner, RoadOption
from configs.config import CFG
from src.sensors import SensorManager
from src.vehicle import EgoVehicle

WEATHER = {
    "clear": carla.WeatherParameters.ClearNoon,
    "cloudy": carla.WeatherParameters.CloudyNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "night": carla.WeatherParameters.ClearNight,
}


def _yaw_delta(a, b) -> float:
    a_yaw = math.radians(a.transform.rotation.yaw)
    b_yaw = math.radians(b.transform.rotation.yaw)
    return (b_yaw - a_yaw + math.pi) % (2 * math.pi) - math.pi


def _branch_yaw_delta(start, branch, lookahead_m: float = 12.0) -> float:
    current = branch
    traveled = 0.0
    while traveled < lookahead_m:
        options = list(current.next(2.0))
        if not options:
            break
        nxt = min(options, key=lambda wp: abs(_yaw_delta(current, wp)))
        traveled += current.transform.location.distance(nxt.transform.location)
        current = nxt
    return _yaw_delta(start, current)


def build_lane_plan(
    start_waypoint,
    length_m: float = 400.0,
    step_m: float = 2.0,
    route: str = "straight",
    route_info: dict | None = None,
):
    plan = [(start_waypoint, RoadOption.LANEFOLLOW)]
    current = start_waypoint
    traveled = 0.0
    route_choice_made = route == "straight"
    route_choice_distance = None
    route_choice_yaw = None
    while traveled < length_m:
        options = list(current.next(step_m))
        if not options:
            break
        if len(options) == 1:
            nxt = options[0]
        else:
            deltas = [_branch_yaw_delta(current, wp) for wp in options]
            if route_choice_made:
                selected = min(range(len(options)), key=lambda i: abs(deltas[i]))
            elif route == "curve":
                curved = [i for i, delta in enumerate(deltas)
                          if math.radians(10) <= abs(delta) <= math.radians(60)]
                if curved:
                    selected = min(curved, key=lambda i: abs(deltas[i]))
                    route_choice_made = True
                    route_choice_distance = traveled
                    route_choice_yaw = math.degrees(deltas[selected])
                else:
                    selected = min(range(len(options)), key=lambda i: abs(deltas[i]))
            else:
                turns = [i for i, delta in enumerate(deltas) if abs(delta) >= math.radians(10)]
                if turns:
                    selected = max(turns, key=lambda i: abs(deltas[i]))
                    route_choice_made = True
                    route_choice_distance = traveled
                    route_choice_yaw = math.degrees(deltas[selected])
                else:
                    selected = min(range(len(options)), key=lambda i: abs(deltas[i]))
            nxt = options[selected]
        traveled += current.transform.location.distance(nxt.transform.location)
        plan.append((nxt, RoadOption.LANEFOLLOW))
        current = nxt
    if route != "straight" and not route_choice_made:
        raise RuntimeError(f"no junction branch available for --route {route} from this spawn point")
    if route_info is not None:
        route_info.update({
            "choice_distance_m": route_choice_distance,
            "choice_yaw_deg": route_choice_yaw,
        })
    return plan


def route_profile(plan, route: str, distance_m: float = 15.0) -> dict:
    start = plan[0][0]
    traveled = 0.0
    max_yaw_change = 0.0
    has_junction = start.is_junction
    for (previous, _), (waypoint, _) in zip(plan, plan[1:]):
        traveled += previous.transform.location.distance(waypoint.transform.location)
        max_yaw_change = max(max_yaw_change, abs(_yaw_delta(start, waypoint)))
        has_junction = has_junction or waypoint.is_junction
        if traveled >= distance_m:
            break
    if route == "straight":
        valid = max_yaw_change < math.radians(10) and not has_junction
    elif route == "curve":
        valid = math.radians(12) <= max_yaw_change <= math.radians(60)
    else:
        valid = has_junction and max_yaw_change >= math.radians(10)
    return {
        "valid": valid,
        "distance_m": min(traveled, distance_m),
        "max_yaw_change_deg": math.degrees(max_yaw_change),
        "has_junction": has_junction,
    }


def camera_bottom_lookahead(cfg) -> float:
    focal_px = cfg.rgb_width / (2.0 * math.tan(math.radians(cfg.fov) / 2.0))
    ray_angle = math.atan((cfg.rgb_height / 2.0) / focal_px)
    return 1.7 / math.tan(ray_angle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RGB + seg + depth + ego dataset")
    parser.add_argument("--host", default=CFG.host)
    parser.add_argument("--port", type=int, default=CFG.port)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--weather", choices=sorted(WEATHER), default=None)
    parser.add_argument("--driver", choices=("tm", "local"), default="tm")
    parser.add_argument("--route", choices=("straight", "curve", "intersection"), default="straight")
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--list-spawns", action="store_true",
                        help="list spawn points with a valid route plan, then exit")
    parser.add_argument("--target-speed", type=float, default=20.0)
    parser.add_argument("--plan-length", type=float, default=400.0)
    parser.add_argument("--speed-difference", type=int, default=30)
    parser.add_argument("--stop-on-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stall-frames", type=int, default=60,
                        help="abort after this many consecutive frames below 1 km/h (0 disables)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(CFG.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    if args.list_spawns:
        for index, transform in enumerate(carla_map.get_spawn_points()):
            waypoint = carla_map.get_waypoint(
                transform.location, lane_type=carla.LaneType.Driving
            )
            info = {}
            try:
                plan = build_lane_plan(
                    waypoint, args.plan_length, route=args.route, route_info=info
                )
            except RuntimeError:
                continue
            profile = route_profile(plan, args.route)
            if not profile["valid"]:
                continue
            print(f"spawn={index} route={args.route} "
                  f"choice_distance_m={info['choice_distance_m']} "
                  f"choice_yaw_deg={info['choice_yaw_deg']} "
                  f"profile_yaw_deg={profile['max_yaw_change_deg']:.1f} "
                  f"junction={profile['has_junction']}")
        return
    original_settings = world.get_settings()
    original_weather = world.get_weather()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = CFG.fixed_delta
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    session_dir = Path(args.out) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    for name in ("rgb", "seg", "depth"):
        (session_dir / name).mkdir(parents=True, exist_ok=True)

    ego = None
    sensors = None
    planner = None
    route_info = {}
    written = 0
    start = None
    aborted = None
    low_speed_frames = 0
    low_speed_rows = []
    lane_gt_valid_count = 0
    observed_yaws = []
    observed_junctions = []
    try:
        if args.weather is not None:
            world.set_weather(WEATHER[args.weather])
        ego = EgoVehicle(world, CFG.vehicle_filter, spawn_index=args.spawn_index)
        sensors = SensorManager(world, ego.actor, CFG)
        start_waypoint = carla_map.get_waypoint(
            ego.actor.get_location(), lane_type=carla.LaneType.Driving
        )
        route_plan = build_lane_plan(
            start_waypoint, args.plan_length, route=args.route, route_info=route_info
        )
        if args.driver == "tm":
            ego.actor.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(ego.actor, args.speed_difference)
            traffic_manager.auto_lane_change(ego.actor, False)
            traffic_manager.set_desired_speed(ego.actor, args.target_speed)
            traffic_manager.set_path(
                ego.actor, [waypoint.transform.location for waypoint, _ in route_plan]
            )
        else:
            planner = LocalPlanner(
                ego.actor,
                opt_dict={
                    "dt": CFG.fixed_delta,
                    "target_speed": args.target_speed,
                    "sampling_radius": 2.0,
                },
                map_inst=carla_map,
            )
            planner.set_global_plan(route_plan, stop_waypoint_creation=True)
        spawn_label = args.spawn_index if args.spawn_index is not None else "auto"
        print(f"session {session_dir} | map {world.get_map().name} | "
              f"driver {args.driver} | spawn {spawn_label} | "
              f"route {args.route} | weather {args.weather or 'unchanged'} | frames {args.frames} | "
              f"route_choice={route_info.get('choice_distance_m')}m/"
              f"{route_info.get('choice_yaw_deg')}deg")
        start = time.monotonic()

        with (session_dir / "ego.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "index", "carla_frame", "timestamp", "speed_kmh",
                "x", "y", "z", "yaw", "lane_offset_m", "lane_gt_valid",
                "is_junction", "collisions", "lane_invasions",
            ])
            for index in range(args.frames):
                if planner is not None:
                    ego.actor.apply_control(planner.run_step())
                previous = sensors.frame_ids()
                world.tick()
                frame = sensors.wait_for_frame(previous)
                if frame is None:
                    print("WARN: sensor timeout, bỏ qua frame", flush=True)
                    continue

                cv2.imwrite(str(session_dir / "rgb" / f"{index:06d}.png"), frame.rgb)
                cv2.imwrite(str(session_dir / "seg" / f"{index:06d}.png"), frame.seg_tags)
                np.save(session_dir / "depth" / f"{index:06d}.npy", frame.depth)

                transform = ego.actor.get_transform()
                try:
                    waypoint = carla_map.get_waypoint(
                        transform.location, lane_type=carla.LaneType.Driving
                    )
                    is_junction = waypoint.is_junction
                    ahead = waypoint.next(camera_bottom_lookahead(CFG))
                    target_waypoint = min(
                        ahead,
                        key=lambda candidate: abs(_yaw_delta(waypoint, candidate)),
                    ) if ahead else waypoint
                    target = target_waypoint.transform.location
                    right = transform.get_right_vector()
                    delta = target - transform.location
                    lane_offset = delta.x * right.x + delta.y * right.y
                    lane_gt_valid = True
                except (RuntimeError, AttributeError):
                    lane_offset = float("nan")
                    lane_gt_valid = False
                    is_junction = False
                row_start = handle.tell()
                writer.writerow([
                    index, frame.frame, f"{frame.timestamp:.3f}", f"{frame.speed_kmh:.2f}",
                    f"{transform.location.x:.3f}", f"{transform.location.y:.3f}",
                    f"{transform.location.z:.3f}", f"{transform.rotation.yaw:.2f}",
                    f"{lane_offset:.3f}", int(lane_gt_valid), int(is_junction),
                    frame.collisions, frame.lane_invasions,
                ])
                written += 1
                lane_gt_valid_count += int(lane_gt_valid)
                observed_yaws.append(transform.rotation.yaw)
                observed_junctions.append(is_junction)
                if args.stop_on_collision and frame.collisions > 0:
                    aborted = "collision"
                    print(f"aborted after {written} frames: collision", flush=True)
                    break
                if frame.speed_kmh >= 1.0:
                    low_speed_frames = 0
                    low_speed_rows.clear()
                else:
                    low_speed_frames += 1
                    low_speed_rows.append((index, row_start))
                if args.stall_frames and low_speed_frames >= args.stall_frames:
                    for stalled_index, _ in low_speed_rows:
                        for folder, suffix in (("rgb", ".png"), ("seg", ".png"), ("depth", ".npy")):
                            (session_dir / folder / f"{stalled_index:06d}{suffix}").unlink(missing_ok=True)
                    handle.flush()
                    handle.seek(low_speed_rows[0][1])
                    handle.truncate()
                    written -= len(low_speed_rows)
                    aborted = "stalled"
                    print(f"stalled tail trimmed; kept {written} frames", flush=True)
                    break
                if written % 20 == 0:
                    elapsed = time.monotonic() - start
                    print(f"{written}/{args.frames} frames | {written / elapsed:.1f} fps | "
                          f"{frame.speed_kmh:.1f} km/h", flush=True)

        elapsed = time.monotonic() - start
        max_yaw_change = max(
            (abs((yaw - observed_yaws[0] + 180.0) % 360.0 - 180.0) for yaw in observed_yaws),
            default=0.0,
        )
        saw_junction = any(observed_junctions)
        if args.route == "straight":
            route_valid = max_yaw_change < 10.0 and not saw_junction
        elif args.route == "curve":
            route_valid = 12.0 <= max_yaw_change <= 60.0
        else:
            route_valid = saw_junction and max_yaw_change >= 10.0
        incomplete_reasons = []
        if written != args.frames:
            incomplete_reasons.append("frame_count")
        if aborted is not None:
            incomplete_reasons.append(aborted)
        if lane_gt_valid_count != written:
            incomplete_reasons.append("invalid_lane_gt")
        if not route_valid:
            incomplete_reasons.append("route_profile_mismatch")
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": args.host,
            "port": args.port,
            "map": world.get_map().name,
            "vehicle": ego.actor.type_id,
            "weather": args.weather or "unchanged",
            "driver": args.driver,
            "route": args.route,
            "spawn_index": args.spawn_index if args.spawn_index is not None else min(
                range(len(carla_map.get_spawn_points())),
                key=lambda index: carla_map.get_spawn_points()[index].location.distance(
                    ego.actor.get_location()
                ),
            ),
            "speed_difference": args.speed_difference if planner is None else None,
            "target_speed": args.target_speed,
            "plan_length": args.plan_length,
            "route_choice_distance_m": route_info.get("choice_distance_m"),
            "route_choice_yaw_deg": route_info.get("choice_yaw_deg"),
            "aborted": aborted,
            "requested_frames": args.frames,
            "observed_max_yaw_change_deg": round(max_yaw_change, 2),
            "observed_junction": saw_junction,
            "route_profile_valid": route_valid,
            "incomplete_reasons": incomplete_reasons,
            "complete": not incomplete_reasons,
            "lane_ground_truth": "CARLA waypoint projected to camera bottom lookahead",
            "frames": written,
            "elapsed_s": round(elapsed, 2),
            "avg_fps": round(written / max(elapsed, 1e-6), 2),
            "fixed_delta": CFG.fixed_delta,
            "rgb_width": CFG.rgb_width,
            "rgb_height": CFG.rgb_height,
            "aux_width": CFG.aux_width,
            "aux_height": CFG.aux_height,
            "fov": CFG.fov,
            "tags": {"roads": 1, "sidewalks": 2, "road_lines": 24},
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {written} frames to {session_dir} in {elapsed:.1f}s "
              f"({written / max(elapsed, 1e-6):.1f} fps)")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if sensors is not None:
            sensors.close()
        if ego is not None and args.driver == "tm" and ego.actor is not None and ego.actor.is_alive:
            try:
                ego.actor.set_autopilot(False, traffic_manager.get_port())
            except RuntimeError:
                pass
        if ego is not None:
            ego.destroy()
        if written == 0 and not (session_dir / "meta.json").exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        try:
            world.set_weather(original_weather)
            world.apply_settings(original_settings)
        except RuntimeError as exc:
            print(f"WARN: không khôi phục được settings: {exc}")
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError as exc:
            print(f"WARN: không tắt được TM sync: {exc}")


if __name__ == "__main__":
    main()
