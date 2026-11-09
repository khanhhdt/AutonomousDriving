import math
import threading
import time
from dataclasses import dataclass
import carla
import numpy as np

@dataclass
class SensorFrame:
    rgb: np.ndarray
    seg: np.ndarray
    seg_tags: np.ndarray
    depth: np.ndarray
    speed_kmh: float
    collisions: int
    lane_invasions: int
    frame: int
    timestamp: float


def _decode_rgb(image: carla.Image) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    return raw.reshape(image.height, image.width, 4)[:, :, :3].copy()


def _decode_depth(image: carla.Image) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape(image.height, image.width, 4).astype(np.float32)
    normalized = (bgra[:, :, 2] + bgra[:, :, 1] * 256.0 + bgra[:, :, 0] * 65536.0) / (256.0 ** 3 - 1.0)
    return (normalized * 1000.0).astype(np.float32)


class SensorManager:
    """Owns RGB + semantic segmentation + depth cameras and event sensors.
    Assembles exactly one SensorFrame per world.tick() using the shared frame id.
    """

    def __init__(self, world: carla.World, vehicle: carla.Actor, cfg) -> None:
        self._world = world
        self._vehicle = vehicle
        self._cfg = cfg
        self._lock = threading.Lock()
        self._ids = {"rgb": -1, "seg": -1, "depth": -1}
        self._rgb = None
        self._timestamp = 0.0
        self._seg = None
        self._tags = None
        self._depth = None
        self._collisions = 0
        self._lane_invasions = 0
        self._actors: list[carla.Actor] = []
        self._spawn()

    def _camera(self, kind: str, width: int, height: int) -> carla.Actor:
        blueprint = self._world.get_blueprint_library().find(f"sensor.camera.{kind}")
        blueprint.set_attribute("role_name", f"ego_project_{kind}")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(self._cfg.fov))
        blueprint.set_attribute("sensor_tick", str(self._cfg.sensor_tick))
        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        return self._world.spawn_actor(blueprint, transform, attach_to=self._vehicle)

    def _spawn(self) -> None:
        rgb_cam = self._camera("rgb", self._cfg.rgb_width, self._cfg.rgb_height)
        seg_cam = self._camera("semantic_segmentation", self._cfg.aux_width, self._cfg.aux_height)
        depth_cam = self._camera("depth", self._cfg.aux_width, self._cfg.aux_height)
        self._actors = [rgb_cam, seg_cam, depth_cam]

        blueprint_library = self._world.get_blueprint_library()
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_bp.set_attribute("role_name", "ego_project_collision")
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        lane_invasion_bp.set_attribute("role_name", "ego_project_lane_invasion")
        collision = self._world.spawn_actor(collision_bp, carla.Transform(), attach_to=self._vehicle)
        lane_invasion = self._world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self._vehicle)
        self._actors += [collision, lane_invasion]

        rgb_cam.listen(self._on_rgb)
        seg_cam.listen(self._on_seg)
        depth_cam.listen(self._on_depth)
        collision.listen(self._on_collision)
        lane_invasion.listen(self._on_lane_invasion)

    def _on_rgb(self, image: carla.Image) -> None:
        rgb = _decode_rgb(image)
        with self._lock:
            self._rgb = rgb
            self._timestamp = image.timestamp
            self._ids["rgb"] = image.frame

    def _on_seg(self, image: carla.Image) -> None:
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        tags = raw.reshape(image.height, image.width, 4)[:, :, 2].copy()
        image.convert(carla.ColorConverter.CityScapesPalette)
        pal = np.frombuffer(image.raw_data, dtype=np.uint8)
        palette = pal.reshape(image.height, image.width, 4)[:, :, :3].copy()
        with self._lock:
            self._tags = tags
            self._seg = palette
            self._ids["seg"] = image.frame

    def _on_depth(self, image: carla.Image) -> None:
        depth = _decode_depth(image)
        with self._lock:
            self._depth = depth
            self._ids["depth"] = image.frame

    def _on_collision(self, _event) -> None:
        with self._lock:
            self._collisions += 1

    def _on_lane_invasion(self, _event) -> None:
        with self._lock:
            self._lane_invasions += 1

    def frame_ids(self) -> dict[str, int]:
        with self._lock:
            return dict(self._ids)

    def wait_for_frame(self, previous_ids: dict[str, int], timeout: float = 2.0) -> SensorFrame | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                updated = all(self._ids[key] != previous_ids.get(key, -1) for key in self._ids)
                if updated and self._rgb is not None and self._seg is not None and self._depth is not None:
                    velocity = self._vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    return SensorFrame(
                        rgb=self._rgb,
                        seg=self._seg,
                        seg_tags=self._tags,
                        depth=self._depth,
                        speed_kmh=speed_kmh,
                        collisions=self._collisions,
                        lane_invasions=self._lane_invasions,
                        frame=max(self._ids.values()),
                        timestamp=self._timestamp,
                    )
            time.sleep(0.001)
        return None

    def close(self) -> None:
        for actor in self._actors:
            try:
                if actor.is_alive:
                    actor.stop()
                    actor.destroy()
            except RuntimeError:
                pass
        self._actors = []
