import math
import carla

class EgoVehicle:
    def __init__(self, world: carla.World, blueprint_filter: str,
                 spawn_index: int | None = None) -> None:
        self._world = world
        self.actor = self._spawn(blueprint_filter, spawn_index)

    def _spawn(self, blueprint_filter: str, spawn_index: int | None = None) -> carla.Actor:
        blueprints = self._world.get_blueprint_library().filter(blueprint_filter)
        if not blueprints:
            raise ValueError(f"blueprint not found: {blueprint_filter}")
        blueprint = blueprints[0]
        blueprint.set_attribute("role_name", "ego_project")
        spawn_points = self._world.get_map().get_spawn_points()
        if spawn_index is not None:
            if not 0 <= spawn_index < len(spawn_points):
                raise ValueError(f"spawn index out of range: {spawn_index}")
            spawn_points = [spawn_points[spawn_index]]
        for spawn_point in spawn_points:
            try:
                return self._world.spawn_actor(blueprint, spawn_point)
            except RuntimeError:
                continue
        raise RuntimeError("no free spawn point")

    def apply(self, steer: float = 0.0, throttle: float = 0.0, brake: float = 0.0,
              hand_brake: bool = False, reverse: bool = False) -> None:
        self.actor.apply_control(carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            hand_brake=hand_brake,
            reverse=reverse,
        ))

    def speed_kmh(self) -> float:
        velocity = self.actor.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def destroy(self) -> None:
        if self.actor is not None and self.actor.is_alive:
            self.actor.destroy()
        self.actor = None
