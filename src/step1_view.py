# scratch script: python -m src.step1_view
import queue

import carla
import cv2
import numpy as np

from configs.config import CFG

client = carla.Client(CFG.host, CFG.port)
client.set_timeout(30.0)
world = client.get_world()
print("Map:", world.get_map().name)

bp = world.get_blueprint_library()
vehicle = world.spawn_actor(bp.filter("vehicle.tesla.model3")[0],
                            world.get_map().get_spawn_points()[0])

cam_bp = bp.find("sensor.camera.rgb")
for k, v in {"image_size_x": "640", "image_size_y": "360",
             "fov": "90", "sensor_tick": "0.05"}.items():
    cam_bp.set_attribute(k, v)
camera = world.spawn_actor(cam_bp,
    carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

frames = queue.Queue()
camera.listen(lambda img: frames.put(img))
try:
    while True:
        img = frames.get()
        arr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
        cv2.imshow("CARLA", arr[:, :, :3])
        if cv2.waitKey(1) == 27:
            break
finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()