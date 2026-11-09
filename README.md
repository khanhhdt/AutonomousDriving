# Autonomous Driving (CARLA) — Lane Detection + Object Avoidance

Mô phỏng xe tự hành trên CARLA: lane detection (OpenCV + semantic segmentation),
quyết định real-time và tránh vật cản.

## Yêu cầu

- CARLA server 0.9.16 chạy trên máy GPU (RPC `:3000`, streaming `:3001`)
- Python 3.11 + `carla` 0.9.16, OpenCV, numpy
- Mạng giữa client và server ≥ 20 MB/s (đo bằng `iperf3`), WiFi nên tắt power save

## Cài đặt

```bash
conda activate carla
pip install -r requirements.txt
# carla: cài từ wheel của server
pip install carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl
```

## Chạy

```bash
# đổi host nếu cần (mặc định 100.85.59.63)
python -m src.main --duration 60            # có cửa sổ RGB + seg
python -m src.main --duration 60 --no-display

CARLA_HOST=192.168.80.87 python -m src.main --duration 60
```

Phím `q`/`ESC` để thoát. Chương trình tự khôi phục chế độ async và hủy actor.

## Phase 2 — Dataset + lane detection

```bash
# tiền kiểm tra các spawn có route phù hợp trên map hiện tại
python -m scripts.record_dataset --route curve --list-spawns
python -m scripts.record_dataset --route intersection --list-spawns

# Town10HD_Opt: spawn 22 cho đoạn thẳng; spawn 62 cho đường cong ngoài giao lộ;
# spawn 2 cho route qua giao lộ (rẽ khoảng 50°).
# Mỗi session chỉ được tính vào benchmark nếu đủ frame, không va chạm và GT hợp lệ.
set -e
for route_spawn in "straight 22" "curve 62" "intersection 2"; do
  set -- $route_spawn
  route=$1
  spawn=$2
  for weather in clear rain; do
    python -m scripts.record_dataset --frames 150 --driver tm --target-speed 10 \
      --plan-length 400 --spawn-index "$spawn" --route "$route" \
      --weather "$weather" --out data/raw/m1
  done
done

# Chỉ session hoàn chỉnh được dùng cho verdict M1; incomplete bị bỏ qua mặc định.
# MP4 và ảnh overlay được lưu trong từng session/eval/.
python -m scripts.eval_lane --session data/raw/m1/session_* --save-video --save-overlay 20
```

- Session ghi mới dùng `rgb/` PNG 768×432, `seg/` tag PNG 384×216 và `depth/` NPY mét;
  mỗi `meta.json` lưu độ phân giải thực tế. Session cũ vẫn giữ nguyên độ phân giải lúc ghi.
- Tag đã xác minh trên server: `Road = 1`, `Sidewalk = 2`, `RoadLines = 24`
- Lane detection: ROI → Gaussian blur/Canny hỗ trợ trích màu trắng-vàng → lọc road-region
  (seg) → IPM → sliding window → polyfit; trả `LaneState.offset_px` (dương = tâm làn lệch phải)
- `lane_offset_m` là độ lệch tới waypoint khoảng 3 m phía trước (chiếu tại đáy camera);
  `lane_gt_valid` đánh dấu GT hợp lệ.
- Recorder dừng sau va chạm hoặc 60 frame liên tiếp xe chạy dưới 1 km/h; phần đuôi đứng yên bị loại khỏi dataset. Session bị dừng sớm được gắn `complete=false` và evaluator loại khỏi M1.
- Eval: IoU mask RoadLines và sai số tâm làn so với centerline fit từ semantic
  RoadLines. **PASS M1** yêu cầu sai số <15 px ở ít nhất 80% frame có segmentation GT
  hợp lệ; detector bỏ lỡ trên các frame đó tính là fail. Báo riêng tỷ lệ frame có GT
  hợp lệ và tỷ lệ detector tạo được centerline để thể hiện coverage. GT waypoint chiếu
  tại đáy camera được in thành số liệu chẩn đoán, không quyết định verdict M1.

## Kiểm tra Phase 1

- 60 s chạy headless: ~20 fps (`19–21`), không timeout sensor
- Sau khi thoát: không còn actor do chương trình tạo
- Server trở lại `synchronous_mode = False`

## Cấu trúc

```
configs/config.py       host/port, sync mode, resolution
src/main.py             vòng lặp 20 Hz
src/sensors.py          RGB + semantic seg + depth + collision/lane invasion
src/vehicle.py          spawn ego, apply_control
src/perception/         lane_detection.py (Phase 2), segmentation, obstacles
scripts/                record_dataset.py, eval_lane.py
src/planning/           FSM (Phase 4)
src/control/            pure pursuit + PID (Phase 3)
third_party/agents/     CARLA 0.9.16 PythonAPI agents (MIT): LocalPlanner + PID
docs/                   sơ đồ kiến trúc
```

`--driver local` dùng `LocalPlanner`/PID vendored từ CARLA PythonAPI (không cần
cài `networkx`/`shapely`). `--plan-length` mặc định 400 m (~1400 frame ở 20 km/h),
tăng nếu ghi dài hơn. `--driver tm` (mặc định) vẫn dùng Traffic Manager.

Phase 3 có thể dùng trực tiếp `agents.navigation.controller.VehiclePIDController`
(đã vendor) hoặc port vào `src/control/pid.py`.
