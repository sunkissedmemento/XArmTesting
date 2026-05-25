"""
HUD Test — Windows compatible
──────────────────────────────
• RealSense color + depth
• YOLO face detection
• Full HUD: corner hitbox, face crosshair, camera crosshair,
  error line, distance label, LOCKED indicator
• NO arm — pure camera test
• Press Q to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
CAM_FPS          = 30
HEADSHOT_Y_RATIO = 0.18
MIN_DISTANCE_M   = 0.25
MAX_DISTANCE_M   = 3.00
MODEL_PATH       = "yolo26n-face.pt"   # same folder as this script

# ── HUD colours (BGR) ───────────────────────────────────────
COL_TARGET  = (  0, 220,  50)
COL_LOCKED  = (200, 255,   0)
COL_WARNING = (  0,  50, 230)
COL_OTHER   = (110, 110, 110)
COL_CROSS   = (255, 255, 255)
COL_LINE    = (  0, 180, 255)

# ── Draw helpers ────────────────────────────────────────────
def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, ratio=0.22):
    w  = x2 - x1;  h  = y2 - y1
    lx = int(w * ratio);  ly = int(h * ratio)
    corners = [
        ((x1, y1+ly), (x1, y1),  (x1+lx, y1)),
        ((x2-lx, y1), (x2, y1),  (x2, y1+ly)),
        ((x2, y2-ly), (x2, y2),  (x2-lx, y2)),
        ((x1+lx, y2), (x1, y2),  (x1, y2-ly)),
    ]
    for p0, corner, p1 in corners:
        cv2.line(img, p0, corner, color, thickness, cv2.LINE_AA)
        cv2.line(img, corner, p1, color, thickness, cv2.LINE_AA)

def draw_crosshair(img, cx, cy, color, size=12, gap=4, thickness=2):
    cv2.line(img, (cx-size, cy), (cx-gap,  cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx+gap,  cy), (cx+size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy-size), (cx, cy-gap),  color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy+gap),  (cx, cy+size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)

def draw_camera_crosshair(img, color):
    cx, cy = FRAME_W // 2, FRAME_H // 2
    gap = 8
    cv2.line(img, (0, cy),      (cx-gap, cy),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx+gap, cy), (FRAME_W, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0),      (cx, cy-gap),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy+gap), (cx, FRAME_H), color, 1, cv2.LINE_AA)
    draw_crosshair(img, cx, cy, color, size=18, gap=gap, thickness=2)

# ── Main ────────────────────────────────────────────────────
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model ready")

pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, CAM_FPS)
cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16,  CAM_FPS)
pipe.start(cfg)

cx_cam = FRAME_W // 2
cy_cam = FRAME_H // 2

print("Running — press Q to quit")

try:
    while True:
        frames      = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # YOLO detection
        results = model(color_image, verbose=False, imgsz=640)
        faces   = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = int(y1 + (y2 - y1) * HEADSHOT_Y_RATIO)

                face_cx = max(0, min((x1 + x2) // 2, FRAME_W - 1))
                face_cy = max(0, min((y1 + y2) // 2, FRAME_H - 1))
                dist_m  = depth_frame.get_distance(face_cx, face_cy)
                if dist_m == 0.0:
                    dist_m = 9.9

                if dist_m <= MAX_DISTANCE_M:
                    faces.append((dist_m, cx, cy, x1, y1, x2, y2))

        faces.sort(key=lambda f: f[0])

        # Draw HUD
        hud = color_image.copy()

        for i, (dist, cx, cy, x1, y1, x2, y2) in enumerate(faces):
            is_target = (i == 0)
            too_close = (dist <= MIN_DISTANCE_M)

            if too_close:
                col = COL_WARNING
            elif is_target:
                err = abs(cx - cx_cam) + abs(cy - cy_cam)
                col = COL_LOCKED if err < 25 else COL_TARGET
            else:
                col = COL_OTHER

            th = 2 if is_target else 1

            draw_corner_box(hud, x1, y1, x2, y2, col, thickness=th)
            draw_crosshair(hud, cx, cy, col, size=10, gap=3, thickness=th)

            if is_target:
                cv2.line(hud, (cx_cam, cy_cam), (cx, cy), COL_LINE, 1, cv2.LINE_AA)

            locked_str = ""
            if is_target and not too_close:
                if abs(cx - cx_cam) < 25 and abs(cy - cy_cam) < 25:
                    locked_str = " LOCKED"
            cv2.putText(hud, f"{dist:.2f}m{locked_str}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, col, 1, cv2.LINE_AA)

        draw_camera_crosshair(hud, COL_CROSS)

        cv2.imshow("HUD Test", hud)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()