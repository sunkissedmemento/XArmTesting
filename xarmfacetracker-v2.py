"""
XArm6 Face Tracker — Gimbal Edition
──────────────────────────────────────
• RealSense color = webcam (face detection + HUD display)
• RealSense depth = get_distance() at face center (distance gate)
• One HUD window: corner hitbox, face crosshair, camera crosshair,
  error line, distance label, LOCKED indicator
• J1 (yaw) + J5 (pitch) dual-axis gimbal tracking
• Joints read BEFORE servo mode — fixes J5 snap-to-0 on init
• Press Q to quit
"""

import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

ROBOT_IP = "192.168.1.207"

# ── Home position (joint angles, degrees) ───────────────────
INITIAL_JOINTS = [0.2, -100.0, -30.0, 0.0, 50.0, 0.0]

# ── Camera ──────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
CAM_FPS          = 30

# ── Headshot anchor ─────────────────────────────────────────
HEADSHOT_Y_RATIO = 0.18   # 0=top 0.18=forehead 0.5=centre

# ── Control loop ────────────────────────────────────────────
LOOP_HZ = 30
DT      = 1.0 / LOOP_HZ

# ── Horizontal tracking (J1 yaw) ────────────────────────────
YAW_GAIN    = -0.007
YAW_ALPHA   =  0.30
MAX_YAW_DEG =  0.7
DEAD_BAND_X =  20

# ── Vertical tracking (J5 pitch) ────────────────────────────
PITCH_GAIN    =  0.005
PITCH_ALPHA   =  0.30
MAX_PITCH_DEG =  0.5
DEAD_BAND_Y   =  20
J5_MIN        = -10.0
J5_MAX        =  90.0

# ── Joint limits ────────────────────────────────────────────
J1_MIN, J1_MAX = -150.0, 150.0

# ── Arm servo ───────────────────────────────────────────────
SERVO_SPEED = 60
SERVO_ACC   = 200

# ── Distance gates ──────────────────────────────────────────
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 3.00

# ── Scan sweep (no face) ────────────────────────────────────
SCAN_SPEED_DEG = 0.40
SCAN_ACCEL     = 0.06
SCAN_J1_MIN    = -60.0
SCAN_J1_MAX    =  60.0

# ── HUD colours (BGR) ───────────────────────────────────────
COL_TARGET  = (  0, 220,  50)   # green  — tracking target
COL_LOCKED  = (200, 255,   0)   # cyan   — crosshairs aligned
COL_WARNING = (  0,  50, 230)   # red    — too close
COL_OTHER   = (110, 110, 110)   # grey   — other faces
COL_CROSS   = (255, 255, 255)   # white  — camera crosshair
COL_LINE    = (  0, 180, 255)   # orange — error line

# ── YOLO ────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\JHIC\Desktop\usb na dilaw\XArm-Testing (Working)\yolo26n-face.pt"

# ═══════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════

_lock      = threading.Lock()
_faces     = []     # [(dist_m, cx, cy, x1, y1, x2, y2), ...]
_frame_hud = None   # annotated color frame — main thread shows this
_running   = True

# ═══════════════════════════════════════════════════════════
# DRAW HELPERS
# ═══════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * t

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
    cv2.line(img, (0, cy),       (cx-gap, cy),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx+gap, cy),  (FRAME_W, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0),       (cx, cy-gap),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy+gap),  (cx, FRAME_H), color, 1, cv2.LINE_AA)
    draw_crosshair(img, cx, cy, color, size=18, gap=gap, thickness=2)

# ═══════════════════════════════════════════════════════════
# CAMERA THREAD — NO imshow here
# ═══════════════════════════════════════════════════════════

def camera_thread():
    global _faces, _frame_hud, _running

    print("🤖  Loading YOLO face model …")
    try:
        model = YOLO(MODEL_PATH)
        print("✅  Model ready")
    except Exception as exc:
        print(f"❌  Model load failed: {exc}")
        _running = False
        return

    # ── RealSense: color (webcam) + depth (distance only) ───
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, CAM_FPS)
    cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16,  CAM_FPS)
    pipe.start(cfg)

    cx_cam = FRAME_W // 2
    cy_cam = FRAME_H // 2

    try:
        while _running:
            # grab both frames — color is the "webcam", depth for distance
            frames      = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # color → numpy → YOLO + HUD (this IS the webcam image)
            color_image = np.asanyarray(color_frame.get_data())

            # ── YOLO face detection on color image ──────────
            results = model(color_image, verbose=False, imgsz=640)
            faces   = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # headshot anchor (forehead)
                    cx = (x1 + x2) // 2
                    cy = int(y1 + (y2 - y1) * HEADSHOT_Y_RATIO)

                    # distance from depth at face center pixel
                    face_cx = clamp((x1 + x2) // 2, 0, FRAME_W - 1)
                    face_cy = clamp((y1 + y2) // 2, 0, FRAME_H - 1)
                    dist_m  = depth_frame.get_distance(face_cx, face_cy)
                    if dist_m == 0.0:
                        dist_m = 9.9   # no depth reading — treat as far

                    if dist_m <= MAX_DISTANCE_M:
                        faces.append((dist_m, cx, cy, x1, y1, x2, y2))

            faces.sort(key=lambda f: f[0])   # closest = primary target

            # ── Draw HUD on color image ──────────────────────
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

                # corner bracket hitbox
                draw_corner_box(hud, x1, y1, x2, y2, col, thickness=th)

                # face crosshair at headshot anchor
                draw_crosshair(hud, cx, cy, col, size=10, gap=3, thickness=th)

                # error line from camera centre to face anchor (primary only)
                if is_target:
                    cv2.line(hud, (cx_cam, cy_cam), (cx, cy),
                             COL_LINE, 1, cv2.LINE_AA)

                # distance + lock label
                locked_str = ""
                if is_target and not too_close:
                    if abs(cx - cx_cam) < 25 and abs(cy - cy_cam) < 25:
                        locked_str = " ◉ LOCKED"
                cv2.putText(hud, f"{dist:.2f}m{locked_str}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, col, 1, cv2.LINE_AA)

            # camera aim crosshair — always on top
            draw_camera_crosshair(hud, COL_CROSS)

            # push to main thread for display
            with _lock:
                _faces     = faces
                _frame_hud = hud

    except Exception as exc:
        print(f"⚠  Camera thread error: {exc}")
    finally:
        pipe.stop()

# ═══════════════════════════════════════════════════════════
# ARM HELPERS
# ═══════════════════════════════════════════════════════════

def connect_arm():
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    return arm

def go_home(arm):
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_servo_angle(
        angle=INITIAL_JOINTS,
        speed=20, mvacc=200,
        wait=True, is_radian=False
    )

def enable_servo_mode(arm):
    arm.set_mode(1)
    arm.set_state(0)

def read_joints(arm):
    code, angles = arm.get_servo_angle()
    return list(angles) if code == 0 else None

# ═══════════════════════════════════════════════════════════
# TRACKING LOOP — main thread, imshow safe here
# ═══════════════════════════════════════════════════════════

def tracking_loop(arm, initial_joints):
    global _running

    cx_cam = FRAME_W / 2.0
    cy_cam = FRAME_H / 2.0

    frozen = initial_joints[:]
    cur_j1 = frozen[0]   # ≈  0.2
    cur_j5 = frozen[4]   # ≈ 50.0 — correct neutral, not 0
    vel_j1 = 0.0
    vel_j5 = 0.0

    scan_vel = 0.0
    scan_dir = 1

    print(f"📐  Tracking — J1={cur_j1:.1f}°  J5={cur_j5:.1f}°")

    while _running:
        t0 = time.perf_counter()

        with _lock:
            faces = list(_faces)
            frame = _frame_hud

        # ── Single HUD window on main thread ────────────────
        if frame is not None:
            cv2.imshow("XArm6 — Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            _running = False
            break

        # ── Tracking control ────────────────────────────────
        if faces:
            dist, cx, cy, *_ = faces[0]

            if dist > MIN_DISTANCE_M:
                err_x = cx - cx_cam
                if abs(err_x) < DEAD_BAND_X: err_x = 0.0
                vel_j1 = lerp(vel_j1,
                              clamp(YAW_GAIN * err_x, -MAX_YAW_DEG, MAX_YAW_DEG),
                              YAW_ALPHA)

                err_y = cy - cy_cam
                if abs(err_y) < DEAD_BAND_Y: err_y = 0.0
                vel_j5 = lerp(vel_j5,
                              clamp(PITCH_GAIN * err_y, -MAX_PITCH_DEG, MAX_PITCH_DEG),
                              PITCH_ALPHA)
            else:
                # too close — stop
                vel_j1 = lerp(vel_j1, 0.0, 0.35)
                vel_j5 = lerp(vel_j5, 0.0, 0.35)

            scan_vel = 0.0

        else:
            # no face — sweep J1, return J5 to neutral (50)
            scan_vel = lerp(scan_vel, SCAN_SPEED_DEG * scan_dir, SCAN_ACCEL)
            vel_j1   = scan_vel
            j5_err   = 50.0 - cur_j5
            vel_j5   = lerp(vel_j5, clamp(j5_err * 0.02, -0.3, 0.3), 0.10)

            if cur_j1 >= SCAN_J1_MAX:    scan_dir = -1
            elif cur_j1 <= SCAN_J1_MIN:  scan_dir =  1

        # ── Apply ────────────────────────────────────────────
        cur_j1 = clamp(cur_j1 + vel_j1, J1_MIN, J1_MAX)
        cur_j5 = clamp(cur_j5 + vel_j5, J5_MIN, J5_MAX)

        frozen[0] = cur_j1
        frozen[4] = cur_j5

        arm.set_servo_angle_j(frozen[:], speed=SERVO_SPEED, mvacc=SERVO_ACC)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, DT - elapsed))

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    global _running

    arm = connect_arm()

    print("🏠  Moving to home position …")
    go_home(arm)

    # Read joints BEFORE servo mode — J5 stays at 50, not 0
    joints = read_joints(arm)
    if joints is None:
        print("❌  Could not read joint angles — aborting.")
        arm.disconnect()
        return
    print(f"📐  Joints at home: {[round(j,1) for j in joints]}")
    print(f"    J5 = {joints[4]:.1f}° (should be ≈ 50)")

    enable_servo_mode(arm)   # switch AFTER reading

    cam_t = threading.Thread(target=camera_thread, daemon=True, name="camera")
    cam_t.start()

    print("⏳  Waiting for camera warm-up …")
    time.sleep(2.0)

    print("🎯  Tracking started — press Q to quit")
    try:
        tracking_loop(arm, joints)
    finally:
        _running = False
        cv2.destroyAllWindows()
        arm.disconnect()
        print("👋  Shutdown complete")


if __name__ == "__main__":
    main()