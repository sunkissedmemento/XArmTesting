import os
import platform
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ── Qt font fix for Pi — must be before cv2 is used ────────
if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_QPA_FONTDIR"]  = "/usr/share/fonts/truetype/dejavu"

# ═══════════════════════════════════════════════════════════
# !! CHANGE THIS FLAG !!
#   True  = Windows / test  (no arm, xarm SDK not needed)
#   False = Raspberry Pi    (connects real arm)
# ═══════════════════════════════════════════════════════════
TEST_MODE = False
# ═══════════════════════════════════════════════════════════

if not TEST_MODE:
    from xarm.wrapper import XArmAPI

# ── Auto-detect platform ────────────────────────────────────
IS_PI      = platform.system() == "Linux" and platform.machine().startswith("aarch")
DEVICE     = "cuda" if __import__("torch").cuda.is_available() else "cpu"
YOLO_IMGSZ = 320 if IS_PI else 640

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

ROBOT_IP       = "192.168.1.207"
INITIAL_JOINTS = [0.2, -100.0, -30.0, 0.0, 50.0, 0.0]

FRAME_W, FRAME_H = 640, 480
CAM_FPS          = 30
DT               = 0.04

# ── Headshot anchor ─────────────────────────────────────────
HEADSHOT_Y_RATIO = 0.18

# ── TRACKING mode gains ─────────────────────────────────────
YAW_GAIN_TRACK    = -0.012
PITCH_GAIN_TRACK  =  0.008
YAW_ALPHA_TRACK   =  0.35
PITCH_ALPHA_TRACK =  0.35
MAX_YAW_DEG       =  1.0
MAX_PITCH_DEG     =  0.8
DEAD_BAND_X       =  30
DEAD_BAND_Y       =  20

# ── LOCKED mode gains (smooth follow) ───────────────────────
YAW_GAIN_LOCK      = -0.004
PITCH_GAIN_LOCK    =  0.003
YAW_ALPHA_LOCK     =  0.15
PITCH_ALPHA_LOCK   =  0.15
MAX_YAW_DEG_LOCK   =  0.4
MAX_PITCH_DEG_LOCK =  0.3

# ── Lock / unlock thresholds ────────────────────────────────
LOCK_PX   = 25
UNLOCK_PX = 50

# ── Joint limits ────────────────────────────────────────────
J1_MIN, J1_MAX = -150.0, 150.0
J5_MIN         = -10.0
J5_MAX         =  90.0

# ── Servo ───────────────────────────────────────────────────
SERVO_SPEED = 80
SERVO_ACC   = 250

# ── Distance gates ──────────────────────────────────────────
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 2.00

# ── Scan sweep ──────────────────────────────────────────────
SCAN_SPEED_DEG = 0.40
SCAN_J1_MIN    = -60.0
SCAN_J1_MAX    =  60.0

# ── HUD colours (BGR) ───────────────────────────────────────
COL_TARGET  = (  0, 220,  50)
COL_LOCKED  = (200, 255,   0)
COL_WARNING = (  0,  50, 230)
COL_OTHER   = (110, 110, 110)
COL_CROSS   = (255, 255, 255)
COL_LINE    = (  0, 180, 255)

# ── YOLO ────────────────────────────────────────────────────
MODEL_PATH = "yolo26n-face.pt"

# ═════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════

_lock    = threading.Lock()
_faces   = []
_frame   = None
_running = True

# ═════════════════════════════════════════════════════════════
# DRAW HELPERS
# ═════════════════════════════════════════════════════════════

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
    cv2.line(img, (0, cy),      (cx-gap, cy),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx+gap, cy), (FRAME_W, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0),      (cx, cy-gap),  color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy+gap), (cx, FRAME_H), color, 1, cv2.LINE_AA)
    draw_crosshair(img, cx, cy, color, size=18, gap=gap, thickness=2)

# ═════════════════════════════════════════════════════════════
# ARM HELPERS — stubbed in TEST_MODE
# ═════════════════════════════════════════════════════════════

class FakeArm:
    def set_servo_angle_j(self, *a, **kw): pass
    def disconnect(self): pass

def connect_arm():
    if TEST_MODE:
        print("🖥️   Test mode — no arm")
        return FakeArm()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    return arm

def go_home(arm):
    if TEST_MODE:
        return
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_servo_angle(
        angle=INITIAL_JOINTS,
        speed=20, mvacc=200,
        wait=True, is_radian=False
    )

def enable_servo(arm):
    if TEST_MODE:
        return
    arm.set_mode(1)
    arm.set_state(0)

def get_joints(arm):
    if TEST_MODE:
        return list(INITIAL_JOINTS)
    code, ang = arm.get_servo_angle()
    return list(ang) if code == 0 else None

# ═════════════════════════════════════════════════════════════
# CAMERA THREAD — builds HUD frame, NO imshow here
# ═════════════════════════════════════════════════════════════

def realsense_thread():
    global _faces, _frame, _running

    print("🤖  Loading face model …")
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        print(f"✅  Model ready on {DEVICE.upper()}")
    except Exception as e:
        print("❌  Failed to load model:", e)
        _running = False
        return

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, CAM_FPS)
    cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16,  CAM_FPS)
    pipeline.start(cfg)

    align  = rs.align(rs.stream.color)
    cx_cam = FRAME_W // 2
    cy_cam = FRAME_H // 2

    while _running:
        try:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)

            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue

            frame = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())

            # ── YOLO ────────────────────────────────────────
            results = model(frame, verbose=False, imgsz=YOLO_IMGSZ)
            faces   = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = int(y1 + (y2 - y1) * HEADSHOT_Y_RATIO)

                    face_cx = clamp((x1 + x2) // 2, 0, FRAME_W - 1)
                    face_cy = clamp((y1 + y2) // 2, 0, FRAME_H - 1)
                    dist_m  = df.get_distance(face_cx, face_cy)
                    if dist_m == 0.0:
                        dist_m = 9.9

                    if dist_m <= MAX_DISTANCE_M:
                        faces.append((dist_m, cx, cy, x1, y1, x2, y2))

            faces.sort(key=lambda f: f[0])

            # ── HUD ─────────────────────────────────────────
            annotated = frame.copy()

            # compute locked state for HUD colour
            locked_hud = False
            if faces:
                _, fcx, fcy, *_ = faces[0]
                if abs(fcx - cx_cam) + abs(fcy - cy_cam) < LOCK_PX:
                    locked_hud = True

            for i, (dist, cx, cy, x1, y1, x2, y2) in enumerate(faces):
                is_target = (i == 0)
                too_close = (dist <= MIN_DISTANCE_M)

                if too_close:
                    col = COL_WARNING
                elif is_target:
                    col = COL_LOCKED if locked_hud else COL_TARGET
                else:
                    col = COL_OTHER

                th = 2 if is_target else 1

                draw_corner_box(annotated, x1, y1, x2, y2, col, thickness=th)
                draw_crosshair(annotated, cx, cy, col, size=10, gap=3, thickness=th)

                if is_target:
                    cv2.line(annotated, (cx_cam, cy_cam), (cx, cy),
                             COL_LINE, 1, cv2.LINE_AA)

                status = " LOCKED" if (is_target and locked_hud) else ""
                cv2.putText(annotated, f"{dist:.2f}m{status}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)

            # mode label
            mode_txt = "LOCKED" if locked_hud else "TRACKING"
            mode_col = COL_LOCKED if locked_hud else COL_TARGET
            cv2.putText(annotated, mode_txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2, cv2.LINE_AA)

            # platform info
            plat = "Pi" if IS_PI else "Win"
            cv2.putText(annotated, f"{plat} | {DEVICE.upper()} | {YOLO_IMGSZ}px",
                        (10, FRAME_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

            if TEST_MODE:
                cv2.putText(annotated, "TEST MODE", (FRAME_W - 130, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

            draw_camera_crosshair(annotated, COL_CROSS)

            with _lock:
                _faces = faces
                _frame = annotated

        except Exception as e:
            print("⚠  Camera error:", e)
            time.sleep(1)

    pipeline.stop()

# ═════════════════════════════════════════════════════════════
# TRACKING LOOP — main thread, imshow safe here
# ═════════════════════════════════════════════════════════════

def tracking_loop(arm):
    global _running

    cx_cam = FRAME_W / 2.0
    cy_cam = FRAME_H / 2.0

    frozen = get_joints(arm)
    if not frozen:
        print("❌  Joint read failed")
        return

    cur_j1  = frozen[0]   # ≈  0.2
    cur_j5  = frozen[4]   # ≈ 50.0
    vel_j1  = 0.0
    vel_j5  = 0.0
    scan_dir = 1
    locked   = False

    title = "Face Tracker [TEST]" if TEST_MODE else "Face Tracker"
    print(f"🎯  Tracking started — press Q to quit")

    while _running:
        t0 = time.time()

        with _lock:
            faces = list(_faces)
            frame = _frame

        # ── imshow on main thread ────────────────────────────
        if frame is not None:
            cv2.imshow(title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            _running = False
            break

        # ── lock state ───────────────────────────────────────
        if faces:
            _, fcx, fcy, *_ = faces[0]
            total_err = abs(fcx - cx_cam) + abs(fcy - cy_cam)
            if not locked and total_err < LOCK_PX:
                locked = True
            elif locked and total_err > UNLOCK_PX:
                locked = False

        # ── arm control ──────────────────────────────────────
        if faces:
            dist, cx, cy, *_ = faces[0]

            if dist > MIN_DISTANCE_M:
                err_x = cx - cx_cam
                err_y = cy - cy_cam

                if locked:
                    yaw_gain    = YAW_GAIN_LOCK
                    pitch_gain  = PITCH_GAIN_LOCK
                    yaw_alpha   = YAW_ALPHA_LOCK
                    pitch_alpha = PITCH_ALPHA_LOCK
                    max_yaw     = MAX_YAW_DEG_LOCK
                    max_pitch   = MAX_PITCH_DEG_LOCK
                    db_x, db_y  = 8, 8
                else:
                    yaw_gain    = YAW_GAIN_TRACK
                    pitch_gain  = PITCH_GAIN_TRACK
                    yaw_alpha   = YAW_ALPHA_TRACK
                    pitch_alpha = PITCH_ALPHA_TRACK
                    max_yaw     = MAX_YAW_DEG
                    max_pitch   = MAX_PITCH_DEG
                    db_x, db_y  = DEAD_BAND_X, DEAD_BAND_Y

                if abs(err_x) < db_x: err_x = 0.0
                if abs(err_y) < db_y: err_y = 0.0

                vel_j1 = lerp(vel_j1,
                              clamp(yaw_gain * err_x, -max_yaw, max_yaw),
                              yaw_alpha)
                vel_j5 = lerp(vel_j5,
                              clamp(pitch_gain * err_y, -max_pitch, max_pitch),
                              pitch_alpha)
            else:
                vel_j1 = lerp(vel_j1, 0.0, 0.35)
                vel_j5 = lerp(vel_j5, 0.0, 0.35)

            scan_dir = 1

        else:
            locked  = False
            vel_j1  = SCAN_SPEED_DEG * scan_dir
            j5_err  = 50.0 - cur_j5
            vel_j5  = lerp(vel_j5, clamp(j5_err * 0.02, -0.3, 0.3), 0.10)

            if cur_j1 >= SCAN_J1_MAX:    scan_dir = -1
            elif cur_j1 <= SCAN_J1_MIN:  scan_dir =  1

        cur_j1 = clamp(cur_j1 + vel_j1, J1_MIN, J1_MAX)
        cur_j5 = clamp(cur_j5 + vel_j5, J5_MIN, J5_MAX)

        frozen[0] = cur_j1
        frozen[4] = cur_j5

        arm.set_servo_angle_j(frozen[:], speed=SERVO_SPEED, mvacc=SERVO_ACC)

        time.sleep(max(0, DT - (time.time() - t0)))

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    global _running

    print(f"🖥️   Platform : {'Raspberry Pi' if IS_PI else 'Windows'}")
    print(f"⚡  Device   : {DEVICE.upper()}")
    print(f"🔍  YOLO size: {YOLO_IMGSZ}")
    print(f"🔧  Mode     : {'TEST (no arm)' if TEST_MODE else 'LIVE (arm connected)'}")

    arm = connect_arm()
    print("🏠  Moving to home …")
    go_home(arm)

    # Read joints BEFORE servo mode — keeps J5 at 50, not 0
    enable_servo(arm)

    cam_t = threading.Thread(target=realsense_thread, daemon=True)
    cam_t.start()

    print("⏳  Waiting for camera …")
    time.sleep(2)

    try:
        tracking_loop(arm)
    finally:
        _running = False
        cv2.destroyAllWindows()
        arm.disconnect()
        print("👋  Done")


if __name__ == "__main__":
    main()
