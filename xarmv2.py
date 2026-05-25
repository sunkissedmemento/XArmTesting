import os
import platform
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ============================================================
# Raspberry Pi / Windows Ready Face Tracker
# FIX: Lock state is now computed once in realsense_thread and
#      shared via _state tuple. tracking_loop reads it from there
#      instead of recomputing independently.
# ============================================================

# ---------- USER SETTINGS ----------
TEST_MODE = False
SHOW_WINDOW = True
ROBOT_IP = "192.168.1.207"
MODEL_PATH = "yolo26n-face.pt"

INITIAL_JOINTS = [0.2, -100.0, -30.0, 0.0, 50.0, 0.0]

# ---------- PLATFORM ----------
MACHINE = platform.machine().lower()
IS_LINUX = platform.system() == "Linux"
IS_PI = IS_LINUX and ("aarch" in MACHINE or "arm" in MACHINE)

if IS_LINUX:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ---------- CAMERA / YOLO ----------
FRAME_W, FRAME_H = 640, 480
CAM_FPS = 15 if IS_PI else 30
YOLO_IMGSZ = 320 if IS_PI else 640
DT = 0.04

# ---------- Headshot anchor ----------
HEADSHOT_Y_RATIO = 0.18

# ---------- TRACKING mode gains ----------
YAW_GAIN_TRACK = -0.012
PITCH_GAIN_TRACK = 0.008
YAW_ALPHA_TRACK = 0.35
PITCH_ALPHA_TRACK = 0.35
MAX_YAW_DEG = 1.0
MAX_PITCH_DEG = 0.8
DEAD_BAND_X = 30
DEAD_BAND_Y = 20

# ---------- LOCKED mode gains ----------
YAW_GAIN_LOCK = -0.004
PITCH_GAIN_LOCK = 0.003
YAW_ALPHA_LOCK = 0.15
PITCH_ALPHA_LOCK = 0.15
MAX_YAW_DEG_LOCK = 0.4
MAX_PITCH_DEG_LOCK = 0.3

# ---------- Lock thresholds ----------
LOCK_PX = 25
UNLOCK_PX = 50

# ---------- Joint limits ----------
J1_MIN, J1_MAX = -150.0, 150.0
J5_MIN, J5_MAX = -10.0, 90.0

# ---------- Servo ----------
SERVO_SPEED = 80
SERVO_ACC = 250

# ---------- Distance gates ----------
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 2.00

# ---------- Scan sweep ----------
SCAN_SPEED_DEG = 0.40
SCAN_J1_MIN = -60.0
SCAN_J1_MAX = 60.0

# ---------- HUD colours ----------
COL_TARGET = (0, 220, 50)
COL_LOCKED = (200, 255, 0)
COL_WARNING = (0, 50, 230)
COL_OTHER = (110, 110, 110)
COL_CROSS = (255, 255, 255)
COL_LINE = (0, 180, 255)

# ---------- xArm import ----------
if not TEST_MODE:
    try:
        from xarm.wrapper import XArmAPI
    except Exception as e:
        print("ERROR: Failed to import xArm SDK.")
        print("Install it on Raspberry Pi first, or set TEST_MODE = True.")
        print("Details:", e)
        raise

# ---------- Shared state ----------
# _state is a tuple: (faces, annotated_frame, locked)
# Bundled so faces and locked are always in sync when read.
_lock = threading.Lock()
_state = ([], None, False)   # <-- CHANGED: was separate _faces/_frame globals
_running = True
_camera_ready = threading.Event()  # <-- NEW: signals camera+model are up


# ============================================================
# HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, ratio=0.22):
    w = x2 - x1
    h = y2 - y1
    lx = int(w * ratio)
    ly = int(h * ratio)

    corners = [
        ((x1, y1 + ly), (x1, y1), (x1 + lx, y1)),
        ((x2 - lx, y1), (x2, y1), (x2, y1 + ly)),
        ((x2, y2 - ly), (x2, y2), (x2 - lx, y2)),
        ((x1 + lx, y2), (x1, y2), (x1, y2 - ly)),
    ]

    for p0, corner, p1 in corners:
        cv2.line(img, p0, corner, color, thickness, cv2.LINE_AA)
        cv2.line(img, corner, p1, color, thickness, cv2.LINE_AA)


def draw_crosshair(img, cx, cy, color, size=12, gap=4, thickness=2):
    cv2.line(img, (cx - size, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy - gap), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)


def draw_camera_crosshair(img, color):
    cx, cy = FRAME_W // 2, FRAME_H // 2
    gap = 8

    cv2.line(img, (0, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (FRAME_W, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, FRAME_H), color, 1, cv2.LINE_AA)

    draw_crosshair(img, cx, cy, color, size=18, gap=gap, thickness=2)


# ============================================================
# ARM HELPERS
# ============================================================

class FakeArm:
    def set_servo_angle_j(self, *args, **kwargs):
        pass

    def disconnect(self):
        pass


def connect_arm():
    if TEST_MODE:
        print("TEST MODE: xArm disabled.")
        return FakeArm()

    print(f"Connecting to xArm at {ROBOT_IP} ...")
    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    if not arm.connected:
        raise RuntimeError("xArm connection failed. Check ROBOT_IP and network.")

    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)

    print("xArm connected.")
    return arm


def go_home(arm):
    if TEST_MODE:
        return

    arm.set_mode(0)
    arm.set_state(0)
    arm.set_servo_angle(
        angle=INITIAL_JOINTS,
        speed=20,
        mvacc=200,
        wait=True,
        is_radian=False
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
    if code != 0 or ang is None:
        return None

    return list(ang)


# ============================================================
# CAMERA THREAD
# ============================================================

def realsense_thread():
    global _state, _running

    print("Loading YOLO face model...")

    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        print(f"Model ready on {DEVICE.upper()}. YOLO size: {YOLO_IMGSZ}")
    except Exception as e:
        print("ERROR: Failed to load YOLO model.")
        print("Check that the model file exists:", MODEL_PATH)
        print("Details:", e)
        _running = False
        _camera_ready.set()  # unblock main thread even on failure
        return

    pipeline = rs.pipeline()
    cfg = rs.config()

    try:
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, CAM_FPS)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, CAM_FPS)
        pipeline.start(cfg)
        print(f"RealSense started: {FRAME_W}x{FRAME_H} @ {CAM_FPS} FPS")
    except Exception as e:
        print("ERROR: RealSense failed to start.")
        print("Details:", e)
        _running = False
        _camera_ready.set()  # unblock main thread even on failure
        return

    align = rs.align(rs.stream.color)
    cx_cam = FRAME_W // 2
    cy_cam = FRAME_H // 2

    # Lock state lives here now — single source of truth
    locked = False  # <-- CHANGED: moved from tracking_loop

    _camera_ready.set()  # signal that startup succeeded

    try:
        while _running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                aligned = align.process(frames)

                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()

                if not cf or not df:
                    continue

                frame = np.asanyarray(cf.get_data())

                results = model(frame, verbose=False, imgsz=YOLO_IMGSZ)
                faces = []

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        face_cx_raw = (x1 + x2) // 2
                        aim_cx = face_cx_raw
                        aim_cy = int(y1 + (y2 - y1) * HEADSHOT_Y_RATIO)

                        face_cx = clamp(face_cx_raw, 0, FRAME_W - 1)
                        face_cy = clamp((y1 + y2) // 2, 0, FRAME_H - 1)

                        dist_m = df.get_distance(face_cx, face_cy)
                        if dist_m == 0.0:
                            dist_m = 9.9

                        # Apply both distance gates here (was: MIN only in tracking_loop)
                        if MIN_DISTANCE_M < dist_m <= MAX_DISTANCE_M:  # <-- CHANGED
                            faces.append((dist_m, aim_cx, aim_cy, x1, y1, x2, y2))

                faces.sort(key=lambda f: f[0])

                # --- Compute lock state once, here ---  <-- CHANGED
                if faces:
                    _, fcx, fcy, *_ = faces[0]
                    total_err = abs(fcx - cx_cam) + abs(fcy - cy_cam)
                    if not locked and total_err < LOCK_PX:
                        locked = True
                    elif locked and total_err > UNLOCK_PX:
                        locked = False
                else:
                    locked = False

                # --- HUD ---
                annotated = frame.copy()

                for i, (dist, aim_cx, aim_cy, x1, y1, x2, y2) in enumerate(faces):
                    is_target = i == 0
                    too_close = dist <= MIN_DISTANCE_M

                    if too_close:
                        col = COL_WARNING
                    elif is_target:
                        col = COL_LOCKED if locked else COL_TARGET
                    else:
                        col = COL_OTHER

                    th = 2 if is_target else 1

                    draw_corner_box(annotated, x1, y1, x2, y2, col, thickness=th)
                    draw_crosshair(annotated, aim_cx, aim_cy, col, size=10, gap=3, thickness=th)

                    if is_target:
                        cv2.line(annotated, (cx_cam, cy_cam), (aim_cx, aim_cy), COL_LINE, 1, cv2.LINE_AA)

                    status = " LOCKED" if (is_target and locked) else ""
                    cv2.putText(
                        annotated,
                        f"{dist:.2f}m{status}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        col,
                        1,
                        cv2.LINE_AA
                    )

                mode_txt = "LOCKED" if locked else "TRACKING"
                mode_col = COL_LOCKED if locked else COL_TARGET

                cv2.putText(
                    annotated,
                    mode_txt,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    mode_col,
                    2,
                    cv2.LINE_AA
                )

                plat = "Pi" if IS_PI else "Win/Linux"
                cv2.putText(
                    annotated,
                    f"{plat} | {DEVICE.upper()} | {YOLO_IMGSZ}px | {CAM_FPS}FPS",
                    (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA
                )

                if TEST_MODE:
                    cv2.putText(
                        annotated,
                        "TEST MODE",
                        (FRAME_W - 130, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                draw_camera_crosshair(annotated, COL_CROSS)

                # Publish faces, frame, and lock state as one atomic tuple
                with _lock:
                    _state = (faces, annotated, locked)  # <-- CHANGED

            except Exception as e:
                print("Camera loop warning:", e)
                time.sleep(0.2)

    finally:
        pipeline.stop()
        print("RealSense stopped.")


# ============================================================
# TRACKING LOOP
# ============================================================

def tracking_loop(arm):
    global _running

    cx_cam = FRAME_W / 2.0
    cy_cam = FRAME_H / 2.0

    frozen = get_joints(arm)
    if not frozen:
        print("ERROR: Joint read failed.")
        _running = False
        return

    cur_j1 = frozen[0]
    cur_j5 = frozen[4]
    vel_j1 = 0.0
    vel_j5 = 0.0
    scan_dir = 1

    title = "Face Tracker [TEST]" if TEST_MODE else "Face Tracker"
    print("Tracking started. Press Q to quit.")

    while _running:
        t0 = time.time()

        # Read all shared state atomically — locked comes from camera thread now
        with _lock:
            faces, frame, locked = _state  # <-- CHANGED: locked read from shared state

        if SHOW_WINDOW and frame is not None:
            cv2.imshow(title, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                _running = False
                break

        if faces:
            dist, cx, cy, *_ = faces[0]

            if dist > MIN_DISTANCE_M:
                err_x = cx - cx_cam
                err_y = cy - cy_cam

                # Use lock state from camera thread — no recomputation
                if locked:  # <-- CHANGED: was recomputed here before
                    yaw_gain = YAW_GAIN_LOCK
                    pitch_gain = PITCH_GAIN_LOCK
                    yaw_alpha = YAW_ALPHA_LOCK
                    pitch_alpha = PITCH_ALPHA_LOCK
                    max_yaw = MAX_YAW_DEG_LOCK
                    max_pitch = MAX_PITCH_DEG_LOCK
                    db_x, db_y = 8, 8
                else:
                    yaw_gain = YAW_GAIN_TRACK
                    pitch_gain = PITCH_GAIN_TRACK
                    yaw_alpha = YAW_ALPHA_TRACK
                    pitch_alpha = PITCH_ALPHA_TRACK
                    max_yaw = MAX_YAW_DEG
                    max_pitch = MAX_PITCH_DEG
                    db_x, db_y = DEAD_BAND_X, DEAD_BAND_Y

                if abs(err_x) < db_x:
                    err_x = 0.0
                if abs(err_y) < db_y:
                    err_y = 0.0

                vel_j1 = lerp(
                    vel_j1,
                    clamp(yaw_gain * err_x, -max_yaw, max_yaw),
                    yaw_alpha
                )

                vel_j5 = lerp(
                    vel_j5,
                    clamp(pitch_gain * err_y, -max_pitch, max_pitch),
                    pitch_alpha
                )
            else:
                vel_j1 = lerp(vel_j1, 0.0, 0.35)
                vel_j5 = lerp(vel_j5, 0.0, 0.35)

            scan_dir = 1

        else:
            vel_j1 = SCAN_SPEED_DEG * scan_dir

            j5_err = 50.0 - cur_j5
            vel_j5 = lerp(vel_j5, clamp(j5_err * 0.02, -0.3, 0.3), 0.10)

            if cur_j1 >= SCAN_J1_MAX:
                scan_dir = -1
            elif cur_j1 <= SCAN_J1_MIN:
                scan_dir = 1

        cur_j1 = clamp(cur_j1 + vel_j1, J1_MIN, J1_MAX)
        cur_j5 = clamp(cur_j5 + vel_j5, J5_MIN, J5_MAX)

        frozen[0] = cur_j1
        frozen[4] = cur_j5

        try:
            arm.set_servo_angle_j(frozen[:], speed=SERVO_SPEED, mvacc=SERVO_ACC)
        except Exception as e:
            print("Arm command warning:", e)
            time.sleep(0.1)

        elapsed = time.time() - t0
        time.sleep(max(0, DT - elapsed))


# ============================================================
# MAIN
# ============================================================

def main():
    global _running

    print("========================================")
    print("Face Tracker")
    print("========================================")
    print(f"Platform : {'Raspberry Pi' if IS_PI else platform.system()}")
    print(f"Machine  : {platform.machine()}")
    print(f"Device   : {DEVICE.upper()}")
    print(f"YOLO size: {YOLO_IMGSZ}")
    print(f"Camera   : {FRAME_W}x{FRAME_H} @ {CAM_FPS} FPS")
    print(f"Display  : {'ON' if SHOW_WINDOW else 'OFF'}")
    print(f"Mode     : {'TEST, no arm' if TEST_MODE else 'LIVE, arm connected'}")
    print("========================================")

    arm = None

    try:
        arm = connect_arm()

        print("Moving to home...")
        go_home(arm)

        enable_servo(arm)

        cam_t = threading.Thread(target=realsense_thread, daemon=True)
        cam_t.start()

        print("Waiting for camera and model...")
        _camera_ready.wait(timeout=30)  # <-- CHANGED: event-based wait, not sleep

        if not _running:
            print("Startup failed.")
            return

        tracking_loop(arm)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print("Fatal error:", e)

    finally:
        _running = False
        time.sleep(0.3)

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

        print("Done.")


if __name__ == "__main__":
    main()