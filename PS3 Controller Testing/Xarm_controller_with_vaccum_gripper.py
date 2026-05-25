#!/usr/bin/env python3
"""
xArm 6 — Gamepad Controller + Vacuum Control
"""

import os
import time
import threading
from pathlib import Path
from inputs import get_gamepad
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.207"

HOME_POS    = [234.1, 7.8, 311.0, 179.8, -1.4, -0.1]
INITIAL_POS = [234.1, 7.8, 311.0, 179.8, -1.4, -0.1]

SPEED_PCT   = 3.0
LIN_VEL_MAX = 40.0 * SPEED_PCT
ANG_VEL_MAX = 15.0 * SPEED_PCT

DT   = 0.03
RAMP = 0.25

X_MIN, X_MAX = -700, 700
Y_MIN, Y_MAX = -700, 700
Z_MIN, Z_MAX = 50, 700

HOME_SPEED = 100
HOME_ACC   = 300

STICK_DEADZONE   = 0.08
TRIGGER_DEADZONE = 0.02
STICK_MAX        = 32767
TRIGGER_MAX      = 255

BTN_CLEAR_ERRORS = "SELECT"
BTN_HOME         = "START"
BTN_ESTOP        = "Y"
BTN_VACUUM_ON    = "A"
BTN_VACUUM_OFF   = "B"
BTN_STOP_PROGRAM = "X"


def ensure_xarm_log_dir():
    sdk_path = Path(os.path.expanduser("~")) / ".UFACTORY" / "log" / "xarm" / "sdk"
    if sdk_path.exists() and not sdk_path.is_dir():
        sdk_path.rename(sdk_path.with_name(f"sdk_backup_{int(time.time())}"))
    sdk_path.mkdir(parents=True, exist_ok=True)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0


def norm_stick(val):
    n = val / STICK_MAX
    return 0.0 if abs(n) < STICK_DEADZONE else max(-1.0, min(1.0, n))


def norm_trigger(val):
    n = val / TRIGGER_MAX
    return 0.0 if n < TRIGGER_DEADZONE else min(1.0, n)


class ControllerState:
    def __init__(self):
        self.LX = self.LY = 0.0
        self.RX = self.RY = 0.0
        self.LT = self.RT = 0.0
        self.LB = self.RB = False
        self.A = self.B = False
        self.X = self.Y = False
        self.START = self.SELECT = False
        self._prev = {}
        self._lock = threading.Lock()
        self._running = False

    def _apply(self, ev_type, code, state):
        with self._lock:
            if ev_type == "Key":
                mapping = {
                    "BTN_SOUTH": "A",
                    "BTN_EAST": "B",
                    "BTN_WEST": "X",
                    "BTN_NORTH": "Y",
                    "BTN_TL": "LB",
                    "BTN_TR": "RB",
                    "BTN_START": "START",
                    "BTN_SELECT": "SELECT",
                }
                if code in mapping:
                    setattr(self, mapping[code], bool(state))

            elif ev_type == "Absolute":
                if code == "ABS_X":
                    self.LX = norm_stick(state)
                elif code == "ABS_Y":
                    self.LY = norm_stick(state)
                elif code == "ABS_RX":
                    self.RX = norm_stick(state)
                elif code == "ABS_RY":
                    self.RY = norm_stick(state)
                elif code == "ABS_Z":
                    self.LT = norm_trigger(state)
                elif code == "ABS_RZ":
                    self.RT = norm_trigger(state)

    def _poll(self):
        while self._running:
            try:
                events = get_gamepad()
                for e in events:
                    self._apply(e.ev_type, e.code, e.state)
            except Exception:
                time.sleep(0.02)

    def start(self):
        self._running = True
        threading.Thread(target=self._poll, daemon=True).start()

    def stop(self):
        self._running = False

    def pressed(self, btn_name):
        cur = getattr(self, btn_name)
        prev = self._prev.get(btn_name, False)
        self._prev[btn_name] = cur
        return cur and not prev


def vacuum_on(arm):
    try:
        arm.set_digital_output(1, 1)
        print("\nVacuum ON")
        return 0
    except Exception:
        pass

    try:
        return arm.set_suction_cup(True, wait=False)
    except Exception:
        pass

    try:
        return arm.set_vacuum_gripper(True, wait=False)
    except Exception:
        pass

    print("\nVacuum ON failed")
    return -1


def vacuum_off(arm):
    try:
        arm.set_digital_output(1, 0)
        print("\nVacuum OFF")
        return 0
    except Exception:
        pass

    try:
        return arm.set_suction_cup(False, wait=False)
    except Exception:
        pass

    try:
        return arm.set_vacuum_gripper(False, wait=False)
    except Exception:
        pass

    print("\nVacuum OFF failed")
    return -1


def clear_all_errors(arm):
    print("\nClearing errors/warnings...")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.2)
    print("Errors cleared")


def connect_arm():
    print(f"\nConnecting to xArm at {ROBOT_IP}...")
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.2)
    print("Connected – servo mode active")
    return arm


def move_to_position(arm, pose, speed=HOME_SPEED, acc=HOME_ACC, wait=True):
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)

    code = arm.set_position(
        *pose,
        speed=speed,
        mvacc=acc,
        radius=-1,
        wait=wait
    )

    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.1)
    return code


def go_home(arm):
    print("\nGoing home...")
    move_to_position(arm, HOME_POS, wait=True)
    print("At home")


def go_initial(arm):
    print("\nGoing to initial position...")
    move_to_position(arm, INITIAL_POS, wait=True)
    print("At initial position")


def run():
    ensure_xarm_log_dir()

    arm = connect_arm()
    ctrl = ControllerState()
    ctrl.start()

    go_initial(arm)

    code, pose = arm.get_position()
    if code != 0 or pose is None:
        print(f"Cannot read position, code={code}. Using INITIAL_POS.")
        x, y, z, roll, pitch, yaw = INITIAL_POS[:6]
    else:
        x, y, z, roll, pitch, yaw = pose[:6]

    vx = vy = vz = 0.0
    vroll = vpitch = vyaw = 0.0

    print("=" * 70)
    print("xArm 6 Gamepad Control – Servo Mode")
    print(f"Loop: {int(1 / DT)} Hz | Speed: {SPEED_PCT:.0f}%")
    print("LEFT STICK  -> X / Y")
    print("LB / LT      -> Z- / Z+")
    print("RIGHT STICK -> Roll / Pitch")
    print("RB / RT      -> Yaw- / Yaw+")
    print("Cross        -> Vacuum ON")
    print("Circle       -> Vacuum OFF")
    print("Triangle     -> Emergency stop")
    print("Square       -> Stop program")
    print("START        -> Go home")
    print("SELECT       -> Clear errors")
    print("=" * 70)

    try:
        while True:
            if ctrl.pressed(BTN_ESTOP):
                print("\nEMERGENCY STOP!")
                arm.emergency_stop()
                break

            if ctrl.pressed(BTN_STOP_PROGRAM):
                print("\nStop requested.")
                break

            if ctrl.pressed(BTN_CLEAR_ERRORS):
                clear_all_errors(arm)
                code, pose = arm.get_position()
                if code == 0 and pose:
                    x, y, z, roll, pitch, yaw = pose[:6]
                vx = vy = vz = vroll = vpitch = vyaw = 0.0

            if ctrl.pressed(BTN_HOME):
                go_home(arm)
                code, pose = arm.get_position()
                if code == 0 and pose:
                    x, y, z, roll, pitch, yaw = pose[:6]
                vx = vy = vz = vroll = vpitch = vyaw = 0.0

            if ctrl.pressed(BTN_VACUUM_ON):
                vacuum_on(arm)

            if ctrl.pressed(BTN_VACUUM_OFF):
                vacuum_off(arm)

            target_vx = ctrl.LX * LIN_VEL_MAX
            target_vy = -ctrl.LY * LIN_VEL_MAX

            target_vz = 0.0
            if ctrl.LB:
                target_vz = -LIN_VEL_MAX
            if ctrl.LT > 0:
                target_vz = ctrl.LT * LIN_VEL_MAX

            target_vroll = ctrl.RX * ANG_VEL_MAX
            target_vpitch = -ctrl.RY * ANG_VEL_MAX

            target_vyaw = 0.0
            if ctrl.RB:
                target_vyaw = -ANG_VEL_MAX
            if ctrl.RT > 0:
                target_vyaw = ctrl.RT * ANG_VEL_MAX

            vx += (target_vx - vx) * RAMP
            vy += (target_vy - vy) * RAMP
            vz += (target_vz - vz) * RAMP
            vroll += (target_vroll - vroll) * RAMP
            vpitch += (target_vpitch - vpitch) * RAMP
            vyaw += (target_vyaw - vyaw) * RAMP

            x += vx * DT
            y += vy * DT
            z += vz * DT
            roll += vroll * DT
            pitch += vpitch * DT
            yaw += vyaw * DT

            x = clamp(x, X_MIN, X_MAX)
            y = clamp(y, Y_MIN, Y_MAX)
            z = clamp(z, Z_MIN, Z_MAX)
            roll = wrap_angle(roll)
            pitch = wrap_angle(pitch)
            yaw = wrap_angle(yaw)

            pose_cmd = [x, y, z, roll, pitch, yaw]

            code = arm.set_servo_cartesian(
                pose_cmd,
                speed=LIN_VEL_MAX,
                acc=300,
                is_radian=False
            )

            if code != 0:
                print(f"\n[ERROR] set_servo_cartesian code={code}")
                vx = vy = vz = vroll = vpitch = vyaw = 0.0

                code2, pose2 = arm.get_position()
                if code2 == 0 and pose2:
                    x, y, z, roll, pitch, yaw = pose2[:6]

            print(
                f"\rX={x:7.1f} Y={y:7.1f} Z={z:7.1f} | "
                f"R={roll:6.1f} P={pitch:6.1f} Yaw={yaw:6.1f}",
                end="",
                flush=True
            )

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        ctrl.stop()
        try:
            arm.set_state(4)
            arm.disconnect()
        except Exception:
            pass
        print("\nDisconnected from robot")


if __name__ == "__main__":
    run()