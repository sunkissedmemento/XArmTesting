"""
xArm 6 — Gamepad Controller
═══════════════════════════════════════════════════════════════
Based on the working keyboard reference — uses mode 1 +
set_servo_cartesian with a velocity ramp for smooth real-time
control, identical to the UFactory Studio jogging pads.

LEFT  STICK  → X / Y
LB  / LT     → Z-  / Z+
RIGHT STICK  → RX / RY  (roll / pitch)
RB  / RT     → RZ- / RZ+ (yaw)

B            → Emergency stop
START        → Return to home
SELECT       → Toggle motion enable

Install: pip install inputs xarm-python-sdk
"""

import os
import time
import threading
from pathlib import Path
from inputs import get_gamepad
from xarm.wrapper import XArmAPI

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
ROBOT_IP = "192.168.1.207"

HOME_POS    = [206.3, -1.1, 112.5, -180, 0, 0]
INITIAL_POS = [0, 0, 0, 0, 0, 0]       # SELECT button destination
HOME_SPEED = 100
HOME_ACC   = 500

SPEED_PCT  = 3.0       # 0.0 – 1.0

LIN_VEL_MAX = 40.0 * SPEED_PCT   # mm/s  at full stick
ANG_VEL_MAX = 15.0 * SPEED_PCT   # deg/s at full stick/trigger

DT   = 0.03     # seconds per tick  (~33 Hz, same as reference)
RAMP = 0.25     # velocity smoothing  (0=sluggish, 1=instant)

# Safety bounds (mm)
X_MIN, X_MAX = -700, 700
Y_MIN, Y_MAX = -700, 700
Z_MIN, Z_MAX =   50, 700

STICK_DEADZONE   = 0.08
TRIGGER_DEADZONE = 0.02
STICK_MAX        = 32767
TRIGGER_MAX      = 255


# ══════════════════════════════════════════════════════════════
#  SDK LOG DIR FIX  (from reference)
# ══════════════════════════════════════════════════════════════
def ensure_xarm_log_dir():
    sdk_path = Path(os.path.expanduser("~")) / ".UFACTORY" / "log" / "xarm" / "sdk"
    if sdk_path.exists() and not sdk_path.is_dir():
        sdk_path.rename(sdk_path.with_name(f"sdk_backup_{int(time.time())}"))
    sdk_path.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0

def norm_stick(val):
    n = val / STICK_MAX
    return 0.0 if abs(n) < STICK_DEADZONE else max(-1.0, min(1.0, n))

def norm_trigger(val):
    n = val / TRIGGER_MAX
    return 0.0 if n < TRIGGER_DEADZONE else min(1.0, n)


# ══════════════════════════════════════════════════════════════
#  CONTROLLER STATE  (background thread)
# ══════════════════════════════════════════════════════════════
class ControllerState:
    def __init__(self):
        self.LX = self.LY = 0.0
        self.RX = self.RY = 0.0
        self.LT = self.RT = 0.0
        self.LB = self.RB = False
        self.A  = self.B  = False
        self.X  = self.Y  = False
        self.START = self.SELECT = False
        self.DPAD_X = self.DPAD_Y = 0
        self._prev    = {}
        self._lock    = threading.Lock()
        self._running = False

    def _apply(self, ev_type, code, state):
        with self._lock:
            if ev_type == "Key":
                m = {"BTN_SOUTH": "A",  "BTN_EAST":   "B",
                     "BTN_WEST":  "X",  "BTN_NORTH":  "Y",
                     "BTN_TL":    "LB", "BTN_TR":     "RB",
                     "BTN_START": "START", "BTN_SELECT": "SELECT"}
                if code in m:
                    setattr(self, m[code], bool(state))
            elif ev_type == "Absolute":
                if   code == "ABS_X":     self.LX = norm_stick(state)
                elif code == "ABS_Y":     self.LY = norm_stick(state)
                elif code == "ABS_RX":    self.RX = norm_stick(state)
                elif code == "ABS_RY":    self.RY = norm_stick(state)
                elif code == "ABS_Z":     self.LT = norm_trigger(state)
                elif code == "ABS_RZ":    self.RT = norm_trigger(state)
                elif code == "ABS_HAT0X": self.DPAD_X = state
                elif code == "ABS_HAT0Y": self.DPAD_Y = state

    def _poll(self):
        while self._running:
            try:
                for e in get_gamepad():
                    self._apply(e.ev_type, e.code, e.state)
            except Exception:
                time.sleep(0.02)

    def start(self):
        self._running = True
        threading.Thread(target=self._poll, daemon=True).start()

    def stop(self):
        self._running = False

    def pressed(self, btn):
        cur  = getattr(self, btn)
        prev = self._prev.get(btn, False)
        self._prev[btn] = cur
        return cur and not prev


# ══════════════════════════════════════════════════════════════
#  xARM SETUP  (mirrors reference exactly)
# ══════════════════════════════════════════════════════════════
def connect_arm():
    print(f"\n🦾  Connecting to xArm at {ROBOT_IP}...")
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)

    # Mode 1 = servo streaming (same as reference)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.2)

    print("✅  Connected (mode 1 servo).\n")
    return arm


def go_home(arm):
    """Switch to mode 0, move home, then return to mode 1."""
    print("\n🏠  Going home...")
    arm.set_state(4);  time.sleep(0.3)
    arm.set_mode(0);   time.sleep(0.3)
    arm.set_state(0);  time.sleep(0.3)
    arm.set_position(*HOME_POS, speed=HOME_SPEED, mvacc=HOME_ACC,
                     radius=-1, wait=True)
    # Back to servo mode
    arm.set_state(4);  time.sleep(0.3)
    arm.set_mode(1);   time.sleep(0.3)
    arm.set_state(0);  time.sleep(0.3)
    print("✅  At home — servo mode active.\n")


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
def run():
    ensure_xarm_log_dir()

    arm  = connect_arm()
    ctrl = ControllerState()
    ctrl.start()

    go_home(arm)

    print("═" * 62)
    print(f"  Mode 1 servo  |  {int(1/DT)} Hz  |  Speed: {int(SPEED_PCT*100)}%")
    print("  LEFT  STICK     →  X / Y")
    print("  LB  / LT        →  Z-  /  Z+")
    print("  RIGHT STICK     →  RX / RY  (roll / pitch)")
    print("  RB  / RT        →  RZ- / RZ+  (yaw)")
    print("  B               →  Emergency stop")
    print("  START / X       →  Return home")
    print("  SELECT          →  Go to initial position [0,0,0,0,0,0]")
    print("═" * 62 + "\n")

    # Seed position
    code, pose = arm.get_position()
    if code != 0 or pose is None:
        print(f"⚠️  Could not read position (code={code}), using HOME_POS.")
        pose = list(HOME_POS)
    x, y, z, roll, pitch, yaw = pose[:6]

    # Smoothed velocities (same ramp system as reference)
    vx = vy = vz = 0.0
    vroll = vpitch = vyaw = 0.0

    enabled = True

    try:
        while True:

            # ── Button events ─────────────────────────────────
            if ctrl.pressed("B"):
                print("\n🛑  Emergency stop!")
                arm.emergency_stop()
                break

            if ctrl.pressed("START") or ctrl.pressed("X"):
                enabled = True
                go_home(arm)
                code, pose = arm.get_position()
                if code == 0 and pose:
                    x, y, z, roll, pitch, yaw = pose[:6]
                vx = vy = vz = vroll = vpitch = vyaw = 0.0

            if ctrl.pressed("SELECT"):
                print("\n🔄  Going to initial position...")
                arm.set_state(4);  time.sleep(0.3)
                arm.set_mode(0);   time.sleep(0.3)
                arm.set_state(0);  time.sleep(0.3)
                arm.set_position(*INITIAL_POS, speed=HOME_SPEED, mvacc=HOME_ACC,
                                 radius=-1, wait=True)
                arm.set_state(4);  time.sleep(0.3)
                arm.set_mode(1);   time.sleep(0.3)
                arm.set_state(0);  time.sleep(0.3)
                code, pose = arm.get_position()
                if code == 0 and pose:
                    x, y, z, roll, pitch, yaw = pose[:6]
                vx = vy = vz = vroll = vpitch = vyaw = 0.0
                enabled = True
                print("✅  At initial position.\n")

            # ── Target velocities from controller ────────────
            #   Left stick  → X (fwd/back),  Y (left/right)
            #   LT trigger  → Z+  |  LB bumper → Z-
            #   Right stick → roll, pitch
            #   RT trigger  → yaw+ | RB bumper → yaw-
            tvx    =  ctrl.LX * LIN_VEL_MAX
            tvy    = -ctrl.LY * LIN_VEL_MAX      # invert Y
            tvz    = (ctrl.LT - (1.0 if ctrl.LB else 0.0)) * LIN_VEL_MAX

            tvroll  =  ctrl.RX * ANG_VEL_MAX
            tvpitch = -ctrl.RY * ANG_VEL_MAX      # invert Y
            tvyaw   = (ctrl.RT - (1.0 if ctrl.RB else 0.0)) * ANG_VEL_MAX

            # ── Velocity ramp (smooth like reference) ────────
            vx     += (tvx     - vx)     * RAMP
            vy     += (tvy     - vy)     * RAMP
            vz     += (tvz     - vz)     * RAMP
            vroll  += (tvroll  - vroll)  * RAMP
            vpitch += (tvpitch - vpitch) * RAMP
            vyaw   += (tvyaw   - vyaw)   * RAMP

            # ── Integrate velocity → position ─────────────────
            x     += vx     * DT
            y     += vy     * DT
            z     += vz     * DT
            roll  += vroll  * DT
            pitch += vpitch * DT
            yaw   += vyaw   * DT

            # ── Safety clamps ─────────────────────────────────
            x     = clamp(x, X_MIN, X_MAX)
            y     = clamp(y, Y_MIN, Y_MAX)
            z     = clamp(z, Z_MIN, Z_MAX)
            roll  = wrap_angle(roll)
            pitch = wrap_angle(pitch)
            yaw   = wrap_angle(yaw)

            # ── Send (mode 1 servo — no queue, real-time) ─────
            code = arm.set_servo_cartesian([x, y, z, roll, pitch, yaw])
            if code != 0:
                print(f"\n[ERROR] set_servo_cartesian failed (code={code}) — resetting velocities")
                vx = vy = vz = vroll = vpitch = vyaw = 0.0
                # Re-sync position
                r, p = arm.get_position()
                if r == 0 and p:
                    x, y, z, roll, pitch, yaw = p[:6]

            print(
                f"\r  X={x:7.1f} Y={y:7.1f} Z={z:7.1f} | "
                f"R={roll:6.1f} P={pitch:6.1f} Yw={yaw:6.1f}   ",
                end="", flush=True
            )

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\n\n👋  Interrupted by user.")
    finally:
        ctrl.stop()
        try:
            arm.set_state(4)
            arm.disconnect()
        except Exception:
            pass
        print("🔌  Disconnected.")


if __name__ == "__main__":
    run()