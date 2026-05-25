"""
Controller Reader — mapped for Xbox-style gamepad via `inputs` library.
Install: pip install inputs

Provides a clean ControllerState object updated in a background thread,
ready to integrate with xArm or any other control loop.
"""

import threading
import time
from inputs import get_gamepad

# ── Axis normalization ────────────────────────────────────────────────────────
STICK_MAX   = 32767   # sticks:   -32768 .. 32767  → -1.0 .. 1.0
TRIGGER_MAX = 255     # triggers:     0  ..   255  →  0.0 .. 1.0
STICK_DEADZONE    = 0.08   # ignore small stick drift
TRIGGER_DEADZONE  = 0.02

def norm_stick(val):
    n = val / STICK_MAX
    return 0.0 if abs(n) < STICK_DEADZONE else max(-1.0, min(1.0, n))

def norm_trigger(val):
    n = val / TRIGGER_MAX
    return 0.0 if n < TRIGGER_DEADZONE else min(1.0, n)


# ── Controller state ──────────────────────────────────────────────────────────
class ControllerState:
    def __init__(self):
        # Buttons  (True = pressed)
        self.A       = False   # BTN_SOUTH
        self.B       = False   # BTN_EAST
        self.X       = False   # BTN_WEST
        self.Y       = False   # BTN_NORTH
        self.LB      = False   # BTN_TL
        self.RB      = False   # BTN_TR
        self.START   = False   # BTN_START
        self.SELECT  = False   # BTN_SELECT

        # Triggers  (0.0 – 1.0)
        self.LT = 0.0          # ABS_Z
        self.RT = 0.0          # ABS_RZ

        # Sticks    (-1.0 – 1.0)
        self.LX = 0.0          # ABS_X   (left  = -1, right = +1)
        self.LY = 0.0          # ABS_Y   (up    = -1, down  = +1)
        self.RX = 0.0          # ABS_RX
        self.RY = 0.0          # ABS_RY

        # D-Pad     (-1, 0, 1)
        self.DPAD_X = 0        # ABS_HAT0X  (left=-1, right=+1)
        self.DPAD_Y = 0        # ABS_HAT0Y  (up=-1,   down=+1)

        self._lock = threading.Lock()
        self._running = False
        self._thread  = None

    # ── Apply a raw event ────────────────────────────────────────────────────
    def _apply(self, ev_type, code, state):
        with self._lock:
            if ev_type == "Key":
                btn_map = {
                    "BTN_SOUTH":  "A",
                    "BTN_EAST":   "B",
                    "BTN_WEST":   "X",
                    "BTN_NORTH":  "Y",
                    "BTN_TL":     "LB",
                    "BTN_TR":     "RB",
                    "BTN_START":  "START",
                    "BTN_SELECT": "SELECT",
                }
                if code in btn_map:
                    setattr(self, btn_map[code], bool(state))

            elif ev_type == "Absolute":
                if   code == "ABS_X":     self.LX = norm_stick(state)
                elif code == "ABS_Y":     self.LY = norm_stick(state)
                elif code == "ABS_RX":    self.RX = norm_stick(state)
                elif code == "ABS_RY":    self.RY = norm_stick(state)
                elif code == "ABS_Z":     self.LT = norm_trigger(state)
                elif code == "ABS_RZ":    self.RT = norm_trigger(state)
                elif code == "ABS_HAT0X": self.DPAD_X = state
                elif code == "ABS_HAT0Y": self.DPAD_Y = state

    # ── Background polling thread ────────────────────────────────────────────
    def _poll(self):
        while self._running:
            try:
                events = get_gamepad()
                for e in events:
                    self._apply(e.ev_type, e.code, e.state)
            except Exception:
                time.sleep(0.1)

    def start(self):
        """Start background polling."""
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        print("Controller polling started.")

    def stop(self):
        self._running = False

    # ── Snapshot helper ──────────────────────────────────────────────────────
    def snapshot(self):
        with self._lock:
            return {
                "buttons": {
                    "A": self.A, "B": self.B, "X": self.X, "Y": self.Y,
                    "LB": self.LB, "RB": self.RB,
                    "START": self.START, "SELECT": self.SELECT,
                },
                "triggers": {"LT": self.LT, "RT": self.RT},
                "left_stick":  {"x": self.LX, "y": self.LY},
                "right_stick": {"x": self.RX, "y": self.RY},
                "dpad": {"x": self.DPAD_X, "y": self.DPAD_Y},
            }

    def __repr__(self):
        s = self.snapshot()
        buttons = [k for k, v in s["buttons"].items() if v]
        return (
            f"Buttons: {buttons or 'none'} | "
            f"LT={s['triggers']['LT']:.2f} RT={s['triggers']['RT']:.2f} | "
            f"LS=({s['left_stick']['x']:+.2f},{s['left_stick']['y']:+.2f}) "
            f"RS=({s['right_stick']['x']:+.2f},{s['right_stick']['y']:+.2f}) | "
            f"DPAD=({s['dpad']['x']},{s['dpad']['y']})"
        )


# ── Usage examples ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ctrl = ControllerState()
    ctrl.start()

    print("Reading controller — press Ctrl+C to stop.\n")

    # Example 1: simple print loop
    try:
        while True:
            print(f"\r{ctrl}", end="", flush=True)
            time.sleep(0.05)   # 20 Hz display refresh
    except KeyboardInterrupt:
        ctrl.stop()
        print("\nDone.")


# ── Integration snippet for xArm ─────────────────────────────────────────────
# 
# from controller_reader import ControllerState
#
# ctrl = ControllerState()
# ctrl.start()
#
# while True:
#     # Map right stick X/Y → xArm X/Y velocity
#     vx = ctrl.RX * 100   # mm/s scale
#     vy = ctrl.RY * 100
#     vz = (ctrl.RT - ctrl.LT) * 100   # triggers → up/down
#
#     if ctrl.B:   # B button → stop
#         arm.emergency_stop()
#         break
#
#     arm.set_position(x=HOME_POS[0] + vx,
#                      y=HOME_POS[1] + vy,
#                      z=HOME_POS[2] + vz,
#                      speed=50, wait=False)
#     time.sleep(0.05)