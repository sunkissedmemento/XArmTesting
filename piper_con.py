import time
from piper_sdk import *
import pygame

# -----------------------------
# Initialize PiPER
# -----------------------------
piper = C_PiperInterface(can_name="can0", can_auto_init=True)
piper.ConnectPort()
time.sleep(1)

piper.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)
time.sleep(1)
piper.EnableArm(motor_num=7, enable_flag=0x02)
time.sleep(1)

# -----------------------------
# Read current joint positions
# -----------------------------
joint_msg = piper.GetArmJointMsgs()
joint = [
    joint_msg.joint_state.joint_1,
    joint_msg.joint_state.joint_2,
    joint_msg.joint_state.joint_3,
    joint_msg.joint_state.joint_4,
    joint_msg.joint_state.joint_5,
    joint_msg.joint_state.joint_6,
]

# Prime controller at current position
piper.JointCtrl(
    joint_1=joint[0],
    joint_2=joint[1],
    joint_3=joint[2],
    joint_4=joint[3],
    joint_5=joint[4],
    joint_6=joint[5],
)
time.sleep(0.2)
print("Starting joints:", joint)

# -----------------------------
# Control parameters
# -----------------------------
DT = 0.05      # 20 Hz
STEP = 800     # step per press
gripper_open = False
gripper_dirty = True
running = True

# -----------------------------
# Joint limits
# -----------------------------
JOINT_LIMITS = [
    (-154685, 155226),
    (-1841, 194970),
    (-173572, 1491),
    (-102592, 101683),
    (-75034, 75238),
    (-121287, 121263),
]

# -----------------------------
# Initialize joystick
# -----------------------------
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()
print("Controller detected:", js.get_name())

# -----------------------------
# Helper: round axis to -1, 0, 1
# -----------------------------
def round_axis(val, tol=0.1):
    """Round axis value to -1, 0, 1 based on tolerance"""
    if val < -1 + tol:
        return -1
    elif val > 1 - tol:
        return 1
    else:
        return 0

# -----------------------------
# Main loop
# -----------------------------
try:
    while running:
        pygame.event.pump()

        axes = [js.get_axis(i) for i in range(js.get_numaxes())]
        buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

        # -----------------------------
        # Map axes to joints ONLY if full press (-1 or 1)
        # -----------------------------
        # Axis 0 → Joint 1
        if round(axes[0]) == -1:
            joint[0] += STEP
        elif round(axes[0]) == 1:
            joint[0] -= STEP


        # Axis 1 → Joint 2
        if round(axes[1]) == -1:
            joint[1] += STEP
        elif round(axes[1]) == 1:
            joint[1] -= STEP

        # Axis 3 → Joint 4
        if round(axes[3]) == -1:
            joint[3] += STEP
        elif round(axes[3]) == 1:
            joint[3] -= STEP

        # Axis 4 → Joint 3
        if round(axes[4]) == -1:
            joint[2] += STEP
        elif round(axes[4]) == 1:
            joint[2] -= STEP

        # Read D-pad (HAT)
        hat_x, hat_y = js.get_hat(0)

        # Joint 5 (Up / Down)
        if hat_y == 1:
            joint[4] += STEP
        elif hat_y == -1:
            joint[4] -= STEP

        # Joint 6 (Left / Right)
        if hat_x == -1:
            joint[5] -= STEP
        elif hat_x == 1:
            joint[5] += STEP

        # Gripper toggle: ✕ button
        if buttons[0] and not gripper_dirty:
            gripper_open = not gripper_open
            gripper_dirty = True
        if not buttons[0]:
            gripper_dirty = False

        # PS button → exit
        if buttons[8]:
            print("Exiting via PS button")
            break

        # -----------------------------
        # Apply Joint limits
        # -----------------------------
        for i in range(6):
            joint[i] = max(JOINT_LIMITS[i][0], min(JOINT_LIMITS[i][1], joint[i]))

        # -----------------------------
        # Gripper control
        # -----------------------------
        if gripper_open:
            piper.GripperCtrl(gripper_angle=45_000, gripper_effort=2000, gripper_code=0x03)
        else:
            piper.GripperCtrl(gripper_angle=0, gripper_effort=2000, gripper_code=0x03)

        # -----------------------------
        # Send joint command
        # -----------------------------
        piper.JointCtrl(
            joint_1=joint[0],
            joint_2=joint[1],
            joint_3=joint[2],
            joint_4=joint[3],
            joint_5=joint[4],
            joint_6=joint[5],
        )

        time.sleep(DT)

finally:
    # -----------------------------
    # Shutdown safely
    # -----------------------------
    piper.GripperCtrl(gripper_angle=0, gripper_effort=2000, gripper_code=0x03)
    piper.JointCtrl(joint_1=0, joint_2=0, joint_3=0, joint_4=0, joint_5=18_000, joint_6=0)
    time.sleep(3)
    piper.DisableArm(motor_num=7, enable_flag=0x01)
    piper.DisconnectPort()
    print("Disconnected safely")
