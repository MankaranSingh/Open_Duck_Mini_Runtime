import numpy as np

FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_GEOMS = [
    "left_foot_bottom_tpu",
]

RIGHT_FEET_GEOMS = [
    "right_foot_bottom_tpu",
]


FEET_COLLISION_INFER = [
    "foot_assembly",
    "foot_assembly_2",
]

HIP_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
]

KNEE_JOINT_NAMES = [
    "left_knee",
    "right_knee",
]

NON_LEG_JOINTS = [
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
]

# There should be a way to get that from the mjModel...
JOINTS_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
JOINT_ORDER_ISAAC = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

ISAAC_TO_MUJOCO = [JOINT_ORDER_ISAAC.index(joint) for joint in JOINTS_ORDER]
MUJOCO_TO_ISAAC = [JOINTS_ORDER.index(joint) for joint in JOINT_ORDER_ISAAC]

DEFAULT_ACTUATOR_POS = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0.5, -0.5, 
                                 0, 0, -0.003, -0.065, 0.635, 1.379, -0.796])

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]
ROOT_BODY = "trunk_assembly"
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
