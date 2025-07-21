import numpy as np

FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_GEOMS = [
    "left_foot_tpu_collision",
]

RIGHT_FEET_GEOMS = [
    "right_foot_tpu_collision",
]

FEET_COLLISION_INFER = [
    "left_foot",
    "right_foot",
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
    "tail",
]

JOINTS_ORDER = [
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "tail",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
]

JOINT_ORDER_ISAAC = [
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "tail",
  ]

ISAAC_TO_MUJOCO = np.array([JOINT_ORDER_ISAAC.index(joint) for joint in JOINTS_ORDER])
MUJOCO_TO_ISAAC = np.array([JOINTS_ORDER.index(joint) for joint in JOINT_ORDER_ISAAC])

DEFAULT_ACTUATOR_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1961, 0.4055, 0.2093, 0.0, 0.0, -0.1961, -0.4055, -0.2093])

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]
ROOT_BODY = "base"
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
