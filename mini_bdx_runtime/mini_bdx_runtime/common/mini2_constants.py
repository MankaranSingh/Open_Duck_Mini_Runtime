# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from etils import epath
import numpy as np


ROOT_PATH = epath.Path(__file__).parent
ASSETS_PATH = ROOT_PATH / "robots" / "mini2"
FLAT_TERRAIN_XML = ASSETS_PATH / "scene_flat_terrain.xml"
ROUGH_TERRAIN_XML = ASSETS_PATH / "scene_rough_terrain.xml"
FLAT_TERRAIN_BACKLASH_XML = ASSETS_PATH / "scene_flat_terrain_backlash.xml"
ROUGH_TERRAIN_BACKLASH_XML = ASSETS_PATH / "scene_rough_terrain_backlash.xml"
URDF = ROOT_PATH / "urdf" / "mini2_bdx" / "mini2_bdx.urdf"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FLAT_TERRAIN_XML,
        "rough_terrain": ROUGH_TERRAIN_XML,
        "flat_terrain_backlash": FLAT_TERRAIN_BACKLASH_XML,
        "rough_terrain_backlash": ROUGH_TERRAIN_BACKLASH_XML,
    }[task_name]


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
