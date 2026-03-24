# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations
from functools import partial
from typing import Callable, Dict, Type, Union
from .base import DomainHandler

# Handlers
from .lerobot_agibot import AGIBOTLeRobotHandler
from .agiworld import AGIWolrdHandler
from .robomind import RobomindHandler
from .droid import DroidHandler
from .real_world import AIRAgilexHandler, AIRAgilexHQHandler, AIRBotHandler, WidowxAirHandler
from .simulations import BridgeHandler, LiberoHandler, VLABenchHandler, RobotWin2Handler, RobocasaHumanHandler, CalvinHandler, RT1Handler
from .lerobotv21 import LeRobotV21Handler
from .x2robot import X2RobotHandler
from .lerobot_pickup import LeRobotPickupHandler
from .lerobot_xarm_lab import LeRobotXArmLabHandler

# Registry values can be a handler class or a functools.partial that binds
# extra kwargs (e.g. view_config) to the handler constructor.
HandlerFactory = Union[Type[DomainHandler], Callable[..., DomainHandler]]

_REGISTRY: Dict[str, HandlerFactory] = {

    # X2Robot
    "x2robot": X2RobotHandler,

    # Lerobot (v2.1 - sim)
    # "lift2": LeRobotV21Handler,
    "pickup-blender": LeRobotPickupHandler,
    "pickup-mujoco": LeRobotPickupHandler,
    "insert-blender": LeRobotPickupHandler,
    "insert-mujoco": LeRobotPickupHandler,
    "insert_centrifuge_5430-blender": LeRobotPickupHandler,
    "screw_loose-blender": LeRobotPickupHandler,
    "xarm-lab-data": LeRobotXArmLabHandler,
    "lerobotv3_bench_vortex_zed_four_stereo": partial(LeRobotXArmLabHandler, view_config={
        "available_views": [
            "observation.images.zed_high_right_left",
            "observation.images.zed_high_right_right",
            "observation.images.zed_high_left_left",
            "observation.images.zed_high_left_right",
            "observation.images.zed_gripper_left",
            "observation.images.zed_gripper_right",
            "observation.images.zed_low_left_left",
            "observation.images.zed_low_left_right",
        ],
        "num_sample": 2,
        "mask_one_rate": 0.25,
    }),
    "lerobotv3_bench_vortex_zed_multiview": partial(LeRobotXArmLabHandler, view_config={
        "available_views": [
            "observation.images.zed_right",
            "observation.images.zed_top",
            "observation.images.zed_back_left",
            "observation.images.zed_front",
            "observation.images.zed_front_left",
        ],
        "num_sample": 2,
        "mask_one_rate": 0.2,
    }),
    "lerobotv3_office_vortex_realsense": partial(LeRobotXArmLabHandler, view_config={
        "available_views": [
            "observation.images.realsense",
        ],
        "num_sample": 1,
        "mask_one_rate": 0.0,
    }),
    "screw_tighten-mujoco": LeRobotPickupHandler,
    
    # LeRobot (parquet)
    "AGIBOT": AGIBOTLeRobotHandler,
    "AGIBOT-challenge": AGIBOTLeRobotHandler,

    # HDF5 (exact)
    "Calvin": CalvinHandler,
    "RT1": RT1Handler,

    # AIR family
    "AIR-AGILEX": AIRAgilexHandler,
    "AIR-AGILEX-HQ": AIRAgilexHQHandler,
    "AIRBOT": AIRBotHandler,
    "widowx-air": WidowxAirHandler,

    # Sim/others
    "Bridge": BridgeHandler,
    "libero": LiberoHandler,
    "VLABench": VLABenchHandler,
    "robotwin2_abs_ee": RobotWin2Handler,
    "robotwin2_clean": RobotWin2Handler,
    "robocasa-human": RobocasaHumanHandler,

    # Robomind
    "robomind-franka": RobomindHandler,
    "robomind-ur": RobomindHandler,
    "robomind-agilex": RobomindHandler,
    "robomind-franka-dual": RobomindHandler,

    # Droid
    "Droid-Left": DroidHandler,
    "Droid-Right": DroidHandler,
    
    
    "agiworld-on-site-pack": AGIWolrdHandler ,
    "agiworld-on-site-pack-extra": AGIWolrdHandler ,
    "agiworld-on-site-conveyor": AGIWolrdHandler ,
    "agiworld-on-site-conveyor-extra": AGIWolrdHandler ,
    "agiworld-on-site-restock": AGIWolrdHandler ,
    "agiworld-on-site-pour": AGIWolrdHandler ,
    "agiworld-on-site-microwave": AGIWolrdHandler ,
    "agiworld-on-site-cloth": AGIWolrdHandler,
    "agiworld-on-site-cloth-2": AGIWolrdHandler
}

def get_handler_cls(dataset_name: str) -> HandlerFactory:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domains/registry.py."
        )
