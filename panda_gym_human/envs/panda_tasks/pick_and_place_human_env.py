import numpy as np

import sys
sys.path.append("..")

from panda_gym_human.envs.robots.panda import Panda
from panda_gym_human.envs.tasks.pick_and_place_human import PickAndPlaceHuman

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet


class PickAndPlaceHumanEnv(RobotTaskEnv):

    def __init__(self, render: bool = True, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceHuman(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)
