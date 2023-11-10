from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
import os
import math
import random
import pybullet as p

from gym.utils import seeding

# from assistive_gym.envs.agents.human_mesh import HumanMesh

class PickAndPlaceHuman(Task):
    
    def __init__(
            self,
            sim: PyBullet,
            get_ee_position ,
            reward_type: str = "sparse",
            distance_threshold: float = 0.02
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.Insert_Flag = False
        self.get_ee_position = get_ee_position

        self.human = np.array([1.0, 0.1, 0.0])

        self.flask = np.array([0.05, 0.0, 0.0])


        self.rack_initial =  np.array([0.05, 0.25, 0.01])
        self.rack_holder_initial =  np.array([0.05, 0.25, 0.0])


        self.rack_goal =  np.array([0.06 , -0.25 , 0.008])
        self.rack_holder_goal =  np.array([0.06 , -0.25 , 0.0])

        self.ee_z_position = np.array([0.0, 0.0, 0.075])


        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=1.5, yaw=45, pitch=-30)

    # def create_human(self):
    #     path = os.getcwd()
    #     self.np_random, seed = seeding.np_random(seed)
    #     self.human = HumanMesh()
    #     human_mesh, v, j = self.human.create_smplx_body(path, 0, self.np_random)

    #     return human_mesh

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.5, width=1.0, height=0.4, x_offset=-0.3)

        self.q = [0,1,0,0]
        r = math.pi
        self.q[0] = math.cos(r/4)
        self.q[1] = math.sin(r/4)
        self.q[2] = math.sin(r/4)
        self.q[3] = math.cos(r/4)


        #------------------------------------- Human --------------------------------------

        human_visual = os.getcwd() + '/Objects/human.obj'
        human_collision = os.getcwd() + '/Objects/human.obj'

        # human_visual1 = self.create_human()
        # human_collision1 = self.create_human()

        mesh_scale_human = [0.5] * 3
        visual_kwargs1 = {
        'fileName': human_visual,
        'meshScale':mesh_scale_human
        }
        # 'rgbaColor':[0.5, 0.5, 0.5, 1.0]}
       
        collision_kwargs1={
        'fileName': human_collision,
        'meshScale':mesh_scale_human
        }

        self.sim._create_geometry(
            'human',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=57.0,
            position=self.human,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #------------------------------------- Flask --------------------------------------
        flask_visual = os.getcwd()+'/Objects/flask.obj'
        flask_collision = os.getcwd()+'/Objects/flask.obj'

        # mesh_scale = [0.5] * 3

        mesh_scale_flask = None

        visual_kwargs1 = {
        'fileName': flask_visual,
        'meshScale':mesh_scale_flask,
        'rgbaColor':[0.8, 0.9, 0.9, 0.5]}
       
        collision_kwargs1={
        'fileName': flask_collision,
        'meshScale':mesh_scale_flask
        }


        self.sim._create_geometry(
            body_name="flask",
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1.0,
            position=self.flask,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1
        )

  
        #------------------------------------- Rack -----------------------------------------------------
        rack_visual = os.getcwd()+'/Objects/Rack.obj'
        rack_collision = os.getcwd()+'/Objects/Rack_vhacd.obj'

        mesh_scale = [0.03] * 3

        visual_kwargs1 = {
        'fileName': rack_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[0.5, 0.5, 0.5, 1.0]}
       
        collision_kwargs1={
        'fileName': rack_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'rack_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=0.5,
            position=self.rack_initial,
            ghost=False,
            #orientation=np.array(self.q),
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #------------------------------------- Rack Holder 1 (start) -----------------------------------------------
        rack_holder_visual = os.getcwd()+'/Objects/Rack_holder.obj'
        rack__holder_collision = os.getcwd()+'/Objects/Rack_holder_vhacd.obj'

        mesh_scale = [0.03] * 3

        visual_kwargs1 = {
        'fileName': rack_holder_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[0, 0, 0, 1.0]}
       
        collision_kwargs1={
        'fileName': rack__holder_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'rack_holder_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder_initial,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #------------------------------------- Rack Holder 2 (finish) -----------------------------------------------
        rack_holder_visual2 = os.getcwd()+'/Objects/Rack_holder.obj'
        rack__holder_collision2 = os.getcwd()+'/Objects/Rack_holder_vhacd.obj'

        mesh_scale = [0.03] * 3

        visual_kwargs1 = {
        'fileName': rack_holder_visual2,
        'meshScale':mesh_scale,
        'rgbaColor':[0, 0, 0, 1.0]}
       
        collision_kwargs1={
        'fileName': rack__holder_collision2,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'rack_holder_target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder_goal,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

    # ------------------------------------ Observation space for the Vial ---------------------------------
    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("rack_initial") 
        object_velocity = self.sim.get_base_velocity("rack_initial")
        object_rotation = self.sim.get_base_rotation("rack_initial")
        object_angular_velocity = self.sim.get_base_angular_velocity("rack_initial")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])

        return observation
       
    # ---------------------------------- Achieved Goal --------------------------------------------------
    def get_achieved_goal(self) -> np.ndarray:
        object_position = self.sim.get_base_position("rack_initial") 
        ee_position = np.array(self.get_ee_position())
        #achieved =  np.concatenate([object_position,object_position, ee_position])
        achieved =  np.concatenate([object_position, ee_position])
        
        return achieved

    # --------------------------------- Reset -----------------------------------------------------------
    def reset(self) -> None:

        rand1 = np.array([random.random()*0.05, random.random()*0.05, 0])
        rand2 = np.array([random.random()*0.05, random.random()*0.05, 0])

        self.Insert_Flag = False
        self.goal1 = self.rack_holder_goal + rand2
        self.goal2 = self.goal1 + self.ee_z_position

        self.goal = np.concatenate([self.goal1, self.goal2]) # desired goal

        self.sim.set_base_pose("rack_initial",  self.rack_initial + rand1, np.array(self.q))
        self.sim.set_base_pose("human", self.human, np.array(self.q))
        self.sim.set_base_pose("flask", self.flask, np.array(self.q))

        self.sim.set_base_pose("rack_holder_initial",  self.rack_holder_initial + rand1 , np.array(self.q))
        self.sim.set_base_pose("rack_holder_target",  self.rack_holder_goal + rand2, np.array(self.q))
        

    # ---------------------- Success flag evaluation ------------------------------------------
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:

        d = distance(achieved_goal, desired_goal)
        self.Insert_Flag = np.array(d < self.distance_threshold, dtype=bool)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:

        contact_point = p.getContactPoints(
            bodyA=self.sim._bodies_idx['rack_initial'],
            bodyB=self.sim._bodies_idx['rack_holder_initial'],
        )

        d = distance(achieved_goal, desired_goal)

        if len(contact_point) != 0:
            reward = -1.2
        else:
            reward = -np.array(d > self.distance_threshold , dtype=np.float64)
            if self.Insert_Flag == True:
                reward = 1.0

        return reward
