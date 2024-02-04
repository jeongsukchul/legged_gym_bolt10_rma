# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import wrap_to_pi

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Bolt10(LeggedRobot):

    def _custom_init(self, cfg):
        self.control_tick = 0

        self.control_time = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1.*contacts, dim=1)==1
        single_contact *= torch.norm(self.commands[:,:3], dim=1) > 0.1 #no reward for zero command
        return 1.*single_contact
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        joint_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-joint_error/self.cfg.rewards.tracking_sigma) * (torch.norm(self.commands[:, :3], dim=1) <= 0.1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # print("orientation_error: ", orientation_error[0])
        return torch.exp(-orientation_error/self.cfg.rewards.orientation_sigma)
    
    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)
    
    def _reward_lower_motion(self):
        return torch.sum(torch.square((self.dof_vel[:, 0:12])), dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Assume right foot has been in contact for .3 seconds.
        # left foot has just landed on ground
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.001 # [0, 1]
        single_contact = torch.sum(1.*contact, dim=1) > 0 
        contact_filt = torch.logical_or(contact, self.last_contacts) # [1, 1]
        self.last_contacts = contact # [1, 1]
        first_contact = (self.feet_air_time > 0.) * contact_filt # [1 1]
        self.feet_air_time += self.dt
        rew_airTime = torch.sum( torch.clip(self.feet_air_time - 0.3, min=0.0, max=0.7) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        rew_airTime *= single_contact #no reward for flying or double support
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_joint_power(self):
        return torch.sum(torch.relu(self.torques * self.dof_vel), dim=1)
    
        #######NOTE#######
        # The jacobian maps the joint velocities to the body's CoM frame velocities
        # The body's CoM frame is determined by the URDF files.
        # The spatial link velocities that the matrix maps to are with respect to the center of mass (COM) of the links, 
        # and are stacked [vx; vy; vz; wx; wy; wz], where vx and wx refer to the linear and rotational velocity IN WORLD FRAME, respectively.
        # The body's origin frame is the joint's frame which connects the link to its parent.default_dof_pos
        # The body's CoM frame is described in the URDF file under the tag <inertial>
        # The <origin> tag describes the CoM frame's linear and angular offset from the body's origin frame
        
    # def _reward_feet_outwards(self):
    #     return torch.square(self.dof_pos[:, 0] - self.default_dof_pos[:, 0]) + torch.square(self.dof_pos[:, 5] - self.default_dof_pos[:, 5])
    
    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # yaw joints symmetry
        error += self.sqrdexp(
            ( (self.dof_pos[:, 0] - self.default_dof_pos[:, 0] ) - (self.dof_pos[:, 5] - self.default_dof_pos[:, 5]) )
            / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp(
            ( (self.dof_pos[:, 1] - self.default_dof_pos[:, 1] ) - (self.dof_pos[:, 6] - self.default_dof_pos[:, 6]) )
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint regularization
        error +=0.8* self.sqrdexp(
            ( (self.dof_pos[:, 2]) + (self.dof_pos[:, 7]))
            / self.cfg.normalization.obs_scales.dof_pos)
        
        
        # error += self.sqrdexp(
        #     ( (self.dof_pos[:, 2]- self.default_dof_pos[:, 2]) + (self.dof_pos[:, 7] - self.default_dof_pos[:, 7]))
        #     / self.cfg.normalization.obs_scales.dof_pos)
        # print("self.dof_pos[0, 6]: ", self.dof_pos[0, 1], "// self.dof_pos[0, 6]: ", self.dof_pos[0, 6])
        return error/4.8
    def _reward_ankle_joint_regularization(self):
        #ankle joint symmetry
        error = self.sqrdexp(
            ( (self.dof_pos[:, 4] - self.default_dof_pos[:, 4]))
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            ( (self.dof_pos[:, 0] - self.default_dof_pos[:, 0]))
            / self.cfg.normalization.obs_scales.dof_pos)
        return error/2

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.mean(torch.square( (self.last_actions - self.actions)/self.dt ), dim=1)
    
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     action_rate = torch.sum(torch.square( (self.last_actions - self.actions) ), dim=1)
    #     # print("action_rate: ", action_rate[0])
    #     return torch.exp(-action_rate/self.cfg.rewards.action_rate_sigma)
    
    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt

    def _reward_ankle_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_ankle_joint_regularization() - self.rwd_ankle_jointRegPrev)
        return delta_phi / self.dt
    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt

    # def _reward_energy_pb(self):
    #     delta_phi = ~self.reset_buf \
    #         * (self._reward_energy() - self.rwd_energyPrev)
    #     return delta_phi / self.dt
    
    def _reward_action_rate_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_action_rate() - self.rwd_actionRatePrev)
        return delta_phi / self.dt

    def _reward_stand_still_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_stand_still() - self.rwd_standStillPrev)
        return delta_phi / self.dt

    def _reward_no_fly_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_no_fly() - self.rwd_noFlyPrev)
        return delta_phi / self.dt

    def _reward_feet_air_time_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_feet_air_time() - self.rwd_feetAirTimePrev)
        return delta_phi / self.dt

    def _init_buffers(self):
        super()._init_buffers()
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2) # originally shape of (num_envs, num_bodies, 6, num_dofs+6)
        # The jacobian maps joint velocities (num_dofs + 6) to spatial velocities of CoM frame of each link in global frame
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Articulations.html#jacobian
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        self.link_com = self.rb_states.view(self.num_envs,self.num_bodies,13)[...,:3]
        self.lfoot_com = self.rb_states.view(self.num_envs,self.num_bodies,13)[:,5,:3]
        self.rfoot_com = self.rb_states.view(self.num_envs,self.num_bodies,13)[:,10,:3]

        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
        # Reconstruct rb_props as tensor        
        for env in range(self.num_envs):
            for key, N in self.body_names_dict.items():
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env], 0)[N]
                # inertia tensors are about link's CoM frame
                # see how inertia tensor is made : https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
                self.rb_mass[env, N] = rb_props.mass

        self.mass_com = torch.sum(self.rb_mass.unsqueeze(-1)*self.link_com,dim=1) /torch.sum(self.rb_mass,dim=1).unsqueeze(-1)
        # self.rb_{property} can be used for dynamics calculation
     
    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_ankle_jointRegPrev = self._reward_ankle_joint_regularization()
        # self.rwd_energyPrev = self._reward_energy()
        self.rwd_actionRatePrev = self._reward_action_rate()

        self.rwd_standStillPrev = self._reward_stand_still()
        self.rwd_noFlyPrev = self._reward_no_fly()
        self.rwd_feetAirTimePrev = self._reward_feet_air_time()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.pre_physics_step()
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.ext_force_robots:
                attack_time = self.control_tick % (self.cfg.domain_rand.ext_force_interval/self.dt) <=self.cfg.domain_rand.ext_force_duration/self.dt \
                    and self.control_tick >= (self.cfg.domain_rand.ext_force_interval/self.dt)
                # if (self.control_time[env_idx,0]/self.cfg.>self.cfg.domain_rand.ext_force_start_time )&(self.control_tick <= (self.cfg.domain_rand.ext_force_start_time+self.cfg.domain_rand.ext_force_duration)/self.dt ):
                if attack_time :
                    ext_force_vector_6d_range = self.cfg.domain_rand.ext_force_vector_6d_range
                    num_buckets = 64
                    ext_force_vector_6d = torch.zeros(6)
                    for i in range(6):
                        force_buckets = torch_rand_float(ext_force_vector_6d_range[i][0], ext_force_vector_6d_range[i][1], (num_buckets,1), device='cpu')
                        bucket_ids = torch.randint(0, num_buckets,(1,))
                        ext_force_vector_6d[i]= force_buckets[bucket_ids]
                    self.ext_forces[:, 0, 0:3] = torch.tensor(ext_force_vector_6d[:3], device=self.device, requires_grad=False)    #index: root, body, force axis(6)
                    self.ext_torques[:, 0, 0:3] = torch.tensor(ext_force_vector_6d[3:], device=self.device, requires_grad=False)
                    
                else :
                    self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed   
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.mass_com = torch.sum(self.rb_mass.unsqueeze(-1)*self.link_com,dim=1) /torch.sum(self.rb_mass,dim=1).unsqueeze(-1)
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def _create_envs(self):
        super()._create_envs()
        collision_mask = [4, 9] # List of shapes for which collision must be detected#3,4,8,9
        for env in self.envs:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
            for j in range(len(rigid_shape_props)):
                if j not in collision_mask:
                    pass
                    rigid_shape_props[j].filter=1
            self.gym.set_actor_rigid_shape_properties(env, 0, rigid_shape_props)

        for name, num in self.body_names_dict.items():
            print("body: ",name,"id ",num)
            # shape_id = self.body_to_shapes[num]
            # print("body : ", name, ", shape index start : ", shape_id.start, ", shape index count : ", shape_id.count)
            # for i in range(shape_id.start, shape_id.start + shape_id.count):
            #     print("shape ", i, " filter : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].filter)
            #     print("shape ", i, " contact offset : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].contact_offset)
                # I guess the best I can try is set the shin's bitmasks as 0    

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        self.control_tick = self.control_tick + 1
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        feet_contact_forces = self.contact_forces[:,self.feet_indices,2]
        feet_contact = self.contact_forces[:,self.feet_indices,2]>1
        if self.num_privileged_obs is not None : 
            self.privileged_obs_buf = torch.cat((
                                        self.mass_com- self.base_pos,#3
                                        self.friction,               #1
                                        feet_contact_forces,         #2
                                        feet_contact,                #2
                                        self.ext_forces.view((self.num_envs,-1)), #link 11* 3 = 33
                                        self.ext_torques.view((self.num_envs,-1)) #same 33
                                        ),dim=-1)
            
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,heights),dim=-1) #187
            else:
                self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction * np.random.uniform(0.8, 1.2)
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction * np.random.uniform(0.8, 1.2)
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        return plane_params.static_friction, plane_params.dynamic_friction
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction* np.random.uniform(0.8, 1.2)
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction* np.random.uniform(0.8, 1.2)
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        return hf_params.static_friction, hf_params.dynamic_friction


    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction* np.random.uniform(0.8, 1.2)
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction* np.random.uniform(0.8, 1.2)
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

        return tm_params.static_friction, tm_params.dynamic_friction


    #helper function
    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)