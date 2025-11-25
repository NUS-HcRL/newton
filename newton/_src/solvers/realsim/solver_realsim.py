# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
from warp.types import float32, matrix

from ...core.types import override
from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass

@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE:
        inertia[particle] = pos_prev[particle]
        return
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]

class SolverVBD(SolverBase):
    
    def __init__(self, model):
        super().__init__(model)

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.handle_self_contact:
            self.simulate_one_step_no_self_contact(state_in, state_out, control, contacts, dt)

            # [TODO]: to be placed when self contact done.
            # self.simulate_one_step_with_collisions_penetration_free(state_in, state_out, control, contacts, dt)
        else:
            self.simulate_one_step_no_self_contact(state_in, state_out, control, contacts, dt)
    
    
    def simulate_one_step_no_self_contact(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
    
        model = self.model
    
        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
                self.inertia,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        for _iter in range(self.iterations):
            self.particle_forces.zero_()
            self.particle_hessians.zero_()

            wp.launch(
                kernel=accumulate_contact_force_and_hessian_no_self_contact,
                dim=self.collision_evaluation_kernel_launch_size,
                inputs=[
                    dt,
                    color,
                    self.particle_q_prev,
                    state_in.particle_q,
                    self.model.particle_colors,
                    # body-particle contact
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self.model.particle_radius,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    self.model.shape_material_mu,
                    self.model.shape_body,
                    state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                    state_in.body_q if self.integrate_with_external_rigid_solver else None,
                    self.model.body_qd,
                    self.model.body_com,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                ],
                outputs=[self.particle_forces, self.particle_hessians],
                device=self.device,
            )

            wp.launch(
                kernel=solve_trimesh_no_self_contact,
                inputs=[
                    dt,
                    self.model.particle_color_groups[color],
                    self.particle_q_prev,
                    state_in.particle_q,
                    state_in.particle_qd,
                    self.model.particle_mass,
                    self.inertia,
                    self.model.particle_flags,
                    self.model.tri_indices,
                    self.model.tri_poses,
                    self.model.tri_materials,
                    self.model.tri_areas,
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    self.adjacency,
                    self.particle_forces,
                    self.particle_hessians,
                ],
                outputs=[
                    state_out.particle_q,
                ],
                dim=self.model.particle_color_groups[color].size,
                device=self.device,
            )

            wp.launch(
                kernel=copy_particle_positions_back,
                inputs=[self.model.particle_color_groups[color], state_in.particle_q],
                outputs=[state_out.particle_q],
                dim=self.model.particle_color_groups[color].size,
                device=self.device,
            )
            
        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )
    
