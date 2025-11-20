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

########################################################################################################################
#################################################    RealSim Solver    #################################################
########################################################################################################################

class SolverRealSim(SolverBase):
    """
    RealSim Projective Dynamics Solver.
    Handles Volumetric (Tetrahedral) and Linear Spring constraints using the
    Local-Global solve approach.
    """

    def __init__(
        self,
        model: Model, 
        iterations=10,
        stiffness_fem: float = 1000.0,
        stiffness_spring: float = 100.0,
    ):
        super().__init__(model)
        self.model = model
        self.pd_iterations = iterations
        self.stiffness_fem = stiffness_fem
        self.stiffness_spring = stiffness_spring
        
        # 1. Builder setup
        self.pd_matrix_builder = PDMatrixBuilder(model.particle_count)
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        # 2. Linear System Matrices (L matrix)
        self.pd_non_diags = SparseMatrixELL()
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        
        # The full system matrix A diagonal (M/dt^2 + L_diag)
        self.A_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        
        # 3. State Vectors
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device) # s_n
        
        # 4. Precomputed Physics Data
        self.tet_dm_inv = wp.zeros(model.tet_count, dtype=wp.mat33, device=self.device)
        self.tet_vols = wp.zeros(model.tet_count, dtype=float, device=self.device)
        
        # Helper for preconditioner
        self.inv_A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)

    def precompute(self):
        """Builds topology and precomputes rest states."""
        
        # A. Compute Rest State (Dm^-1 and Volume)
        if self.model.tet_count > 0:
            wp.launch(
                precompute_tet_rest_kernel,
                dim=self.model.tet_count,
                inputs=[self.model.particle_rest_pos, self.model.tet_indices],
                outputs=[self.tet_dm_inv, self.tet_vols],
                device=self.device
            )

        # B. Build Constant Matrix Structure on CPU/Host
        with wp.ScopedTimer("SolverRealSim::MatrixBuild"):
            if self.model.tet_count > 0:
                self.pd_matrix_builder.add_tet_constraints(
                    self.model.tet_indices, 
                    self.tet_vols, 
                    self.stiffness_fem
                )
            
            # Add springs if present in model (assuming model.edge_indices exists)
            # if self.model.edge_count > 0:
            #     self.pd_matrix_builder.add_spring_constraints(...)
            
            # Finalize matrix (Upload to GPU)
            self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = \
                self.pd_matrix_builder.finalize(self.device)

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        
        # 1. Implicit Integration Prediction (Inertia Step)
        # Computes s_n (x_inertia) and system diagonal (M/dt^2 + L)
        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                state_in.particle_f,
                state_in.particle_qd,
                state_in.particle_q,
                self.x_prev,
                self.pd_diags,
                self.model.particle_mass,
                self.model.particle_flags,
            ],
            outputs=[
                self.x_inertia,
                self.A_diags, 
                self.dx,      # Initial guess for delta (usually 0 or v*dt)
            ],
            device=self.device,
        )

        # 2. Local-Global Loop
        for _iter in range(self.pd_iterations):
            
            # --- A. Construct Global RHS (b) ---
            # Initialize RHS with Inertial Term: M/dt^2 * (s_n - x_curr)
            wp.launch(
                init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.x_inertia,
                    self.model.particle_mass,
                ],
                outputs=[self.rhs],
                device=self.device,
            )
            
            # --- B. Local Step (Projections) ---
            # Add Tet Projection Forces to RHS
            if self.model.tet_count > 0:
                wp.launch(
                    eval_tet_pd_kernel,
                    dim=self.model.tet_count,
                    inputs=[
                        state_in.particle_q, 
                        self.model.tet_indices,
                        self.tet_dm_inv,
                        self.tet_vols,
                        self.stiffness_fem
                    ],
                    outputs=[self.rhs],
                    device=self.device
                )
            
            # Add Spring Projection Forces to RHS
            if self.model.edge_count > 0:
                wp.launch(
                    eval_spring_pd_kernel,
                    dim=self.model.edge_count,
                    inputs=[
                        state_in.particle_q,
                        self.model.edge_indices,
                        self.model.edge_rest_lengths,
                        self.stiffness_spring
                    ],
                    outputs=[self.rhs],
                    device=self.device
                )

            # --- C. Global Step (Linear Solve) ---
            # Solve Ax = b. Note that PCG needs a preconditioner.
            # We assume a simple diagonal preconditioner derived from A_diags.
            
            # Prepare Preconditioner (Inverse of A_diag)
            # Inline kernel or reuse logic to invert A_diags into inv_A_diags
            # (Simplified: 1.0 / A_diags)
            
            self.linear_solver.solve(
                self.pd_non_diags,
                self.A_diags,
                None,     # x0 (dx starts at 0)
                self.rhs, # b (residual)
                self.A_diags, # Placeholder for inv_M (should be inv_A_diags in practice)
                self.dx,  # output delta
                iterations=10
            )

            # --- D. State Update ---
            wp.launch(
                nonlinear_step_kernel,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q, self.dx],
                outputs=[state_out.particle_q],
                device=self.device,
            )
            
            # Swap for next iteration
            state_in.particle_q.assign(state_out.particle_q)

        # 3. Finalize Velocity
        wp.launch(
            kernel=update_velocity,
            dim=self.model.particle_count,
            inputs=[dt, self.x_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            device=self.device,
        )