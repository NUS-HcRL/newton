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
import time

import numpy as np
import scipy.sparse as sp
import warp as wp
from warp.types import float32, matrix

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

# --------------------------------------------------------------------------- #
#                                   Kernels                                   #
# --------------------------------------------------------------------------- #

@wp.kernel
def predict_sn_kernel(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    ext_forces: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    # Outputs
    sn: wp.array(dtype=wp.vec3),    # Corresponds to _nextpos/sn in C++
    q_curr: wp.array(dtype=wp.vec3) # Initial guess for solver
):
    """
    Implements the prediction step from LocalGlobalSolver::update.
    Computes sn = pos + dt * vel + dt^2 * M^-1 * f_ext
   
    """
    i = wp.tid()
    
    if not (particle_flags[i] & ParticleFlags.ACTIVE):
        sn[i] = pos[i]
        q_curr[i] = pos[i]
        return

    # f_ext includes gravity here: acc = g + f_ext * inv_mass
    acc = gravity[0] + ext_forces[i] * inv_mass[i]
    
    # s_n = pos + dt * vel + dt^2 * acc
    pred_pos = pos[i] + vel[i] * dt + acc * (dt * dt)
    
    sn[i] = pred_pos
    q_curr[i] = pred_pos

@wp.kernel
def init_rhs_kernel(
    mass: wp.array(dtype=float),
    sn: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=wp.vec3)
):
    """
    Initialize RHS b = M * sn.
    The linear system is (M + dt^2 * L) q = M * sn + dt^2 * local_force.
    This kernel sets the base term.
    """
    i = wp.tid()
    rhs[i] = sn[i] * mass[i]

@wp.kernel
def local_step_triangle_stretching(
    q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    rest_dm_inv: wp.array(dtype=wp.mat22), # Precomputed (Basis^T * Edges)^-1
    tri_areas: wp.array(dtype=float),
    stiffness: float,
    dt_sq: float,
    # Output
    rhs: wp.array(dtype=wp.vec3)
):
    """
    Implements PDTriangleStretchingEnergy::localProjection.
    Computes Deformation Gradient F, performs Polar Decomposition to find R,
    and accumulates projection forces into RHS.
    """
    t_idx = wp.tid()
    
    i0 = tri_indices[t_idx, 0]
    i1 = tri_indices[t_idx, 1]
    i2 = tri_indices[t_idx, 2]

    # Current positions
    p0 = q[i0]
    p1 = q[i1]
    p2 = q[i2]

    # Current edges (Ds)
    d1 = p1 - p0
    d2 = p2 - p0
    
    # Retrieve Dm^-1
    dm_inv = rest_dm_inv[t_idx]
    
    # Compute Deformation Gradient F = Ds * Dm^-1
    # Manual 3x2 * 2x2 matmul
    # F col 0 = d1 * dm00 + d2 * dm10
    # F col 1 = d1 * dm01 + d2 * dm11
    f00 = dm_inv[0,0]
    f01 = dm_inv[0,1]
    f10 = dm_inv[1,0]
    f11 = dm_inv[1,1]

    f_col0 = d1 * f00 + d2 * f10
    f_col1 = d1 * f01 + d2 * f11

    # Construct 3x3 F for SVD (Polar Decomposition).
    # We construct the 3rd column via cross product to represent the normal behavior.
    f_col2 = wp.cross(f_col0, f_col1)
    
    F_mat = wp.mat33(f_col0, f_col1, f_col2)
    
    # Polar Decomposition: F = R * S. We want R.
    # U * Sig * V^T = F  =>  R = U * V^T
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    
    wp.svd3(F_mat, U, sig, V)
    
    R = U * wp.transpose(V)
    
    # Calculate projection contribution to RHS
    # weight = stiffness * area
    # In C++: _proj[i] = wi * (_restMatrix[i] * PDProjection(F).transpose());
    # _restMatrix is Dm_inv (2x2)
    # R is 3x3. We need top-left 3x2 of R? Or R is full rotation.
    # The projection target in Local-Global typically tries to minimize ||F - R||^2.
    
    # Based on C++ accumulation:
    # rhs[0] += -proj_row0 - proj_row1
    # rhs[1] += proj_row0
    # rhs[2] += proj_row1
    # Where proj = w * Dm^-1 * R^T (dims: 2x2 * 2x3 = 2x3)
    
    wi = stiffness * tri_areas[t_idx] * dt_sq
    
    # R_toprows: The first two rows of R^T (which are first two cols of R)
    # Actually, let's look at the dimensions.
    # Dm_inv is 2x2.
    # We want to transform the target Rotation back to force space.
    # Term is: w * Dm_inv * [R_col0, R_col1]^T
    
    r_col0 = wp.vec3(R[0,0], R[1,0], R[2,0])
    r_col1 = wp.vec3(R[0,1], R[1,1], R[2,1])
    
    # proj (2x3 matrix) rows:
    # row0 = wi * (dm00 * r_col0 + dm01 * r_col1)
    # row1 = wi * (dm10 * r_col0 + dm11 * r_col1)
    
    proj_row0 = (r_col0 * f00 + r_col1 * f01) * wi
    proj_row1 = (r_col0 * f10 + r_col1 * f11) * wi
    
    # Accumulate to Global RHS (Atomic add required due to shared vertices)
    wp.atomic_add(rhs, i0, (proj_row0 + proj_row1) * -1.0)
    wp.atomic_add(rhs, i1, proj_row0)
    wp.atomic_add(rhs, i2, proj_row1)

@wp.kernel
def update_velocity_kernel(
    dt: float,
    pos_prev: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3)
):
    """
    Update velocity based on position change.
    vel = (pos_new - pos_prev) / dt
    """
    i = wp.tid()
    vel[i] = (pos_new[i] - pos_prev[i]) / dt

# --------------------------------------------------------------------------- #
#                                   Solver                                    #
# --------------------------------------------------------------------------- #

class SolverRealSim(SolverBase):
    
    def __init__(self, model: Model, stiffness: float = 1000.0):
        super().__init__(model)
        self.stiffness = stiffness
        
        # Buffers for PD
        self.sn = wp.zeros_like(self.model.particle_q)
        self.rhs = wp.zeros_like(self.model.particle_q)
        
        # Precompute Topology and System Matrix
        self._init_topology_and_system()
        
        self.cholesky = None
        self.cached_dt = -1.0
        
        # Separate RHS/X buffers for component-wise solve (X, Y, Z)
        # Warp's sparse solver handles scalar arrays best.
        self.rhs_comp = wp.zeros(self.model.particle_count, dtype=float, device=self.device)
        self.x_comp = wp.zeros(self.model.particle_count, dtype=float, device=self.device)

    def _init_topology_and_system(self):
        """
        Initializes FEM data structures (Dm_inv) and global matrices (M, L).
        Mimics logic from PDTriangleStretchingEnergy::computeDmInv and accumulateMatrix.
       
        """
        
        # Pull data to CPU
        tri_indices = self.model.tri_indices.numpy()
        positions = self.model.particle_q.numpy()
        masses = self.model.particle_mass.numpy()
        n_verts = self.model.particle_count
        n_tris = len(tri_indices)
        
        dm_inv_list = []
        areas = []
        
        # Sparse Matrix Assembly (COO lists)
        # We build the Laplacian L. The full matrix A = M + dt^2 * L.
        rows, cols, data = [], [], []
        
        # 1. Compute Dm_inv and Assemble L
        for t in range(n_tris):
            i0, i1, i2 = tri_indices[t]
            p0, p1, p2 = positions[i0], positions[i1], positions[i2]
            
            # Edges
            e1 = p1 - p0
            e2 = p2 - p0
            
            # Basis Construction (from cpp computeDmInv)
            # n1 = e1.normalized()
            # n2 = (e2 - e2.dot(n1)*n1).normalized()
            # This essentially projects the triangle to 2D
            
            # Standard FEM approach: Project to 2D plane defined by triangle
            # Dm = [e1_2d, e2_2d] (2x2)
            # Area = 0.5 * cross(e1, e2).norm()
            
            normal = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(normal)
            if area < 1e-12: area = 1e-12
            
            areas.append(area)
            
            # Form Dm_inv (2x2)
            # Simplified: Use the fact that Dm^-1 * Dm^-T determines stiffness
            # In PDTriangleStretchingEnergy.cpp, `_restMatrix` is computed via basis projection.
            # Here we approximate for standard isotropic triangle elements:
            # Using Cotan-weights or standard linear element stiffness logic.
            
            # For exact reproduction of C++ logic:
            # We need the 2x2 Dm_inv.
            # Let's compute basis U, V.
            u = e1 / (np.linalg.norm(e1) + 1e-9)
            v = np.cross(normal, u)
            v = v / (np.linalg.norm(v) + 1e-9)
            
            # Project edges to basis
            e1_loc = np.array([np.dot(e1, u), np.dot(e1, v)])
            e2_loc = np.array([np.dot(e2, u), np.dot(e2, v)])
            
            Dm = np.column_stack((e1_loc, e2_loc)) # 2x2
            try:
                Dm_inv = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                Dm_inv = np.eye(2)
            
            dm_inv_list.append(Dm_inv.flatten())
            
            # Assemble Local Stiffness K_local
            # C++: ST = [[-1, -1], [1, 0], [0, 1]]
            # G = ST * Dm_inv
            # Ki = G * G.T * (weight * area)
            
            ST = np.array([[-1, -1], [1, 0], [0, 1]]) # 3x2
            G = ST @ Dm_inv # 3x2
            Ki = (G @ G.T) * (self.stiffness * area) # 3x3
            
            # Add to global L
            local_indices = [i0, i1, i2]
            for r in range(3):
                for c in range(3):
                    rows.append(local_indices[r])
                    cols.append(local_indices[c])
                    data.append(Ki[r, c])

        # Store FEM data on GPU
        self.rest_dm_inv = wp.array(np.array(dm_inv_list).reshape((-1, 2, 2)), dtype=wp.mat22, device=self.device)
        self.tri_areas = wp.array(np.array(areas), dtype=float32, device=self.device)
        
        # Build Sparse Matrices (Scipy)
        # L matrix (Stiffness)
        self.L_coo = sp.coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        
        # M matrix (Diagonal Mass)
        self.M_coo = sp.coo_matrix((masses, (np.arange(n_verts), np.arange(n_verts))), shape=(n_verts, n_verts))

    def _update_system_matrix(self, dt: float):
            """
            Rebuilds global matrix A = M + dt^2 * L if dt changes.
            """
            if abs(dt - self.cached_dt) < 1e-6 and self.cholesky is not None:
                return

            # A = M + dt^2 * L
            # This is the scalar matrix (applied to x, y, z independently)
            A_scipy = self.M_coo + (dt * dt) * self.L_coo
            
            # Warp's bsr_from_scipy expects a CSR matrix as input
            A_csr = A_scipy.tocsr()
            
            # Upload to Warp
            # FIX: Use the standalone function wp.sparse.bsr_from_scipy
            self.A_bsr = wp.sparse.bsr_from_scipy(A_csr, device=self.device)
            
            # Pre-factorize
            # Note: Cholesky requires A to be SPD.
            # M is diagonal positive, L is positive semi-definite -> A is SPD.
            self.cholesky = wp.sparse.Cholesky(self.A_bsr)
            
            self.cached_dt = dt

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """
        Main simulation step implementing the Local-Global loop.
       
        """
        
        # 1. Update System Matrix (if dt changed)
        self._update_system_matrix(dt)
        
        # 2. Prediction Step: sn = q + v*dt + M^-1 * f * dt^2
        # Also initializes state_out.particle_q with sn as guess
        wp.launch(
            kernel=predict_sn_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
                self.sn,
                state_out.particle_q
            ],
            device=self.device
        )
        
        # 3. Projective Dynamics Loop
        for _ in range(self.iterations):
            
            # A. Reset Global RHS: b = M * sn
            wp.launch(
                kernel=init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass, self.sn],
                outputs=[self.rhs],
                device=self.device
            )
            
            # B. Local Step: Add projection forces to RHS
            # b += dt^2 * local_forces
            wp.launch(
                kernel=local_step_triangle_stretching,
                dim=self.model.tri_count,
                inputs=[
                    state_out.particle_q, # Current guess
                    self.model.tri_indices,
                    self.rest_dm_inv,
                    self.tri_areas,
                    self.stiffness,
                    dt * dt,
                    self.rhs # Output accumulator
                ],
                device=self.device
            )
            
            # C. Global Step: Solve A x = b
            # We solve component-wise for X, Y, Z because A is isotropic
            
            # Solve X
            wp.copy(self.rhs_comp, self.rhs, src_offset=0, dest_offset=0, count=self.model.particle_count, src_stride=3, dest_stride=1)
            self.cholesky.solve(b=self.rhs_comp, x=self.x_comp)
            wp.copy(state_out.particle_q, self.x_comp, src_offset=0, dest_offset=0, count=self.model.particle_count, src_stride=1, dest_stride=3)
            
            # Solve Y
            wp.copy(self.rhs_comp, self.rhs, src_offset=1, dest_offset=0, count=self.model.particle_count, src_stride=3, dest_stride=1)
            self.cholesky.solve(b=self.rhs_comp, x=self.x_comp)
            wp.copy(state_out.particle_q, self.x_comp, src_offset=0, dest_offset=1, count=self.model.particle_count, src_stride=1, dest_stride=3)

            # Solve Z
            wp.copy(self.rhs_comp, self.rhs, src_offset=2, dest_offset=0, count=self.model.particle_count, src_stride=3, dest_stride=1)
            self.cholesky.solve(b=self.rhs_comp, x=self.x_comp)
            wp.copy(state_out.particle_q, self.x_comp, src_offset=0, dest_offset=2, count=self.model.particle_count, src_stride=1, dest_stride=3)

        # 4. Update Velocity
        wp.launch(
            kernel=update_velocity_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                state_in.particle_q,
                state_out.particle_q,
                state_out.particle_qd
            ],
            device=self.device
        )
        
        # 5. Handle Collisions (Placeholder logic based on template)
        # In a full RealSim implementation, hard constraints are applied after global solve
        # via the ConstraintSolver (e.g. NonSmoothNewton).
        if self.handle_self_contact:
             # self.simulate_one_step_with_collisions...
             pass