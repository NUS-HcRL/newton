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


import warp as wp

from ...geometry import ParticleFlags

@wp.kernel
def precompute_tet_rest_kernel(
    verts: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.vec4i),
    # outputs
    dm_inv_out: wp.array(dtype=wp.mat33),
    vol_out: wp.array(dtype=float)
):
    """
    Computes the inverse rest shape matrix (Dm^-1) and volume for tetrahedrons.
    Dm = [p1-p0, p2-p0, p3-p0]
    """
    tid = wp.tid()
    idx = indices[tid]
    
    p0, p1, p2, p3 = verts[idx[0]], verts[idx[1]], verts[idx[2]], verts[idx[3]]

    # Dm columns are edges from p0
    dm = wp.mat33(p1-p0, p2-p0, p3-p0)
    
    vol = wp.determinant(dm) / 6.0
    vol_out[tid] = wp.abs(vol)
    
    # Store inverse for deformation gradient computation
    dm_inv_out[tid] = wp.inverse(dm)

@wp.kernel
def eval_tet_pd_kernel(
    pos: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.vec4i),
    dm_invs: wp.array(dtype=wp.mat33),
    vols: wp.array(dtype=float),
    stiffness: float,
    # outputs
    rhs: wp.array(dtype=wp.vec3),
):
    """
    Projective Dynamics Local Step for Tetrahedrons (Corotational / FEM).
    Computes the optimal rotation R and adds contributions to the Global Step RHS.
    """
    tid = wp.tid()
    idx = indices[tid]
    
    # 1. Current Deformed Shape Ds
    p0, p1, p2, p3 = pos[idx[0]], pos[idx[1]], pos[idx[2]], pos[idx[3]]
    ds = wp.mat33(p1-p0, p2-p0, p3-p0)
    
    # 2. Deformation Gradient F = Ds * Dm^-1
    dm_inv = dm_invs[tid]
    F = ds * dm_inv
    
    # 3. Polar Decomposition to find Rotation R (The Projection)
    R, S = wp.polar_decomposition(F)
    
    # 4. Compute RHS contribution force
    # In PD, the force target is often formulated as k * Vol * (R * Dm^-1) for the basis vectors.
    weight = stiffness * vols[tid]
    
    # Effective "Target" deformation gradient contribution
    target_basis = R * dm_inv
    
    # Force vectors for p1, p2, p3 relative to p0
    f1 = wp.vec3(target_basis[0,0], target_basis[1,0], target_basis[2,0]) * weight
    f2 = wp.vec3(target_basis[0,1], target_basis[1,1], target_basis[2,1]) * weight
    f3 = wp.vec3(target_basis[0,2], target_basis[1,2], target_basis[2,2]) * weight
    f0 = -(f1 + f2 + f3)

    # Accumulate into Global RHS
    wp.atomic_add(rhs, idx[0], f0)
    wp.atomic_add(rhs, idx[1], f1)
    wp.atomic_add(rhs, idx[2], f2)
    wp.atomic_add(rhs, idx[3], f3)

@wp.kernel
def eval_spring_pd_kernel(
    pos: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    stiffness: float,
    # outputs
    rhs: wp.array(dtype=wp.vec3),
):
    """Projective Dynamics Local Step for Springs."""
    tid = wp.tid()
    idx = edges[tid]
    
    p0, p1 = pos[idx[0]], pos[idx[1]]
    
    diff = p1 - p0
    curr_len = wp.length(diff)
    
    # Avoid divide by zero
    dir = diff / (curr_len + 1.0e-6)
    
    # Target relative vector (manifold projection)
    target_vec = dir * rest_lengths[tid]
    
    # Force weight
    w = stiffness
    
    # Add projection force contribution to RHS
    force = target_vec * w
    
    wp.atomic_add(rhs, idx[0], -force)
    wp.atomic_add(rhs, idx[1], force)

# --- Utility Kernels (Standard) ---

@wp.kernel
def init_step_kernel(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    f_ext: wp.array(dtype=wp.vec3),
    v_curr: wp.array(dtype=wp.vec3),
    x_curr: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    pd_diags: wp.array(dtype=float),
    particle_masses: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    # outputs
    x_inertia: wp.array(dtype=wp.vec3),
    static_A_diags: wp.array(dtype=float),
    dx: wp.array(dtype=wp.vec3),
):
    """
    Predicts the inertial position (s_n) and sets up the system diagonal.
    """
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last

    if not particle_flags[tid] & ParticleFlags.ACTIVE:
        x_inertia[tid] = x_prev[tid]
        static_A_diags[tid] = 0.0
        dx[tid] = wp.vec3(0.0)
    else:
        v_prev = v_curr[tid]
        mass = particle_masses[tid]
        
        # System Matrix Diagonal A_ii = M_i / dt^2 + L_ii
        static_A_diags[tid] = pd_diags[tid] + mass / (dt * dt)
        
        # Inertial prediction s_n = x + v*dt + dt^2*M^-1*f_ext
        x_inertia[tid] = x_last + v_prev * dt + (gravity[0] + f_ext[tid] / mass) * (dt * dt)
        dx[tid] = v_prev * dt

@wp.kernel
def init_rhs_kernel(
    dt: float,
    x_curr: wp.array(dtype=wp.vec3),
    x_inertia: wp.array(dtype=wp.vec3),
    particle_masses: wp.array(dtype=float),
    # outputs
    rhs: wp.array(dtype=wp.vec3),
):
    """
    Initializes RHS with the momentum term: M/dt^2 * (s_n - x_curr)
    Note: If solving for dx, RHS is residual. If solving for x, RHS is b.
    Here we assume formulation: A * dx = b_total - A * x_curr
    """
    tid = wp.tid()
    rhs[tid] = (x_inertia[tid] - x_curr[tid]) * particle_masses[tid] / (dt * dt)

@wp.kernel
def nonlinear_step_kernel(
    x_in: wp.array(dtype=wp.vec3),
    dx: wp.array(dtype=wp.vec3),
    # outputs
    x_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + dx[tid]
    # dx[tid] = wp.vec3(0.0) # Optional reset

@wp.kernel
def update_velocity(
    dt: float,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - prev_pos[particle]) / dt