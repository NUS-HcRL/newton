# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A module for building RealSim models."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import Axis, AxisType, Devicelike
from ..builder import ModelBuilder
from .model_realsim import RealSimModel

class RealSimModelBuilder(ModelBuilder):
    """Builder for RealSim Projective Dynamics models.
    
    Automatically computes topological constraints (Hinges, Cotangent Weights)
    for Isometric Bending when cloth grids are added.
    """

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        super().__init__(up_axis=up_axis, gravity=gravity)

        # Bending Constraint Data (Lists, populated during build)
        self.bending_indices = []
        self.bending_weights = []
        self.bending_rest_norms = []

    def add_realsim_cloth_grid(
        self,
        pos, rot, vel,
        dim_x, dim_y, cell_x, cell_y,
        mass,
        fix_left=False, fix_right=False, fix_top=False, fix_bottom=False,
        particle_radius=None,
    ):
        """Creates a cloth grid and immediately computes RealSim bending constraints."""
        
        # 1. Add standard particles and triangles using base ModelBuilder
        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)
        
        self.add_cloth_grid(
            pos=pos, rot=rot, vel=vel,
            dim_x=dim_x, dim_y=dim_y,
            cell_x=cell_x, cell_y=cell_y,
            mass=mass,
            fix_left=fix_left, fix_right=fix_right, fix_top=fix_top, fix_bottom=fix_bottom,
            particle_radius=particle_radius,
            # We don't use Newton's built-in spring/bending logic, so we set them to zero/None
            tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
            edge_ke=0.0, edge_kd=0.0
        )
        
        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        # 2. Compute RealSim Bending Constraints (Hinges)
        # We extract the newly added topology to compute hinges
        self._compute_bending_topology(start_vertex, end_vertex, start_tri, end_tri)

    def _compute_bending_topology(self, start_vert, end_vert, start_tri, end_tri):
        """Internal: Finds hinges and computes cotangent weights for the range."""
        
        # Convert to numpy for easier topology processing
        tri_indices_np = np.array(self.tri_indices[start_tri:end_tri]).reshape(-1, 3)
        # Map indices back to 0-based relative to the mesh for adjacency search
        local_tris = tri_indices_np - start_vert
        
        positions_np = np.array(self.particle_q[start_vert:end_vert])
        
        # Build Edge -> Triangle Map
        edges_dict = {}
        for t_idx, tri in enumerate(local_tris):
            for i in range(3):
                # store edge as sorted tuple
                v0, v1 = sorted((int(tri[i]), int(tri[(i + 1) % 3])))
                key = (v0, v1)
                if key not in edges_dict:
                    edges_dict[key] = []
                # (triangle_index, third_vertex_index)
                edges_dict[key].append((t_idx, int(tri[(i + 2) % 3])))

        # Find Internal Edges (Hinges)
        for edge, adj in edges_dict.items():
            if len(adj) == 2:
                # Found a hinge!
                v0, v1 = edge               # Shared edge
                v2 = adj[0][1]              # Left opposite
                v3 = adj[1][1]              # Right opposite
                
                # Get world positions
                p0 = positions_np[v0]
                p1 = positions_np[v1]
                p2 = positions_np[v2]
                p3 = positions_np[v3]
                
                # --- Cotangent Weight Computation ---
                l01 = np.linalg.norm(p1 - p0)
                l02 = np.linalg.norm(p2 - p0)
                l03 = np.linalg.norm(p3 - p0)
                l12 = np.linalg.norm(p2 - p1)
                l13 = np.linalg.norm(p3 - p1)
                
                # Heron's Areas
                s0 = 0.5 * (l01 + l02 + l12)
                val0 = max(1e-12, s0 * (s0 - l01) * (s0 - l02) * (s0 - l12))
                A0 = np.sqrt(val0)
                
                s1 = 0.5 * (l01 + l13 + l03)
                val1 = max(1e-12, s1 * (s1 - l01) * (s1 - l03) * (s1 - l13))
                A1 = np.sqrt(val1)
                
                # Cotangents
                cot02 = (l01**2 + l02**2 - l12**2) / (4.0 * A0)
                cot12 = (l01**2 + l12**2 - l02**2) / (4.0 * A0)
                cot03 = (l01**2 + l03**2 - l13**2) / (4.0 * A1)
                cot13 = (l01**2 + l13**2 - l03**2) / (4.0 * A1)
                
                w0 = cot02 + cot03
                w1 = cot12 + cot13
                w2 = -(cot02 + cot12)
                w3 = -(cot03 + cot13)
                
                # Rest Norm (Weighted curvature)
                weighted_sum = p0*w0 + p1*w1 + p2*w2 + p3*w3
                rest_norm = np.linalg.norm(weighted_sum)
                
                # Store Global Indices
                self.bending_indices.append([start_vert + v0, start_vert + v1, start_vert + v2, start_vert + v3])
                self.bending_weights.append([w0, w1, w2, w3])
                self.bending_rest_norms.append(rest_norm)

    def finalize(self, device: Devicelike | None = None, requires_grad: bool = False) -> RealSimModel:
        """Finalize and transfer data to GPU."""
        model = super().finalize(device=device, requires_grad=requires_grad)
        realsim_model = RealSimModel.from_model(model)

        with wp.ScopedDevice(device):
            # Transfer bending data to Warp arrays
            if len(self.bending_indices) > 0:
                realsim_model.bending_indices = wp.array(self.bending_indices, dtype=wp.vec4i, requires_grad=requires_grad)
                realsim_model.bending_weights = wp.array(self.bending_weights, dtype=wp.vec4, requires_grad=requires_grad)
                realsim_model.bending_rest_norms = wp.array(self.bending_rest_norms, dtype=float, requires_grad=requires_grad)
            else:
                # Empty arrays if no bending
                realsim_model.bending_indices = wp.zeros(0, dtype=wp.vec4i)
                realsim_model.bending_weights = wp.zeros(0, dtype=wp.vec4)
                realsim_model.bending_rest_norms = wp.zeros(0, dtype=float)

        return realsim_model