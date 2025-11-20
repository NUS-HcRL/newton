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

import numpy as np
import warp as wp
from .linear_solver import NonZeroEntry

# Reuse add_connection from your previous file
@wp.func
def add_connection(v0: int, v1: int, counts: wp.array(dtype=int), neighbors: wp.array2d(dtype=int)):
    for slot in range(counts[v0]):
        if neighbors[v0, slot] == v1:
            return slot
    slot = counts[v0]
    if slot < neighbors.shape[1]:
        neighbors[v0, slot] = v1
        counts[v0] += 1
        return slot
    return -1

@wp.kernel
def add_tet_constraints_kernel(
    num_tets: int,
    tet_indices: wp.array(dtype=wp.vec4i),
    tet_vols: wp.array(dtype=float),
    stiffness: float,
    # outputs
    neighbors: wp.array2d(dtype=int),
    neighbor_counts: wp.array(dtype=int),
    nz_values: wp.array2d(dtype=float),
    diags: wp.array(dtype=float),
):
    """Accumulate Laplacian-like contributions from Tetrahedrons."""
    tid = wp.tid()
    
    idx = tet_indices[tid]
    w = stiffness * tet_vols[tid]
    
    # Connect all 4 vertices of the tet
    for i in range(4):
        for j in range(4):
            v0 = idx[i]
            v1 = idx[j]
            
            # Simplified stiffness weight for topology
            weight = w 
            
            if v0 == v1:
                diags[v0] += weight * 3.0 
            else:
                slot = add_connection(v0, v1, neighbor_counts, neighbors)
                if slot >= 0:
                    # Off-diagonals are negative in Laplacian
                    nz_values[v0, slot] -= weight 

@wp.kernel
def assemble_nz_ell_kernel(
    neighbors: wp.array2d(dtype=int),
    nz_values: wp.array2d(dtype=float),
    neighbor_counts: wp.array(dtype=int),
    # outputs
    nz_ell: wp.array2d(dtype=NonZeroEntry),
):
    tid = wp.tid()
    for k in range(neighbor_counts[tid]):
        nz_entry = NonZeroEntry()
        nz_entry.value = nz_values[tid, k]
        nz_entry.column_index = neighbors[tid, k]
        nz_ell[k, tid] = nz_entry


class PDMatrixBuilder:
    """Helper class for building RealSim PD matrix in sparse ELL format."""

    def __init__(self, num_verts: int, max_neighbor: int = 64):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = wp.zeros(num_verts, dtype=wp.int32, device="cpu")
        self.diags = wp.zeros(num_verts, dtype=wp.float32, device="cpu")
        self.values = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.float32, device="cpu")
        self.neighbors = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.int32, device="cpu")

    def add_tet_constraints(
        self,
        tet_indices: wp.array, 
        tet_vols: wp.array,    
        stiffness: float
    ):
        if tet_indices.shape[0] == 0:
            return

        # Ensure inputs are on CPU for building
        tet_indices_cpu = tet_indices.to("cpu")
        tet_vols_cpu = tet_vols.to("cpu")

        wp.launch(
            add_tet_constraints_kernel,
            dim=tet_indices.shape[0],
            inputs=[
                tet_indices.shape[0],
                tet_indices_cpu,
                tet_vols_cpu,
                stiffness
            ],
            outputs=[self.neighbors, self.counts, self.values, self.diags],
            device="cpu",
        )

    def finalize(self, device):
        diag = wp.array(self.diags, dtype=float, device=device)
        num_nz = wp.array(self.counts, dtype=int, device=device)
        nz_ell = wp.array2d(shape=(self.max_neighbors, self.num_verts), dtype=NonZeroEntry, device=device)

        nz_values = wp.array2d(self.values, dtype=float, device=device)
        neighbors = wp.array2d(self.neighbors, dtype=int, device=device)

        wp.launch(
            assemble_nz_ell_kernel,
            dim=self.num_verts,
            inputs=[neighbors, nz_values, num_nz],
            outputs=[nz_ell],
            device=device,
        )
        return diag, num_nz, nz_ell