# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""RealSim model class derived from the Newton model class."""

from __future__ import annotations

from ...core.types import Devicelike
from ..model import Model

class RealSimModel(Model):
    """RealSimModel derived from Newton model.
    
    Stores topology and precomputed data for Projective Dynamics constraints
    (specifically Isometric Bending and ADMM Tetrahedra).
    """

    def __init__(self, device: Devicelike | None = None):
        super().__init__(device=device)
        
        # --- Isometric Bending Data ---
        self.bending_indices = None
        """Indices of the 4-vertex hinges [v0, v1, v2, v3], shape [hinge_count, 4], int32."""
        
        self.bending_weights = None
        """Precomputed Cotangent weights for bending, shape [hinge_count, 4], float32."""
        
        self.bending_rest_norms = None
        """Target weighted norm (curvature) for bending, shape [hinge_count], float32."""
        
        # --- Tetrahedral Data ---
        # While standard Model has tet_indices, we might store precomputed volumes/DmInv here
        # to avoid recomputing them every time solver starts, though Solver can also handle them.
        self.tet_dm_inv = None
        self.tet_vols = None

    @classmethod
    def from_model(cls, model: Model):
        """Creates a RealSimModel instance from an existing Newton Model."""
        realsim_model = cls.__new__(cls)
        realsim_model.__dict__.update(model.__dict__)
        
        realsim_model.bending_indices = None
        realsim_model.bending_weights = None
        realsim_model.bending_rest_norms = None
        realsim_model.tet_dm_inv = None
        realsim_model.tet_vols = None
        
        return realsim_model