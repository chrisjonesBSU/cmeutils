import gsd
import numpy as np

from cmeutils.polymers import get_bond_vectors
from cmeutils.tests.base_test import BaseTest


class TestPolymers(BaseTest):
    def test_get_bond_vectors(self, butane_gsd):
        with gsd.hoomd.open(butane_gsd) as traj:
            snap = traj[0]
        vecs = get_bond_vectors(snapshot=snap)
        for v in vecs:
            assert np.allclose(1, np.linalg.norm(v), atol=1e-3)

    def test_get_bond_vectors_filter(self, pekk_cg_gsd):
        with gsd.hoomd.open(pekk_cg_gsd) as traj:
            snap = traj[0]
        vecs = get_bond_vectors(snapshot=snap)
        ek_vecs = get_bond_vectors(snapshot=snap, bond_type_filter=["E-K"])
        kk_vecs = get_bond_vectors(snapshot=snap, bond_type_filter=["K-K"])
        assert len(ek_vecs) + len(kk_vecs) == len(vecs)

    def test_radius_of_gyration(self, butane_gsd):
        pass

    def test_end_to_end_distance(self, butane_gsd):
        pass

    def test_persistence_length(self, butane_gsd):
        pass
