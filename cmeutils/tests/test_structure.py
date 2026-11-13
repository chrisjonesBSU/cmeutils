import math

import freud
import gsd.hoomd
import numpy as np
import pytest
from scipy.integrate import trapezoid

from cmeutils.geometry import get_quaternions
from cmeutils.structure import (
    angle_distribution,
    bond_distribution,
    concentration_profile,
    diffraction_pattern,
    dihedral_distribution,
    gsd_rdf,
    order_parameter,
    structure_factor,
    structure_factor_from_positions,
    structure_factor_nufft,
)
from cmeutils.tests.base_test import BaseTest, cubic_crystal


class TestStructure(BaseTest):
    def test_bad_combo(self, butane_gsd):
        with pytest.raises(ValueError):
            dihedral_distribution(
                butane_gsd,
                "c3",
                "c3",
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )
        with pytest.raises(ValueError):
            angle_distribution(
                butane_gsd,
                "c3",
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )
        with pytest.raises(ValueError):
            bond_distribution(
                butane_gsd,
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )

    def test_dihedral_distribution_deg(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", start=0, stop=1, degrees=True
        )
        for phi in dihedrals:
            assert -180 <= phi <= 180

    def test_dihedral_distribution_rad(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", degrees=False
        )
        for phi in dihedrals:
            assert -math.pi <= phi <= math.pi

    def test_dihedral_distribution_not_found(self, butane_gsd):
        with pytest.raises(ValueError):
            dihedral_distribution(butane_gsd, "c3", "c3", "c3", "c")

    def test_dihedral_distribution_histogram(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", histogram=True
        )
        for phi in dihedrals[:, 0]:
            assert -math.pi <= phi <= math.pi

    def test_dihedral_distribution_pdf(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd,
            "c3",
            "c3",
            "c3",
            "c3",
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = dihedrals[:, 0][1] - dihedrals[:, 0][0]
        assert np.allclose(np.sum(dihedrals[:, 1] * bin_width), 1, 1e-3)

    def test_dihedral_distribution_pmf(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd,
            "c3",
            "c3",
            "c3",
            "c3",
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(dihedrals[:, 1]), 1, 1e-3)

    def test_angle_distribution_deg(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "cc", "ss", "cc", degrees=True)
        for ang in angles:
            assert 80 < ang < 100

    def test_angle_distribution_range(self, p3ht_gsd):
        angles = angle_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            "cc",
            start=0,
            stop=1,
            histogram=True,
            degrees=True,
            theta_min=10,
            theta_max=180,
        )
        assert np.allclose(angles[0, 0], 10, atol=0.5)
        assert np.allclose(angles[-1, 0], 180, atol=0.5)

    def test_angle_distribution_rad(self, p3ht_gsd):
        angles = angle_distribution(
            p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, degrees=False
        )
        for ang in angles:
            assert 1.40 < ang < 1.75

    def test_angle_distribution_order(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "ss", "cc", "cd", start=0, stop=1)
        angles2 = angle_distribution(
            p3ht_gsd, "cd", "cc", "ss", start=0, stop=1
        )
        assert angles.shape[0] > 0
        assert angles.shape == angles2.shape
        assert np.array_equal(angles, angles2)

    def test_angle_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            angle_distribution(p3ht_gsd, "cc", "xx", "cc", start=0, stop=1)

    def test_bond_distribution(self, p3ht_gsd):
        bonds = bond_distribution(p3ht_gsd, "cc", "ss", start=0, stop=1)
        for bond in bonds:
            print(bond)
            assert 0.45 < bond < 0.52

    def test_bond_distribution_range(self, p3ht_gsd):
        bonds = bond_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            start=0,
            stop=1,
            l_min=0,
            l_max=1,
            histogram=True,
        )
        assert np.allclose(bonds[0, 0], 0, atol=0.5)
        assert np.allclose(bonds[-1, 0], 1, atol=0.5)

    def test_bond_distribution_order(self, p3ht_gsd):
        bonds = bond_distribution(p3ht_gsd, "cc", "ss", start=0, stop=1)
        bonds2 = bond_distribution(p3ht_gsd, "ss", "cc", start=0, stop=1)
        assert bonds.shape == bonds2.shape
        assert np.array_equal(bonds, bonds2)

    def test_bond_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            bond_distribution(p3ht_gsd, "xx", "ss", start=0, stop=1)

    def test_bond_histogram(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=True
        )
        bonds_no_hist = bond_distribution(
            p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=False
        )
        assert bonds_hist.ndim == 2
        assert bonds_no_hist.ndim == 1

    def test_bond_dist_pdf(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = bonds_hist[:, 0][1] - bonds_hist[:, 0][0]
        assert np.allclose(np.sum(bonds_hist[:, 1] * bin_width), 1, 1e-3)

    def test_bond_dist_pmf(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(bonds_hist[:, 1]), 1, 1e-3)

    def test_bond_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            bond_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                start=0,
                stop=1,
                histogram=True,
                l_min=0.52,
                l_max=0.60,
            )

    def test_angle_histogram(self, p3ht_gsd):
        angles_hist = angle_distribution(
            p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, histogram=True
        )
        angles_no_hist = angle_distribution(
            p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, histogram=False
        )
        assert angles_hist.ndim == 2
        assert angles_no_hist.ndim == 1

    def test_angle_dist_pdf(self, p3ht_gsd):
        angles_hist = angle_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            "cc",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = angles_hist[:, 0][1] - angles_hist[:, 0][0]
        assert np.allclose(np.sum(angles_hist[:, 1] * bin_width), 1, 1e-3)

    def test_angle_dist_pmf(self, p3ht_gsd):
        angles_hist = angle_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            "cc",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(angles_hist[:, 1]), 1, 1e-3)

    def test_angle_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            angle_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                "cc",
                start=0,
                stop=1,
                histogram=True,
                theta_min=120,
                theta_max=180,
            )

    def test_diffraction_pattern(self, gsdfile_bond):
        views = get_quaternions(n_views=5)
        dp = diffraction_pattern(gsdfile_bond, views=views)
        assert isinstance(dp, freud.diffraction.DiffractionPattern)

    def test_structure_factor_direct(self, gsdfile_bond):
        sf = structure_factor(gsdfile_bond, k_min=0.2, k_max=5)
        assert isinstance(sf, freud.diffraction.StaticStructureFactorDirect)

    def test_structure_factor_debye(self, gsdfile_bond):
        sf = structure_factor(gsdfile_bond, k_min=0.2, k_max=5, method="debye")
        assert isinstance(sf, freud.diffraction.StaticStructureFactorDebye)

    def test_structure_factor_bad_method(self, gsdfile_bond):
        with pytest.raises(ValueError):
            structure_factor(gsdfile_bond, k_min=0.2, k_max=5, method="a")

    def test_rdf_bad_args(self, AB_chain_gsd):
        with pytest.raises(ValueError):
            gsd_rdf(
                gsdfile=AB_chain_gsd,
                start=0,
                stop=10,
                exclude_bond_depth=2,
                exclude_all_bonded=True,
            )

        with pytest.raises(ValueError):
            gsd_rdf(
                gsdfile=AB_chain_gsd,
                A_name="A",
                start=0,
                stop=10,
            )

    def test_gsd_rdf(self, AB_chain_gsd):
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )
        assert isinstance(rdf, freud.density.RDF)
        _ = rdf.rdf
        assert scale_factor == 1

    def test_gsd_rdf_exclude_all_bonded(self, AB_chain_gsd):
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_all_bonded=True,
        )
        assert scale_factor == 1
        assert np.array_equal(rdf.rdf, np.zeros_like(rdf.rdf))

    def test_gsd_rdf_exclusions(self, AB_chain_gsd):
        """Exclude bonded neighbor."""
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=1,
            exclude_all_bonded=False,
        )
        assert scale_factor != 1
        # In this GSD file, the bond length is ~1, so the first peak shows up around r=1
        zero_indices = np.where(rdf.bin_centers < 1.5)[0]
        zero_array = np.zeros_like(zero_indices)
        rdf_values = rdf.rdf[zero_indices]
        assert np.allclose(rdf_values, zero_array)

        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=2,
            exclude_all_bonded=False,
        )
        assert scale_factor != 1
        # The second peak shows up around r=2, should be gone
        zero_indices = np.where(rdf.bin_centers < 2.5)[0]
        zero_array = np.zeros_like(zero_indices)
        rdf_values = rdf.rdf[zero_indices]
        assert np.allclose(rdf_values, zero_array)

    def test_gsd_rdf_normalization(self, AB_chain_gsd):
        # Chain has 10 particles total
        # All pairs are being used, N total = (10 * 9) / 2 = 45
        # Exclude bond depth of 1 should exclude all bonded pairs = 9
        # 45 / (45 - 9) = 1.25
        _, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=1,
            exclude_all_bonded=False,
        )
        assert np.isclose(scale_factor, 1.25)

        # Excl bond depth of 2 should result in 9 (bonds) + 8 (angles) exclusions
        # 45 / (45 - 17) = 1.607

        _, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=2,
            exclude_all_bonded=False,
        )
        assert np.isclose(scale_factor, 1.6071428571428572)

    def test_gsd_rdf_r_max(self, LJ_gsd):
        """Test 2 RDFs with different r_cuts. The values of the shared r_cut region should be very close"""
        rdf, _ = gsd_rdf(
            gsdfile=LJ_gsd,
            start=0,
            stop=10,
            r_max=3,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )
        rdf2, _ = gsd_rdf(
            gsdfile=LJ_gsd,
            start=0,
            stop=10,
            r_max=4,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )

        check_indices = np.where(rdf.bin_centers < 3)[0]
        check_indices2 = np.where(rdf2.bin_centers < 3)[0]
        rdf_integral = trapezoid(
            rdf.rdf[check_indices], rdf.bin_centers[check_indices]
        )
        rdf2_integral = trapezoid(
            rdf2.rdf[check_indices2], rdf2.bin_centers[check_indices2]
        )
        assert np.isclose(rdf_integral, rdf2_integral, rtol=0.05)

    def test_order_parameter(self, p3ht_gsd, p3ht_cg_gsd, mapping):
        r_max = 2
        a_max = 30

        order, cl_idx = order_parameter(
            p3ht_gsd, p3ht_cg_gsd, mapping, r_max, a_max
        )

        assert np.isclose(order[0], 0.33125)
        assert len(cl_idx[0]) == 160

    def test_conc_profiel(self, slab_snapshot):
        A_indices = np.arange(20)
        B_indices = np.arange(20, 40)
        d_profile, A_count, B_count, total_count = concentration_profile(
            slab_snapshot, A_indices, B_indices, n_bins=5, box_axis=0
        )
        assert (
            len(d_profile) == len(A_count) == len(B_count) == len(total_count)
        )
        assert (A_count / total_count)[0] == 1
        assert (A_count / total_count)[-1] == 0
        assert (B_count / total_count)[-1] == 1
        assert (B_count / total_count)[0] == 0

    def test_strides(self, gsdfile_bond):
        bonds = bond_distribution(
            gsdfile_bond, "A", "B", histogram=False, stride=2
        )
        assert len(bonds) == 10
        gsd_rdf(gsdfile_bond, "A", "B", stride=5)
        structure_factor(gsdfile_bond, k_min=0.2, k_max=5, stride=2)


def _box_matrix(box):
    Lx, Ly, Lz, xy, xz, yz = box
    return np.array([[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz], [0, 0, Lz]])


def _direct_sk(positions, box, k_min, k_max, bins):
    """Exact full-shell bin means of |sum_j exp(i k.r_j)|^2 / N."""
    # Every nonzero box-commensurate k vector with |k| < k_max.
    H = _box_matrix(box)
    n_max = np.ceil(k_max * np.linalg.norm(H, axis=0) / (2 * np.pi))
    axes = [np.arange(-m, m + 1) for m in n_max.astype(int)]
    n = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    k = n @ (2 * np.pi * np.linalg.inv(H))
    k_mag = np.linalg.norm(k, axis=1)
    keep = (k_mag > 0) & (k_mag < k_max)
    k, k_mag = k[keep], k_mag[keep]

    edges = np.linspace(k_min, k_max, bins + 1)
    # Bins are only comparable if no grid |k| sits on an edge.
    assert np.min(np.abs(k_mag[:, None] - edges[None, :])) > 1e-6
    S_k = np.abs(np.exp(1j * k @ positions.T).sum(axis=1)) ** 2
    S_k /= len(positions)
    idx = np.searchsorted(edges, k_mag, side="right") - 1
    ok = (idx >= 0) & (idx < bins)
    n_k = np.bincount(idx[ok], minlength=bins)
    S = np.bincount(idx[ok], weights=S_k[ok], minlength=bins)
    return S / np.where(n_k > 0, n_k, np.nan), n_k


class TestStructureFactorNUFFT(BaseTest):
    @pytest.mark.parametrize(
        "box",
        [[3.1, 3.7, 4.3, 0, 0, 0], [3.1, 3.7, 4.3, 0.3, -0.2, 0.15]],
    )
    @pytest.mark.parametrize("k_range", [(1.03, 9.03, 40), (0.0, 3.0, 1)])
    def test_direct_sum(self, box, k_range):
        # Exact match to a brute-force sum over the full k shell
        box = np.array(box, dtype=float)
        k_min, k_max, bins = k_range
        rng = np.random.default_rng(0)
        H = _box_matrix(box)
        positions = (H @ rng.uniform(-0.5, 0.5, (3, 50))).T
        unwrapped = positions + (H @ rng.integers(-3, 4, (3, 50))).T
        k, S, n_k = structure_factor_from_positions(
            unwrapped, box, k_min=k_min, k_max=k_max, bins=bins
        )
        S_ref, n_ref = _direct_sk(positions, box, k_min, k_max, bins)
        edges = np.linspace(k_min, k_max, bins + 1)
        assert np.allclose(k, 0.5 * (edges[1:] + edges[:-1]))
        assert np.array_equal(n_k, n_ref)
        assert np.allclose(S, S_ref, rtol=1e-6, equal_nan=True)

    def test_fcc_100_extinct(self):
        positions, box = cubic_crystal("fcc", a=1.0, cells=4)
        k100, k111 = 2 * np.pi, 2 * np.pi * np.sqrt(3)
        _, S, _ = structure_factor_from_positions(
            positions, box, k_min=k100 - 0.01, k_max=k100 + 0.01, bins=1
        )
        assert np.isclose(S[0], 0, atol=1e-6)
        _, S, _ = structure_factor_from_positions(
            positions, box, k_min=k111 - 0.01, k_max=k111 + 0.01, bins=1
        )
        assert np.isclose(S[0], len(positions), rtol=1e-6)

    def test_groups(self):
        rng = np.random.default_rng(1)
        box = np.array([3.1, 3.7, 4.3, 0.3, -0.2, 0.15])
        positions = (_box_matrix(box) @ rng.uniform(-0.5, 0.5, (3, 60))).T
        N = len(positions)
        _, S, _ = structure_factor_from_positions(positions, box, 1.0, 8.0, 20)
        _, S_one, _ = structure_factor_from_positions(
            positions, box, 1.0, 8.0, 20, groups=["a"] * N
        )
        assert np.allclose(S, S_one, equal_nan=True)
        # One particle per group: sum_l |F_l|^2 / N = 1 exactly.
        _, S_each, _ = structure_factor_from_positions(
            positions, box, 1.0, 8.0, 20, groups=np.arange(N)
        )
        assert np.allclose(S_each[~np.isnan(S_each)], 1)

    def test_empty_bins_nan(self):
        # Grid |k| in a 4^3 box are 2 pi sqrt(n) / 4 >= 1.57.
        _, S, n_k = structure_factor_from_positions(
            [[0, 0, 0]], [4, 4, 4], 0.0, 2.0, bins=4
        )
        assert np.array_equal(n_k, [0, 0, 0, 6])
        assert np.all(np.isnan(S[:3]))
        assert np.isclose(S[3], 1)

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            structure_factor_from_positions(np.zeros((0, 3)), [4, 4, 4], 1, 5)
        with pytest.raises(ValueError):
            structure_factor_from_positions(np.zeros((5, 2)), [4, 4, 4], 1, 5)
        with pytest.raises(ValueError):
            structure_factor_from_positions(np.zeros((5, 3)), [4, 4, 0], 1, 5)
        with pytest.raises(ValueError):
            structure_factor_from_positions(np.zeros((5, 3)), [4, 4], 1, 5)
        with pytest.raises(ValueError):
            structure_factor_from_positions(np.zeros((5, 3)), [4, 4, 4], 5, 1)
        with pytest.raises(ValueError):
            structure_factor_from_positions(
                np.zeros((5, 3)), [4, 4, 4], 1, 5, groups=[0, 1]
            )

    def test_gsd(self, crystal_gsd):
        # Frames are rigid shifts of one crystal, so the average equals
        # any single frame.
        k, S = structure_factor_nufft(
            crystal_gsd, k_min=1.03, k_max=20.03, bins=95, types="A"
        )
        with gsd.hoomd.open(crystal_gsd) as traj:
            frame = traj[0]
        A = frame.particles.position[frame.particles.typeid == 0]
        k_ref, S_ref, _ = structure_factor_from_positions(
            A, frame.configuration.box, 1.03, 20.03, 95
        )
        assert np.allclose(k, k_ref)
        assert np.allclose(S, S_ref, atol=1e-6, equal_nan=True)
        assert np.nanmax(S) > 10  # Bragg peaks; an ideal gas would give ~1
        _, S_all = structure_factor_nufft(
            crystal_gsd, k_min=1.03, k_max=20.03, bins=95, stride=2
        )
        assert not np.allclose(S_all, S, equal_nan=True)

    def test_gsd_frame_average(self, crystal_gsd):
        _, S = structure_factor_nufft(crystal_gsd, 1.03, 20.03, stop=None)
        with gsd.hoomd.open(crystal_gsd) as traj:
            frames = [
                structure_factor_from_positions(
                    f.particles.position, f.configuration.box, 1.03, 20.03
                )[1]
                for f in traj
            ]
        assert np.allclose(S, np.mean(frames, axis=0), equal_nan=True)

    def test_gsd_bad_type(self, crystal_gsd):
        with pytest.raises(ValueError):
            structure_factor_nufft(crystal_gsd, 1.0, 5.0, types="C")
