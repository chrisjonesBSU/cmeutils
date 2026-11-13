from os import path

import gsd.hoomd
import numpy as np
import pytest
from pymbar.testsystems import correlated_timeseries_example

from cmeutils.visualize import FresnelGSD

asset_dir = path.join(path.dirname(__file__), "assets")


class BaseTest:
    @pytest.fixture
    def slab_snapshot(self):
        frame = gsd.hoomd.Frame()
        A_positions = np.random.uniform(-5, 0, size=(20, 3))
        B_positions = np.random.uniform(0, 5, size=(20, 3))
        frame.particles.position = np.concatenate([A_positions, B_positions])
        frame.configuration.box = np.array([10.0, 10.0, 10.0, 0, 0, 0])
        return frame

    @pytest.fixture
    def gsdfile(self, tmp_path):
        filename = tmp_path / "test.gsd"
        create_gsd(filename)
        return filename

    @pytest.fixture
    def gsdfile_bond(self, tmp_path):
        filename = tmp_path / "test_bond.gsd"
        create_gsd(filename, add_bonds=True)
        return filename

    @pytest.fixture
    def gsdfile_images(self, tmp_path):
        filename = tmp_path / "test_images.gsd"
        create_gsd(filename, images=True)
        return filename

    @pytest.fixture
    def snap(self, gsdfile):
        with gsd.hoomd.open(name=gsdfile, mode="r") as f:
            snap = f[-1]
        return snap

    @pytest.fixture
    def snap_bond(self, gsdfile_bond):
        with gsd.hoomd.open(name=gsdfile_bond, mode="r") as f:
            snap = f[-1]
        return snap

    @pytest.fixture
    def AB_chain_gsd(self):
        return path.join(asset_dir, "AB-traj.gsd")

    @pytest.fixture
    def LJ_gsd(self):
        return path.join(asset_dir, "lj-fluid.gsd")

    @pytest.fixture
    def butane_gsd(self):
        return path.join(asset_dir, "butanes.gsd")

    @pytest.fixture
    def p3ht_gsd(self):
        return path.join(asset_dir, "p3ht.gsd")

    @pytest.fixture
    def p3ht_cg_gsd(self):
        return path.join(asset_dir, "p3ht-cg.gsd")

    @pytest.fixture
    def pekk_cg_gsd(self):
        return path.join(asset_dir, "pekk-cg.gsd")

    @pytest.fixture
    def mapping(self):
        return np.loadtxt(path.join(asset_dir, "mapping.txt"), dtype=int)

    @pytest.fixture
    def p3ht_xml(self):
        return path.join(asset_dir, "p3ht.xml")

    @pytest.fixture(scope="session")
    def correlated_data_tau100_n10000(self):
        return correlated_timeseries_example(N=10000, tau=100, seed=432)

    @pytest.fixture
    def crystal_gsd(self, tmp_path):
        """FCC crystal of type A plus random type B, 4 frames each shifted."""
        positions, box = cubic_crystal("fcc", a=1.0, cells=3)
        rng = np.random.default_rng(2)
        filename = tmp_path / "crystal.gsd"
        with gsd.hoomd.open(filename, mode="w") as traj:
            for i in range(4):
                frame = gsd.hoomd.Frame()
                frame.configuration.step = i
                frame.configuration.box = box
                B = rng.uniform(-1.5, 1.5, size=(20, 3))
                frame.particles.N = len(positions) + len(B)
                frame.particles.types = ["A", "B"]
                frame.particles.typeid = [0] * len(positions) + [1] * len(B)
                frame.particles.position = np.concatenate(
                    [positions + 0.1 * i, B]
                )
                traj.append(frame)
        return filename

    @pytest.fixture
    def p3ht_fresnel(self):
        return FresnelGSD(gsd_file=path.join(asset_dir, "p3ht.gsd"))


def create_frame(i, add_bonds, images, seed=42):
    np.random.seed(seed)
    s = gsd.hoomd.Frame()
    s.configuration.step = i
    s.particles.N = 5
    s.particles.types = ["A", "B"]
    s.particles.typeid = [0, 0, 1, 1, 1]
    s.particles.position = np.random.random(size=(5, 3))
    s.configuration.box = [3, 3, 3, 0, 0, 0]
    if add_bonds:
        s.bonds.N = 3
        s.bonds.types = ["A-B", "B-B"]
        s.bonds.typeid = [0, 0, 1]
        s.bonds.group = [[0, 2], [1, 3], [3, 4]]
    if images:
        s.particles.image = np.full(shape=(5, 3), fill_value=i)
    else:
        s.particles.image = np.zeros(shape=(5, 3))
    s.validate()
    return s


def create_gsd(filename, add_bonds=False, images=False):
    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.extend(
            [
                create_frame(i, add_bonds=add_bonds, images=images)
                for i in range(10)
            ]
        )


def cubic_crystal(kind, a=1.0, cells=4):
    """Positions and box of a cubic crystal of cells**3 conventional cells."""
    basis = {
        "sc": [[0, 0, 0]],
        "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "fcc": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    }[kind]
    cell = np.stack(
        np.meshgrid(*[np.arange(cells)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    positions = (cell[:, None, :] + np.array(basis)[None]).reshape(-1, 3) * a
    L = cells * a
    return positions - L / 2, np.array([L, L, L, 0, 0, 0])
