import warnings

import freud
import gsd
import gsd.hoomd
import MDAnalysis as mda
from MDAnalysis.analysis import polymer
import numpy as np
import rowan
from rowan import vector_vector_rotation

from cmeutils import gsd_utils
from cmeutils.gsd_utils import get_molecule_cluster
from cmeutils.geometry import (
        get_plane_normal, angle_between_vectors, dihedral_angle
)
from cmeutils.plotting import get_histogram


def radius_of_gyration(gsd_file, start, stop):
    trajectory = gsd.hoomd.open(gsd_file, mode="rb")
    rg_values = []
    rg_means = []
    rg_std = []
    for snap in trajectory[start: stop]:
        clusters, cl_props = gsd_utils.get_molecule_cluster(snap=snap)
        rg_values.extend(cl_props.radii_of_gyration)
        rg_means.append(np.mean(cl_props.radii_of_gyration))
        rg_std.append(np.std(cl_props.radii_of_gyration))
    return rg_means, rg_std, rg_values


def end_to_end_distance(gsd_file, head_index, tail_index, start, stop):
    re_array = [] # distances
    re_means = [] # mean re distance
    re_stds = [] # std of re distances
    vectors = [] # end-to-end vectors (list of lists)
    with gsd.hoomd.open(gsd_file) as traj:
        for snap in traj[start:stop]:
            snap_res = [] # snap vectors
            cl, cl_prop = get_molecule_cluster(snap=snap)
            for i in cl.cluster_keys:
                head = snap.particles.position[i[head_index]]
                tail = snap.particles.position[i[tail_index]]
                vec = tail - head
                snap_res.append(vec)
            re_array.extend(np.linalg.norm(snap_res, axis=0))
            re_means.append(np.mean(np.linalg.norm(snap_res)))
            re_stds.append(np.std(np.linalg.norm(snap_res)))
            vectors.append(snap_res)
    return (re_array, re_means, re_stds, vectors)


def nematic_order_param(vectors, director):
    """
    vectors: list of vectors
    """
    vectors = np.asarray(vectors)
    orientations = rowan.normalize(np.append(np.zeros((vectors.shape[0], 1)), vectors, axis=1))
    nematic = freud.order.Nematic(np.asarray(director))
    nematic.compute(orientations)
    return nematic


def persistence_length(gsd_file, start, stop, select_atoms_arg, window_size):
    """Performs time-average sampling of persistence length"""
    lp_results = []
    sampling_windows = np.arange(start, stop + 1, window_size)
    for idx, frame in enumerate(sampling_windows):
        try:
            u = mda.Universe(gsd_file)
            chains = u.atoms.fragments
            backbones = [chain.select_atoms(select_atoms_arg) for chain in chains]
            sorted_backbones = [polymer.sort_backbone(bb) for bb in backbones]
            _pl = polymer.PersistenceLength(sorted_backbones)
            pl = _pl.run(start=frame, stop=sampling_windows[idx+1] - 1)
            lp_results.append(pl.results.lp)
        except IndexError:
            pass
    lp_std = np.std(lp_results)
    return np.mean(lp_results), lp_std
