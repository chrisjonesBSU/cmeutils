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


def radius_of_gyration(gsd_file, start=0, stop=-1):
    """Calculates the radius of gyration.

    Parameters
    ----------
    gsd_file : str; required
        Path to a gsd_file
    start: int; optional; default 0
        The frame index of the trajectory to begin with
    stop: int; optional; default -1
        The frame index of the trajectory to end with 

    Returns
    -------
    rg_array : List of arrays of floats
        Array of individual chain Rg values for each frame
    rg_means : List of floats
        Average Rg values for each frame
    rg_std : List of floats
        Standard deviations of Rg values for each frame
    """
    trajectory = gsd.hoomd.open(gsd_file, mode="rb")
    rg_values = []
    rg_means = []
    rg_std = []
    for snap in trajectory[start: stop]:
        clusters, cl_props = gsd_utils.get_molecule_cluster(snap=snap)
        rg_values.append(cl_props.radii_of_gyration)
        rg_means.append(np.mean(cl_props.radii_of_gyration))
        rg_std.append(np.std(cl_props.radii_of_gyration))
    return rg_means, rg_std, rg_values


def end_to_end_distance(gsd_file, head_index, tail_index, start=0, stop=-1):
    """Calculates the chain end-to-end distances.

    Parameters
    ----------
    gsd_file : str; required
        Path to a gsd_file
    head_index : int; required
        The index of the first bead on the polymer chains
    tail_index: int; required
        The index of the last bead on the polymer chains
    start: int; optional; default 0
        The frame index of the trajectory to begin with
    stop: int; optional; default -1
        The frame index of the trajectory to end with 

    Returns
    -------
    re_array : List of arrays of floats
        Array of individual chain Re values for each frame
    re_means : List of floats
        Average Re values for each frame
    re_std : List of floats
        Standard deviations of Re values for each frame
    vectors : List of arrays
        The Re vector for each chain for every frame
    """
    re_array = [] # distances
    re_means = [] # mean re distance
    re_stds = [] # std of re distances
    vectors = [] # end-to-end vectors (list of lists)
    #TODO: PBC conditions. 
    with gsd.hoomd.open(gsd_file) as traj:
        for snap in traj[start:stop]:
            unwrap_adj = snap.particles.image * snap.system.box[:3]
            unwrap_pos = snap.particles.position + unwrap_adj 
            snap_res = [] # snap vectors
            cl, cl_prop = get_molecule_cluster(snap=snap)
            for i in cl.cluster_keys:
                #head = snap.particles.position[i[head_index]]
                head = unwrap_pos[i[head_index]]
                tail = unwrap_pos[i[tail_index]]
                #tail = snap.particles.position[i[tail_index]]
                vec = tail - head
                snap_res.append(vec)
            re_array.extend(np.linalg.norm(snap_res, axis=0))
            re_means.append(np.mean(np.linalg.norm(snap_res)))
            re_stds.append(np.std(np.linalg.norm(snap_res)))
            vectors.append(snap_res)
    return (re_array, re_means, re_stds, vectors)


def nematic_order_param(vectors, director):
    """Finds the nematic (S2) order parameter for a list of vectors

    Parameters
    ----------
    vectors : sequence of vectors; required
        The list of vectors to use in the nematic order parameter calculation
    director : numpy.ndarray, shape=(1,3)
        The nematic director of the reference state

    Returns
    -------
    freud.order.Nematic
    """
    vectors = np.asarray(vectors)
    orientations = rowan.normalize(np.append(np.zeros((vectors.shape[0], 1)), vectors, axis=1))
    nematic = freud.order.Nematic(np.asarray(director))
    nematic.compute(orientations)
    return nematic


def persistence_length(gsd_file, select_atoms_arg, window_size, start=0, stop=1):
    """Performs time-average sampling of persistence length using MDAnalysis
    
    See:
    https://docs.mdanalysis.org/stable/documentation_pages/analysis/polymer.html

    Parameters
    ----------
    gsd_file : str; required
        Path to a gsd_file
    slect_atoms_arg : str; required
        Valid argument to MDAnalysis.universe.select_atoms
    window_size : int; required
        The number of frames to use in 
    start: int; optional; default 0
        The frame index of the trajectory to begin with
    stop: int; optional; default -1
        The frame index of the trajectory to end with 
    """
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
    return np.mean(lp_results), np.std(lp_results)
