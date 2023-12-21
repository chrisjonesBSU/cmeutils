import gsd
import gsd.hoomd
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import polymer

from cmeutils import gsd_utils
from cmeutils.gsd_utils import get_molecule_cluster


def get_bond_vectors(snapshot, bond_type_filter=None):
    """Get all normalized bond vectors of a certain bond type.

    Parameters
    ---------
    snapshot : gsd.hoomd.Frame, required
        Frame of the GSD trajectory file to use
    bond_types : list-like, required
        List of bond types to find bond vectors
        Choose from options found in gsd.hoomd.Frame.bonds.types

    Returns
    ------
    vectors : List of arrays (shape=(1,3)
        List of all normalized bond vectors matching bond_types

    """
    if not bond_type_filter:
        bond_type_filter = snapshot.bonds.types
    vectors = []
    for bond in bond_type_filter:
        if bond not in snapshot.bonds.types:
            raise ValueError(
                f"Bond type {bond} not found in snapshot.bonds.types"
            )
        bond_id = snapshot.bonds.types.index(bond)
        bond_indices = np.where(snapshot.bonds.typeid == bond_id)[0]
        for i in bond_indices:
            bond_group = snapshot.bonds.group[i]
            p1 = snapshot.particles.position[bond_group[0]]
            p2 = snapshot.particles.position[bond_group[1]]
            vectors.append((p2 - p1) / np.linalg.norm(p2 - p1))
    return vectors


def radius_of_gyration(gsd_file, start=0, stop=-1):
    """Calculates the radius of gyration using Freud's cluster module.

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
    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    rg_values = []
    rg_means = []
    rg_std = []
    for snap in trajectory[start:stop]:
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
    re_array = []  # distances (List of arrays)
    re_means = []  # mean re distances
    re_stds = []  # std of re distances
    vectors = []  # end-to-end vectors (List of arrays)
    with gsd.hoomd.open(gsd_file, "r") as traj:
        for snap in traj[start:stop]:
            unwrap_adj = snap.particles.image * snap.configuration.box[:3]
            unwrap_pos = snap.particles.position + unwrap_adj
            cl, cl_prop = get_molecule_cluster(snap=snap)
            # Create arrays with length of N polymer chains
            snap_re_vectors = np.zeros(shape=(len(cl.cluster_keys), 3))
            snap_re_distances = np.zeros(len(cl.cluster_keys))
            # Iterate through each polymer chain
            for idx, i in enumerate(cl.cluster_keys):
                head = unwrap_pos[i[head_index]]
                tail = unwrap_pos[i[tail_index]]
                vec = tail - head
                snap_re_vectors[idx] = vec
                snap_re_distances[idx] = np.linalg.norm(vec)

            re_array.append(snap_re_vectors)
            re_means.append(np.mean(snap_re_distances))
            re_stds.append(np.std(snap_re_distances))
            vectors.append(snap_re_vectors)
    return (np.array(re_means), np.array(re_stds), re_array, vectors)


def persistence_length(
    gsd_file, select_atoms_arg, window_size, start=0, stop=1
):
    """Performs time-average sampling of persistence length using MDAnalysis.

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
            backbones = [
                chain.select_atoms(select_atoms_arg) for chain in chains
            ]
            sorted_backbones = [polymer.sort_backbone(bb) for bb in backbones]
            _pl = polymer.PersistenceLength(sorted_backbones)
            pl = _pl.run(start=frame, stop=sampling_windows[idx + 1] - 1)
            lp_results.append(pl.results.lp)
        except IndexError:
            pass
    return np.mean(lp_results), np.std(lp_results)
