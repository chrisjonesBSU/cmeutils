import freud
import gsd
import gsd.hoomd
import numpy as np

from cmeutils.gsd_utils import frame_to_freud_system, snapshot_to_graph


def rdf(
    gsdfile,
    A_name=None,
    B_name=None,
    start=0,
    stop=-1,
    stride=1,
    r_max=None,
    r_min=0,
    bins=100,
    exclude_bond_depth=None,
    exclude_all_bonded=False,
    update_bond_graph=False,
):
    if any([A_name, B_name]) and not all([A_name, B_name]):
        raise ValueError(
            "If A_name or B_name is given, the other must be defined as well."
        )

    with gsd.hoomd.open(gsdfile, mode="r") as trajectory:
        snap = trajectory[0]
        # Use first frame, keep this one is update_bond_graph is False
        bond_graph = snapshot_to_graph(snap)

        # Use a value just less than half the minimum box length.
        if r_max is None:
            r_max = np.nextafter(
                np.min(snap.configuration.box[:3]) * 0.49, 0, dtype=np.float32
            )

        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)

        # Filter particles by type A and type B
        if A_name is not None and B_name is not None:
            type_A = snap.particles.typeid == snap.particles.types.index(A_name)
            type_B = snap.particles.typeid == snap.particles.types.index(B_name)
            type_A_indices = np.where(type_A)[0]
            type_B_indices = np.where(type_B)[0]
            exclude_ii = (A_name == B_name)
        else: # Use all particles for this RDF
            type_A = type_B = np.ones(N, dtype=bool) # All True  
            type_A_indices = type_B_indices = np.arange(snap.particles.N) 
            exclude_ii = True

        # Build up pair exclusions if exclude_bond_depth or exclude_all_bonded
        # Use this for each frame RDF unless update_bond_graph = True
        if exclude_bond_depth:
            excluded_pairs = get_excluded_pairs(bond_graph, exclude_bond_depth)

        for snap in trajectory[start:stop:stride]:
            A_xyz = snap.particles.position[type_A_indices]
            B_xyz = snap.particles.position[type_B_indices]

            box = snap.configuration.box
            system = (box, A_xyz)
            aq = freud.locality.AABBQuery.from_system(system)
            nlist = aq.query(
                B_xyz, {"r_max": r_max, "exclude_ii": True}
            ).toNeighborList()





def get_excluded_pairs(bond_graph, exclude_bond_depth):
    """Returns a set of (i, j) pairs to exclude based on step distance of a bond graph."""
    excluded_pairs = set()
    for i in bond_graph.nodes:
        lengths = nx.single_source_shortest_path_length(bond_graph, i, cutoff=exclude_bond_depth)
        for j, dist in lengths.items():
            if j > i and dist <= exclude_bond_depth:  # j > i avoids storing both (i,j) and (j,i)
                excluded_pairs.add((i, j))
    return excluded_pairs


def filter_nlist(nlist, excluded_pairs):
    i_idx = nlist.query_point_indices
    j_idx = nlist.point_indices

    # Normalize so smaller index is always first, to match excluded_pairs convention
    lo = np.minimum(i_idx, j_idx)
    hi = np.maximum(i_idx, j_idx)

    keep = np.array([
        (lo[k], hi[k]) not in excluded_pairs
        for k in range(len(i_idx))
    ])

    return freud.locality.NeighborList.from_arrays(
        num_query_points=nlist.num_query_points,
        num_points=nlist.num_points,
        query_point_indices=i_idx[keep],
        point_indices=j_idx[keep],
        distances=nlist.distances[keep],
    )
