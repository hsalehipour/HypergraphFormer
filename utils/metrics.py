"""
Metric functions for evaluating hypergraph generation quality.

Provides canonical metrics:
- GED (Graph Edit Distance): structural similarity between access graphs
- Delta metric: shape regularity deviation between room lists
- Area proportion error: mean absolute difference between room-area proportions
  (each room's area / total apartment area), paired 1-to-1 within each type
- Room-type accuracy: multiset comparison of room types
"""

import math
import warnings

from .hypergraph import Hypergraph

ROOM_ORDER = ['kitchen', 'living', 'dining', 'bathroom', 'bed', 'foyer', 'storage', 'extra']


def _room_sort_key(room_tuple):
    """Sort key for room tuples whose first element is a room-type string."""
    roomname = room_tuple[0].lower()
    if 'bath' in roomname:
        roomname = 'bathroom'
    try:
        return ROOM_ORDER.index(roomname)
    except ValueError:
        return len(ROOM_ORDER)


def produce_room_list(generated_mesh):
    """Extract a sorted list of ``(room_name, perimeter, square_perimeter)``
    tuples from an RGL mesh, ordered by :data:`ROOM_ORDER`.

    This is the canonical input format for :func:`compute_room_similarity_score`.
    """
    room_data = []
    for i in range(generated_mesh.faceList.Count):
        face = generated_mesh.faceList[i]
        square_perimeter = 4 * math.sqrt(face.Area)
        room_data.append((face.merge_id, face.Perimeter, square_perimeter))

    return sorted(room_data, key=_room_sort_key)


def produce_room_area_list(generated_mesh):
    """Extract a sorted list of ``(room_name, area)`` tuples from an RGL mesh.

    The ordering follows :data:`ROOM_ORDER` and is intended for metrics that
    pair rooms by category, similar to :func:`produce_room_list`.
    """
    room_data = []
    for i in range(generated_mesh.faceList.Count):
        face = generated_mesh.faceList[i]
        room_data.append((face.merge_id, face.Area))

    return sorted(room_data, key=_room_sort_key)


def produce_room_list_from_polygons(room_polygons):
    """Like :func:`produce_room_list` but from pre-computed polygon dicts.

    Each item in *room_polygons* must have ``mergeid`` (str) and ``coords``
    (list of ``[x, y]`` points) -- the format written by
    ``data_preparation.py --data-format access``.
    """
    from shapely.geometry import Polygon

    room_data = []
    for rp in room_polygons:
        coords = rp.get("coords", [])
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if poly.is_empty or poly.area < 1e-6:
            continue
        area = poly.area
        perimeter = poly.length
        square_perimeter = 4 * math.sqrt(area)
        room_data.append((rp["mergeid"], perimeter, square_perimeter))

    return sorted(room_data, key=_room_sort_key)


def produce_room_area_list_from_polygons(room_polygons):
    """Like :func:`produce_room_area_list` but from pre-computed polygon dicts.

    See :func:`produce_room_list_from_polygons` for the expected input format.
    """
    from shapely.geometry import Polygon

    room_data = []
    for rp in room_polygons:
        coords = rp.get("coords", [])
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if poly.is_empty or poly.area < 1e-6:
            continue
        room_data.append((rp["mergeid"], poly.area))

    return sorted(room_data, key=_room_sort_key)


def _canonical_room_type_key(merge_id) -> str:
    """Map a mesh ``merge_id`` string to a canonical room-type key for area aggregation.

    Aligns loosely with :func:`produce_room_area_list` ordering (``bath`` → ``bathroom``,
    longest-prefix match against :data:`ROOM_ORDER`).
    """
    s = str(merge_id).lower().strip()
    if "bath" in s:
        return "bathroom"
    if s in ROOM_ORDER:
        return s
    best = None
    best_len = 0
    for r in ROOM_ORDER:
        if r == "extra":
            continue
        if s.startswith(r) and len(r) >= best_len:
            best, best_len = r, len(r)
    if best is not None:
        return best
    return s


def _aggregate_area_by_room_type(room_area_list):
    """Sum areas per canonical room type.

    Args:
        room_area_list: ``[(merge_id, area), ...]`` as from :func:`produce_room_area_list`.

    Returns:
        ``dict`` mapping canonical room-type string → total area.
    """
    from collections import defaultdict

    sums: defaultdict[str, float] = defaultdict(float)
    for merge_id, area in room_area_list:
        key = _canonical_room_type_key(merge_id)
        sums[key] += float(area)
    return dict(sums)


def accessgraph_dict_to_nx(ag):
    """Build NetworkX graph from access graph dict (nodes, edges) for GED."""
    import networkx as nx
    g = nx.Graph()
    for node_id, room_type in (ag or {}).get("nodes", {}).items():
        g.add_node(str(node_id), label=str(room_type))
    for edge in (ag or {}).get("edges", []) or []:
        u, v = str(edge[0]), str(edge[1])
        if u in g and v in g and u != v:
            g.add_edge(u, v)
    return g


def ged_between_graphs(G_pred, G_true, _id, timeout=20000, upper_bound=40, timeout_capture_best=False):
    """Core GED computation on pre-built NetworkX graphs.

    This is the low-level algorithm shared by
    :func:`compute_ged_from_hypergraph` (which builds graphs from
    hypergraph dicts) and the benchmark evaluation scripts (which build
    graphs from pixel masks or polygon outputs).

    Args:
        G_pred: Predicted NetworkX graph (nodes must have a ``label`` attribute).
        G_true: Ground-truth NetworkX graph (nodes must have a ``label`` attribute).
        _id: Sample identifier for log messages.
        timeout: Timeout in milliseconds (default: 20 000).
        upper_bound: Upper bound passed to ``nx.optimize_graph_edit_distance``.
        timeout_capture_best: If *True*, return the best value found
            before timeout instead of ``None``.

    Returns:
        ``(min_dist, num_gt_nodes)`` -- *min_dist* is ``None`` when the
        computation times out without improvement.
    """
    import networkx as nx

    result_holder = [upper_bound]

    def _inner():
        try:
            for dist in nx.optimize_graph_edit_distance(
                G_pred, G_true,
                node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                upper_bound=upper_bound,
            ):
                result_holder[0] = dist
            return result_holder[0], len(G_true.nodes())
        except Exception as e:
            print(f"⚠️  Error computing GED for sample {_id}: {e}")
            return None, None

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_inner)
    try:
        return future.result(timeout=timeout / 1000.0)
    except FuturesTimeoutError:
        if not timeout_capture_best:
            print(f'⚠️  GED timed out for sample {_id}, returning None')
            return None, None
        best_value = result_holder[0]
        if best_value < upper_bound:
            print(f'⚠️  GED timed out for sample {_id}, returning best found: {best_value:.1f}')
            return best_value, len(G_true.nodes())
        else:
            print(f'⚠️  GED timed out for sample {_id}, no improvement found')
            return None, None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def compute_ged_from_hypergraph(accessgraph, hypergraph, _id, timeout=20000,
                                upper_bound=40, timeout_capture_best=False,
                                correct_accessgraph=False, bounds=None):
    """Compute GED between an access-graph dict and a hypergraph dict.

    Builds NetworkX graphs from the raw dictionaries and delegates to
    :func:`ged_between_graphs`.

    When *correct_accessgraph* is ``True``, the access graph extracted from
    *hypergraph* is rewired based on actual mesh polygon adjacency (via
    :meth:`~utils.hypergraph.Hypergraph.correct_accessgraph`).  This
    requires *bounds* to generate the mesh.  RGL is lazy-loaded and cached
    by the Hypergraph class.

    Args:
        accessgraph: Access graph dict with ``nodes`` and ``edges``.
        hypergraph: Hypergraph BSP-tree dict.
        _id: Sample identifier for log messages.
        timeout: Timeout in milliseconds (default: 20 000).
        upper_bound: Upper bound for GED computation.
        timeout_capture_best: If *True*, return the best partial result
            on timeout.
        correct_accessgraph: If *True*, correct the access graph derived
            from *hypergraph* using mesh adjacency before GED computation.
        bounds: Boundary metadata dict for mesh generation (required when
            *correct_accessgraph* is ``True``).

    Returns:
        ``(min_dist, num_gt_nodes)`` -- see :func:`ged_between_graphs`.
    """
    if correct_accessgraph and not (isinstance(bounds, dict) and bounds.get("corners")):
        raise ValueError(
            f"correct_accessgraph=True requires bounds with 'corners' for sample {_id}"
        )

    try:
        hg = Hypergraph(hypergraph)

        if correct_accessgraph:
            pred_ag = hg.correct_accessgraph(bounds)
        else:
            warnings.warn(
                f"Access graph correction skipped for sample {_id} "
                f"(correct_accessgraph=False); GED uses raw leaf connectivity",
                stacklevel=2,
            )
            pred_ag = hg.access_graph

        if not pred_ag.get("nodes"):
            print(f"⚠️  No leaf nodes found in hypergraph for sample {_id}")
            return None, None

        G_pred = accessgraph_dict_to_nx(pred_ag)
        G_true = accessgraph_dict_to_nx(accessgraph)

    except Exception as e:
        print(f"⚠️  Error building graphs for sample {_id}: {e}")
        return None, None

    return ged_between_graphs(
        G_pred, G_true, _id,
        timeout=timeout, upper_bound=upper_bound,
        timeout_capture_best=timeout_capture_best,
    )


def compute_adjacency_ged(
    gt: dict,
    hypergraph: dict,
    _id,
    *,
    bounds: dict,
    timeout: int = 20000,
    upper_bound: int = 40,
    timeout_capture_best: bool = False,
    scale_factor: float = 1.05,
):
    """Compute GED between **mesh-adjacency** graphs of GT and prediction.

    Unlike :func:`compute_ged_from_hypergraph` (which compares declared
    access graphs), this metric compares which rooms physically touch in
    the *mesh* — built from ``room_polygons`` for GT (preferred when
    available, since they are the dataset-stamped true geometry) or from
    ``Hypergraph(gt['hypergraph']).adjacency_graph(bounds)`` as a fallback,
    and from ``Hypergraph(hypergraph).adjacency_graph(bounds)`` for the
    prediction.

    Args:
        gt: Ground-truth dict with ``room_polygons`` and/or ``hypergraph``
            (and ``bounds`` already pulled out by the caller).
        hypergraph: Predicted hypergraph BSP-tree dict.
        _id: Sample identifier for log messages.
        bounds: Boundary metadata dict for prediction mesh generation.
        timeout: Timeout in milliseconds (default: 20 000), same as GED.
        upper_bound: Upper bound for GED computation.
        timeout_capture_best: If *True*, return the best partial result
            on timeout.
        scale_factor: Polygon scale factor for adjacency tests (default
            ``1.05``, matching :meth:`Hypergraph.correct_accessgraph`).

    Returns:
        ``(min_dist, num_gt_nodes)`` -- see :func:`ged_between_graphs`.
    """
    from utils.hypergraph import Hypergraph, adjacency_graph_from_polygons

    if not (isinstance(bounds, dict) and bounds.get("corners")):
        raise ValueError(
            f"compute_adjacency_ged requires bounds with 'corners' for sample {_id}"
        )

    try:
        if isinstance(gt, dict) and gt.get("room_polygons"):
            gt_ag = adjacency_graph_from_polygons(
                gt["room_polygons"], scale_factor=scale_factor,
            )
        elif isinstance(gt, dict) and gt.get("hypergraph"):
            gt_ag = Hypergraph(gt["hypergraph"]).adjacency_graph(
                bounds, scale_factor=scale_factor,
            )
        else:
            print(
                f"⚠️  No GT geometry (room_polygons or hypergraph) for sample {_id}"
            )
            return None, None

        pred_ag = Hypergraph(hypergraph).adjacency_graph(
            bounds, scale_factor=scale_factor,
        )

        if not pred_ag.get("nodes"):
            print(f"⚠️  No leaf nodes found in hypergraph for sample {_id}")
            return None, None

        G_pred = accessgraph_dict_to_nx(pred_ag)
        G_true = accessgraph_dict_to_nx(gt_ag)

    except Exception as e:
        print(f"⚠️  Error building adjacency graphs for sample {_id}: {e}")
        return None, None

    return ged_between_graphs(
        G_pred, G_true, _id,
        timeout=timeout, upper_bound=upper_bound,
        timeout_capture_best=timeout_capture_best,
    )


def compute_room_similarity_score(room_list_A, room_list_B, strict=False, duplicate_mismatched=False, area_sorting=True):
    """
    Compute delta metric between two room lists, measuring shape regularity deviation.

    Each room list is a list of (room_name, perimeter, square_perimeter) tuples,
    where square_perimeter = 4 * sqrt(area). The per-room score is:
        sigma = abs(1 - (L_sa * L_B) / (L_A * L_sb))

    Args:
        room_list_A: List of (room_name, perimeter, square_perimeter) tuples.
        room_list_B: List of (room_name, perimeter, square_perimeter) tuples.
        strict: If True, require room name lists to match exactly (positional
            pairing); return 0.25 on mismatch.  If False (default), group rooms
            by category and pair within each group (soft matching).
        duplicate_mismatched: (soft mode only) When True, duplicate the shorter
            side within a category so every room gets a pair.  When False
            (default), only the first min(count_A, count_B) rooms are paired.
        area_sorting: (soft mode only) Sort rooms within each category by
            square_perimeter (descending) before pairing.  Default True.

    Returns:
        float: Mean sigma across all paired rooms. In strict mode returns 0.25
        on mismatch; in soft mode returns NaN when no pairs can be formed.
    """
    if strict:
        rooms_A = [t[0] for t in room_list_A]
        rooms_B = [t[0] for t in room_list_B]
        if rooms_A != rooms_B:
            return 0.25
        list_of_sigmas = []
        for i in range(len(room_list_A)):
            L_A = room_list_A[i][1]
            L_B = room_list_B[i][1]
            L_sa = room_list_A[i][2]
            L_sb = room_list_B[i][2]
            sigma = abs(1 - (L_sa * L_B) / (L_A * L_sb))
            list_of_sigmas.append(sigma)
        return sum(list_of_sigmas) / len(list_of_sigmas)

    rooms_by_category_A = {}
    rooms_by_category_B = {}
    for room_tuple in room_list_A:
        rooms_by_category_A.setdefault(room_tuple[0], []).append(room_tuple)
    for room_tuple in room_list_B:
        rooms_by_category_B.setdefault(room_tuple[0], []).append(room_tuple)

    all_categories = set(rooms_by_category_A.keys()) | set(rooms_by_category_B.keys())
    list_of_sigmas = []
    for category in all_categories:
        sublist_A = rooms_by_category_A.get(category, [])
        sublist_B = rooms_by_category_B.get(category, [])
        if area_sorting:
            sublist_A = sorted(sublist_A, key=lambda x: x[2], reverse=True)
            sublist_B = sorted(sublist_B, key=lambda x: x[2], reverse=True)
        count_A = len(sublist_A)
        count_B = len(sublist_B)
        if count_A == 0 or count_B == 0:
            continue
        if duplicate_mismatched:
            if count_A > count_B:
                sublist_B = (sublist_B * ((count_A // count_B) + 1))[:count_A]
            elif count_B > count_A:
                sublist_A = (sublist_A * ((count_B // count_A) + 1))[:count_B]
            n_pairs = max(count_A, count_B)
        else:
            n_pairs = min(count_A, count_B)
        for i in range(n_pairs):
            L_A = sublist_A[i][1]
            L_B = sublist_B[i][1]
            L_sa = sublist_A[i][2]
            L_sb = sublist_B[i][2]
            sigma = abs(1 - (L_sa * L_B) / (L_A * L_sb))
            list_of_sigmas.append(sigma)

    if len(list_of_sigmas) == 0:
        return float('nan')
    return sum(list_of_sigmas) / len(list_of_sigmas)


def compute_room_area_proportion_error(room_list_A, room_list_B, aggregate=False):
    """Compute mean area-proportion error between two apartment meshes.

    Each room list is a list of ``(merge_id, area)`` tuples from
    :func:`produce_room_area_list`.

    Proportions are computed as ``area_i / total_area`` for each room (or
    room-type sum when *aggregate* is True).  The error is the mean of
    ``|prop_pred_i - prop_gt_i|`` and is bounded in ``[0, 1]``.

    Args:
        room_list_A: Predicted rooms -- ``[(merge_id, area), ...]``.
        room_list_B: Ground-truth rooms -- ``[(merge_id, area), ...]``.
        aggregate: If ``True``, sum all room areas per canonical type before
            computing proportions (one term per type).  If ``False``
            (default), keep individual rooms: within each canonical type,
            rooms are sorted by area and paired 1-to-1; unmatched rooms
            on either side contribute their full proportion as error.

    Returns:
        dict with keys:

        - ``area_proportion_error`` (float): mean absolute proportion
          error (NaN if both apartments have zero total area).
        - ``num_terms`` (int): number of proportion-error terms averaged.
        - ``per_term_errors`` (list[float]): individual absolute
          proportion errors.
        - ``types_only_in_pred`` (list[str]): canonical type keys present
          only in pred.
        - ``types_only_in_gt`` (list[str]): canonical type keys present
          only in GT.
    """
    import numpy as np
    from collections import defaultdict

    total_A = sum(area for _, area in room_list_A)
    total_B = sum(area for _, area in room_list_B)

    if total_A == 0 and total_B == 0:
        warnings.warn(
            "compute_room_area_proportion_error: both apartments have zero "
            "total area; returning nan",
            stacklevel=2,
        )
        return {
            'area_proportion_error': float('nan'),
            'num_terms': 0,
            'per_term_errors': [],
            'types_only_in_pred': [],
            'types_only_in_gt': [],
        }

    if aggregate:
        sums_A = _aggregate_area_by_room_type(room_list_A)
        sums_B = _aggregate_area_by_room_type(room_list_B)

        all_types = sorted(set(sums_A.keys()) | set(sums_B.keys()))
        only_in_A = sorted(set(sums_A.keys()) - set(sums_B.keys()))
        only_in_B = sorted(set(sums_B.keys()) - set(sums_A.keys()))

        per_term_errors = []
        for category in all_types:
            prop_A = sums_A.get(category, 0.0) / total_A if total_A > 0 else 0.0
            prop_B = sums_B.get(category, 0.0) / total_B if total_B > 0 else 0.0
            per_term_errors.append(abs(prop_A - prop_B))
    else:
        rooms_A: defaultdict[str, list[float]] = defaultdict(list)
        rooms_B: defaultdict[str, list[float]] = defaultdict(list)
        for merge_id, area in room_list_A:
            rooms_A[_canonical_room_type_key(merge_id)].append(float(area))
        for merge_id, area in room_list_B:
            rooms_B[_canonical_room_type_key(merge_id)].append(float(area))

        all_types = sorted(set(rooms_A.keys()) | set(rooms_B.keys()))
        only_in_A = sorted(set(rooms_A.keys()) - set(rooms_B.keys()))
        only_in_B = sorted(set(rooms_B.keys()) - set(rooms_A.keys()))

        per_term_errors = []
        for category in all_types:
            areas_a = sorted(rooms_A.get(category, []), reverse=True)
            areas_b = sorted(rooms_B.get(category, []), reverse=True)
            n = max(len(areas_a), len(areas_b))
            for i in range(n):
                a = areas_a[i] if i < len(areas_a) else 0.0
                b = areas_b[i] if i < len(areas_b) else 0.0
                prop_a = a / total_A if total_A > 0 else 0.0
                prop_b = b / total_B if total_B > 0 else 0.0
                per_term_errors.append(abs(prop_a - prop_b))

    avg_error = float(np.mean(per_term_errors)) if per_term_errors else float('nan')

    return {
        'area_proportion_error': avg_error,
        'num_terms': len(per_term_errors),
        'per_term_errors': per_term_errors,
        'types_only_in_pred': only_in_A,
        'types_only_in_gt': only_in_B,
    }


def room_type_accuracy(expected_room_types, generated_room_types, strict=True):
    """Core room-type multiset comparison on two flat lists.

    This is the low-level function shared by
    :func:`compute_room_type_accuracy` (which extracts lists from
    hypergraph dicts) and the benchmark evaluation scripts (which
    extract lists from pixel masks or polygon outputs).

    The continuous definitions are always used internally:

    - ``accuracy`` = ``overlap / max(|E|, |G|)``
    - ``room_count_accuracy`` = ``min(|E|, |G|) / max(|E|, |G|)``
    - ``room_type_accuracy`` = ``overlap / min(|E|, |G|)``

    where ``overlap`` is the multiset intersection size.

    Args:
        expected_room_types: List of expected room-type labels.
        generated_room_types: List of generated room-type labels.
        strict: When ``True`` (default), the continuous values are
            thresholded to binary: 1.0 if the continuous value equals
            1.0, else 0.0.  When ``False``, the raw continuous values
            in ``[0, 1]`` are returned.

    Returns:
        dict with keys:

        - ``accuracy`` (float): combined accuracy (count **and** types).
        - ``room_count_accuracy`` (float): room-count accuracy only.
        - ``room_type_accuracy`` (float): room-type-set accuracy only
          (among the rooms that could potentially match, how many do).
    """
    from collections import Counter

    expected_counter = Counter(expected_room_types)
    generated_counter = Counter(generated_room_types)

    n_expected = len(expected_room_types)
    n_generated = len(generated_room_types)

    all_keys = set(expected_counter) | set(generated_counter)

    denom = max(n_expected, n_generated)
    if denom == 0:
        return {
            'accuracy': float('nan'),
            'room_count_accuracy': float('nan'),
            'room_type_accuracy': float('nan'),
        }

    overlap = sum(min(expected_counter[k], generated_counter[k])
                  for k in all_keys)
    accuracy = overlap / denom
    rc_accuracy = min(n_expected, n_generated) / denom
    min_count = min(n_expected, n_generated)
    rt_accuracy = (overlap / min_count) if min_count > 0 else float('nan')

    if strict:
        import math
        accuracy = 1.0 if accuracy == 1.0 else 0.0
        rc_accuracy = 1.0 if rc_accuracy == 1.0 else 0.0
        if not math.isnan(rt_accuracy):
            rt_accuracy = 1.0 if rt_accuracy == 1.0 else 0.0

    return {
        'accuracy': accuracy,
        'room_count_accuracy': rc_accuracy,
        'room_type_accuracy': rt_accuracy,
    }


def compute_room_type_accuracy(accessgraph, hypergraph, strict=True):
    """Compare room-type multisets between an access-graph dict and a hypergraph dict.

    Extracts room-type lists from the raw dictionaries and delegates to
    :func:`room_type_accuracy`.

    Args:
        accessgraph: Access graph dict with ``nodes`` mapping id -> room_type.
        hypergraph: Hypergraph BSP-tree dict.
        strict: Passed through to :func:`room_type_accuracy`.

    Returns:
        dict -- see :func:`room_type_accuracy`.
    """
    expected_room_types = list(accessgraph.get('nodes', {}).values())
    leaves = Hypergraph(hypergraph).leaves
    generated_room_types = [leaf['mergeid'] for leaf in leaves]
    return room_type_accuracy(expected_room_types, generated_room_types, strict=strict)
