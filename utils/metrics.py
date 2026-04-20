"""
Metric functions for evaluating hypergraph generation quality.

Provides canonical metrics:
- GED (Graph Edit Distance): structural similarity between access graphs
- Delta metric: shape regularity deviation between room lists
- Area difference: relative error of **total area per canonical room type** (bathrooms summed, etc.),
  averaged over room types present in **both** apartments only
- Room-type accuracy: multiset comparison of room types
"""

import math

from .hypergraph import Hypergraph

ROOM_ORDER = ['kitchen', 'living', 'dining', 'bathroom', 'bed', 'foyer', 'storage', 'extra']


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

    def _sort_key(room_tuple):
        roomname = room_tuple[0].lower()
        if 'bath' in roomname:
            roomname = 'bathroom'
        try:
            return ROOM_ORDER.index(roomname)
        except ValueError:
            return len(ROOM_ORDER)

    return sorted(room_data, key=_sort_key)


def produce_room_area_list(generated_mesh):
    """Extract a sorted list of ``(room_name, area)`` tuples from an RGL mesh.

    The ordering follows :data:`ROOM_ORDER` and is intended for metrics that
    pair rooms by category, similar to :func:`produce_room_list`.
    """
    room_data = []
    for i in range(generated_mesh.faceList.Count):
        face = generated_mesh.faceList[i]
        room_data.append((face.merge_id, face.Area))

    def _sort_key(room_tuple):
        roomname = room_tuple[0].lower()
        if 'bath' in roomname:
            roomname = 'bathroom'
        try:
            return ROOM_ORDER.index(roomname)
        except ValueError:
            return len(ROOM_ORDER)

    return sorted(room_data, key=_sort_key)


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


def ged_between_graphs(G_pred, G_true, _id, timeout=10000, upper_bound=40, timeout_capture_best=False):
    """Core GED computation on pre-built NetworkX graphs.

    This is the low-level algorithm shared by
    :func:`compute_ged_from_hypergraph` (which builds graphs from
    hypergraph dicts) and the benchmark evaluation scripts (which build
    graphs from pixel masks or polygon outputs).

    Args:
        G_pred: Predicted NetworkX graph (nodes must have a ``label`` attribute).
        G_true: Ground-truth NetworkX graph (nodes must have a ``label`` attribute).
        _id: Sample identifier for log messages.
        timeout: Timeout in milliseconds (default: 10 000).
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


def compute_ged_from_hypergraph(accessgraph, hypergraph, _id, timeout=10000,
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
        timeout: Timeout in milliseconds (default: 10 000).
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
    try:
        hg = Hypergraph(hypergraph)

        if correct_accessgraph:
            pred_ag = hg.correct_accessgraph(bounds)
        else:
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
        on mismatch; in soft mode returns 0.6 when no pairs can be formed.
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
        return 0.6
    return sum(list_of_sigmas) / len(list_of_sigmas)


def compute_room_area_difference_score(room_list_A, room_list_B, strict=False, duplicate_mismatched=False, area_sorting=True):
    """Compute mean relative area error between two apartment meshes.

    Each room list is a list of ``(merge_id, area)`` tuples from
    :func:`produce_room_area_list`.

    **Strict mode** (``strict=True``): requires identical merge-id order; per-room
    relative error ``abs(A - B) / B``.

    **Default (non-strict)**: for each canonical room type (kitchen, bathroom, …),
    **sum** all room face areas of that type in each apartment, then take
    ``abs(A_sum - B_sum) / B_sum`` for each type that appears in **both**
    apartments. Types present in only one apartment are **skipped**. The score is
    the mean of those relative errors over matching types.

    ``duplicate_mismatched`` and ``area_sorting`` are **ignored** in non-strict mode
    (kept for API compatibility).
    """
    if strict:
        rooms_A = [t[0] for t in room_list_A]
        rooms_B = [t[0] for t in room_list_B]
        if rooms_A != rooms_B:
            return 0.25
        sigmas = []
        for i in range(len(room_list_A)):
            area_A = room_list_A[i][1]
            area_B = room_list_B[i][1]
            if area_B == 0:
                continue
            sigmas.append(abs(area_A - area_B) / area_B)
        return sum(sigmas) / len(sigmas) if sigmas else 0.6

    sums_A = _aggregate_area_by_room_type(room_list_A)
    sums_B = _aggregate_area_by_room_type(room_list_B)
    common_types = set(sums_A.keys()) & set(sums_B.keys())
    sigmas = []
    for category in common_types:
        area_A = sums_A[category]
        area_B = sums_B[category]
        if area_B == 0:
            continue
        sigmas.append(abs(area_A - area_B) / area_B)

    if len(sigmas) == 0:
        return 0.6
    return sum(sigmas) / len(sigmas)


def room_type_accuracy(expected_room_types, generated_room_types):
    """Core room-type multiset comparison on two flat lists.

    This is the low-level function shared by
    :func:`compute_room_type_accuracy` (which extracts lists from
    hypergraph dicts) and the benchmark evaluation scripts (which
    extract lists from pixel masks or polygon outputs).

    Args:
        expected_room_types: List of expected room-type labels.
        generated_room_types: List of generated room-type labels.

    Returns:
        dict with keys:

        - ``room_count_match`` (bool)
        - ``room_types_match`` (bool)
        - ``accuracy`` (float): 1.0 if both match, else 0.0
        - ``expected_rooms`` / ``generated_rooms`` (int)
        - ``expected_types`` / ``generated_types`` (dict)
        - ``missing_types`` / ``extra_types`` (list)
        - ``count_error`` (int): absolute count difference
        - ``type_error`` (int): number of mismatched room slots
        - ``relative_count_error`` (float)
        - ``excess_rooms`` (int): positive = too many, negative = too few
    """
    from collections import Counter

    expected_counter = Counter(expected_room_types)
    generated_counter = Counter(generated_room_types)

    n_expected = len(expected_room_types)
    n_generated = len(generated_room_types)

    room_count_match = n_expected == n_generated
    room_types_match = expected_counter == generated_counter

    missing_types = []
    extra_types = []
    for room_type, count in expected_counter.items():
        diff = count - generated_counter.get(room_type, 0)
        if diff > 0:
            missing_types.extend([room_type] * diff)
    for room_type, count in generated_counter.items():
        diff = count - expected_counter.get(room_type, 0)
        if diff > 0:
            extra_types.extend([room_type] * diff)

    count_error = abs(n_generated - n_expected)
    type_error = (len(missing_types) + len(extra_types)) // 2

    return {
        'room_count_match': room_count_match,
        'room_types_match': room_types_match,
        'accuracy': 1.0 if (room_count_match and room_types_match) else 0.0,
        'expected_rooms': n_expected,
        'generated_rooms': n_generated,
        'expected_types': dict(expected_counter),
        'generated_types': dict(generated_counter),
        'missing_types': missing_types,
        'extra_types': extra_types,
        'count_error': count_error,
        'type_error': type_error,
        'relative_count_error': count_error / n_expected if n_expected > 0 else 0.0,
        'excess_rooms': n_generated - n_expected,
    }


def compute_room_type_accuracy(accessgraph, hypergraph):
    """Compare room-type multisets between an access-graph dict and a hypergraph dict.

    Extracts room-type lists from the raw dictionaries and delegates to
    :func:`room_type_accuracy`.

    Args:
        accessgraph: Access graph dict with ``nodes`` mapping id -> room_type.
        hypergraph: Hypergraph BSP-tree dict.

    Returns:
        dict -- see :func:`room_type_accuracy`.
    """
    expected_room_types = list(accessgraph.get('nodes', {}).values())
    leaves = Hypergraph(hypergraph).leaves
    generated_room_types = [leaf['mergeid'] for leaf in leaves]
    return room_type_accuracy(expected_room_types, generated_room_types)


def bucketed_room_accuracy(accessgraphs, hypergraphs, buckets,
                           skip_ids=None):
    """Compute room-type accuracy grouped into room-count buckets.

    Args:
        accessgraphs: ``{sample_id: accessgraph_dict}``
        hypergraphs:  ``{sample_id: hypergraph_dict}``
        buckets: Ordered list of bucket boundaries.  Each entry is an
            ``int`` or a string like ``'<=4'`` / ``'>=9'``.  Integers
            create an exact-match bucket; ``'<=N'`` catches everything
            at or below *N*; ``'>=N'`` catches everything at or above *N*.
        skip_ids: Optional set of sample IDs to exclude.

    Returns:
        dict with keys:

        - ``categories``: ``{label: {matches, total}}``
        - ``category_order``: list of bucket labels in input order
        - ``overall_matches``: int
        - ``overall_total``: int
        - ``skipped``: int
    """
    from tqdm import tqdm

    skip_ids = skip_ids or set()

    lower_bound = None
    upper_bound = None
    exact_values = []

    for b in buckets:
        s = str(b)
        if s.startswith("<=") or s.startswith("≤"):
            lower_bound = int(s.lstrip("<=≤"))
        elif s.startswith(">=") or s.startswith("≥"):
            upper_bound = int(s.lstrip(">=≥"))
        else:
            exact_values.append(int(s))

    def _bucket_label(n):
        if lower_bound is not None and n <= lower_bound:
            return f"<={lower_bound}"
        if upper_bound is not None and n >= upper_bound:
            return f">={upper_bound}"
        if n in exact_values:
            return str(n)
        return None

    category_labels = []
    for b in buckets:
        s = str(b)
        if s.startswith("<=") or s.startswith("≤"):
            category_labels.append(f"<={s.lstrip('<=≤')}")
        elif s.startswith(">=") or s.startswith("≥"):
            category_labels.append(f">={s.lstrip('>=≥')}")
        else:
            category_labels.append(s)

    categories = {label: {"matches": 0, "total": 0} for label in category_labels}
    overall_matches = 0
    overall_total = 0
    skipped = 0

    for sample_id in tqdm(hypergraphs, desc="Comparing rooms"):
        if sample_id in skip_ids:
            skipped += 1
            continue
        if sample_id not in accessgraphs:
            continue

        result = compute_room_type_accuracy(
            accessgraphs[sample_id], hypergraphs[sample_id]
        )
        label = _bucket_label(result["expected_rooms"])
        if label is None:
            continue

        categories[label]["total"] += 1
        overall_total += 1
        if result["room_types_match"]:
            categories[label]["matches"] += 1
            overall_matches += 1

    return {
        "categories": categories,
        "category_order": category_labels,
        "overall_matches": overall_matches,
        "overall_total": overall_total,
        "skipped": skipped,
    }
