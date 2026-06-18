"""
Hypergraph class consolidating all operations on BSP-tree hypergraph dicts.

Three dependency tiers:
  Layer 1 (pure Python) -- always available
  Layer 2 (NetworkX)    -- lightweight, already in requirements.txt
  Layer 3 (RGL/.NET)    -- lazy-loaded, only needed for geometry operations
"""

import re
import json
import math
import copy
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _parse_first_json_object(s: str):
    """Parse the first top-level JSON object from *s*, ignoring trailing text.

    Models often emit valid hypergraph JSON followed by commentary or extra tokens,
    which makes ``json.loads`` fail with "Extra data: ...".  ``JSONDecoder.raw_decode``
    stops after the first complete value.
    """
    s = s.strip()
    dec = json.JSONDecoder()
    i = s.find("{")
    if i == -1:
        return None
    try:
        obj, _end = dec.raw_decode(s[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def repair_bsp_tree_names(hypergraph: dict) -> dict:
    """Reassign every node ``name`` from tree layout (``root``, ``rootL``, ``rootR``, …).

    Model outputs sometimes reuse the same ``name`` on different leaves (e.g. two
    ``rootRLL``), which breaks :attr:`Hypergraph.access_graph` and mesh ``unique_id``
    mapping. This walks the BSP tree and sets:

    - ``root`` for the root;
    - ``<parent>L`` / ``<parent>R`` for the first two children;
    - ``<parent>_2``, ``_3``, … if a node has more than two children.

    Leaf ``connected`` lists are rewritten to use the new names. When an old name was
    ambiguous, neighbors are resolved by **symmetric** ``connected`` (each side lists
    the other’s old name) when possible; otherwise the first matching final leaf is used
    and a warning is logged.

    Internal nodes get ``mergeid`` set to the new structural name; final leaves keep
    their room-type ``mergeid``. The special neighbor ``outside`` is left unchanged.

    Args:
        hypergraph: Root BSP dict (must contain ``children``, ``name``, etc.).

    Returns:
        Deep copy of *hypergraph* with updated ``name``, ``mergeid`` (internals), and
        ``connected`` (final nodes only).
    """
    if not isinstance(hypergraph, dict):
        raise TypeError(f"repair_bsp_tree_names expected dict root, got {type(hypergraph)!r}")

    hg = copy.deepcopy(hypergraph)
    all_nodes: List[dict] = []

    def collect(n: dict) -> None:
        all_nodes.append(n)
        for c in n.get("children") or []:
            collect(c)

    collect(hg)

    def assign_names(node: dict, parent_path: str, idx: int) -> None:
        if parent_path == "":
            new_name = "root"
        elif idx == 0:
            new_name = parent_path + "L"
        elif idx == 1:
            new_name = parent_path + "R"
        else:
            new_name = parent_path + f"_{idx}"

        node["_old_name"] = node.get("name")
        node["name"] = new_name
        if not node.get("final"):
            node["mergeid"] = new_name
        children = node.get("children") or []
        for i, ch in enumerate(children):
            assign_names(ch, new_name, i)

    assign_names(hg, "", 0)

    by_old: Dict[str, List[dict]] = {}
    for n in all_nodes:
        old = n.get("_old_name")
        if old is None:
            continue
        key = str(old)
        by_old.setdefault(key, []).append(n)

    for n in all_nodes:
        if not n.get("final"):
            continue
        con = n.get("connected")
        if not con:
            continue
        new_con: List[str] = []
        my_old = n.get("_old_name")
        for s in con:
            st = str(s).strip()
            if st.lower() == "outside":
                new_con.append(st)
                continue
            cands = by_old.get(st, [])
            if len(cands) == 1:
                new_con.append(cands[0]["name"])
            elif len(cands) == 0:
                logger.warning(
                    "repair_bsp_tree_names: leaf %r connected to unknown name %r",
                    my_old,
                    st,
                )
                new_con.append(st)
            else:
                sym = [c for c in cands if my_old in (c.get("connected") or [])]
                if len(sym) == 1:
                    new_con.append(sym[0]["name"])
                else:
                    finals = [c for c in cands if c.get("final")]
                    chosen = finals[0] if finals else cands[0]
                    if len(sym) != 1:
                        logger.warning(
                            "repair_bsp_tree_names: ambiguous neighbor %r for leaf %r "
                            "(%d candidates); picked %r",
                            st,
                            my_old,
                            len(cands),
                            chosen.get("name"),
                        )
                    new_con.append(chosen["name"])
        n["connected"] = new_con

    for n in all_nodes:
        n.pop("_old_name", None)

    return hg


# ---------------------------------------------------------------------------
# Mesh / polygon adjacency primitives
#
# Shared by ``Hypergraph.correct_accessgraph`` (intersection of declared edges
# with mesh adjacency) and ``Hypergraph.adjacency_graph`` /
# ``adjacency_graph_from_polygons`` (full mesh-adjacency graph).
#
# Geometry rules (kept identical to the original
# ``correct_accessgraph._polygons_adjacent`` so behaviour is unchanged):
#   - point contact  -> not adjacent
#   - shared edge    -> adjacent if length > 1e-4
#   - shared overlap -> adjacent if area   > max(1e-4, 1e-3 * min(area))
#   - mixed geometry collection -> any of the above qualifies
# ---------------------------------------------------------------------------


def _polygons_adjacent(a, b) -> bool:
    """True if two Shapely polygons share a non-sliver wall."""
    if a.is_empty or b.is_empty:
        return False
    try:
        inter = a.intersection(b)
    except Exception:
        return False
    if inter.is_empty:
        return False
    gt = inter.geom_type
    if gt == "Point":
        return False
    min_area = max(1e-4, 1e-3 * min(a.area, b.area))
    if gt in ("LineString", "LinearRing", "MultiLineString"):
        return inter.length > 1e-4
    if gt == "Polygon":
        return inter.area > min_area
    if gt == "GeometryCollection":
        for g in inter.geoms:
            if g.geom_type in ("LineString", "LinearRing", "MultiLineString") and g.length > 1e-4:
                return True
            if g.geom_type == "Polygon" and g.area > min_area:
                return True
    return False


def _build_mesh_polygons(mesh, leaf_name_to_id, scale_factor):
    """Return (polys, face_node) parallel lists for each mesh face."""
    from shapely.affinity import scale as _scale
    from shapely.geometry import Polygon

    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    polys, face_node = [], []
    for i in range(n):
        face = mesh.faceList[i]
        uid = getattr(face, "unique_id", "") or ""
        face_node.append(leaf_name_to_id.get(uid))
        pts = face.getPoints()
        corners = [(float(pts[j].X), float(pts[j].Y))
                   for j in range(int(getattr(pts, "Count", 0) or 0))]
        poly = Polygon(corners) if len(corners) >= 3 else Polygon()
        if not poly.is_empty and abs(float(scale_factor) - 1.0) > 1e-9:
            poly = _scale(poly, xfact=float(scale_factor), yfact=float(scale_factor),
                          origin="centroid")
        polys.append(poly)
    return polys, face_node


def _polygon_adjacency_pairs_by_id(polys, ids) -> set[tuple[str, str]]:
    """Return all adjacent (u, v) id-pairs, sorted by integer id."""
    out: set[tuple[str, str]] = set()
    for i, u in enumerate(ids):
        if u is None or polys[i].is_empty:
            continue
        for j in range(i + 1, len(polys)):
            v = ids[j]
            if v is None or polys[j].is_empty or u == v:
                continue
            try:
                if _polygons_adjacent(polys[i], polys[j]):
                    out.add(tuple(sorted((str(u), str(v)), key=int)))
            except Exception:
                continue
    return out


def adjacency_graph_from_polygons(room_polygons, *, scale_factor=1.05):  # re-export for backward compat
    from utils.visualization_helpers import adjacency_graph_from_polygons as _impl
    return _impl(room_polygons, scale_factor=scale_factor)


class Hypergraph:
    """A BSP-tree hypergraph representing a spatial floor plan subdivision.

    Wraps a plain dict and exposes tree traversal, validation, graph
    extraction, area conversion, mesh generation, and response-parsing
    as a single cohesive interface.
    """

    __slots__ = ("_data", "_bounds")

    _rgl_cache = None  # class-level: (RGL, Collections) or None

    # -- construction --------------------------------------------------------

    def __init__(self, data: Union[str, Dict], *, bounds: Optional[Dict] = None):
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict or JSON string, got {type(data).__name__}")
        self._data = data
        self._bounds = bounds

    # -- raw access ----------------------------------------------------------

    @property
    def data(self) -> dict:
        """The underlying BSP tree dict."""
        return self._data

    @property
    def bounds(self) -> Optional[Dict]:
        """Optional boundary dict loaded from a dataset file, or ``None``."""
        return self._bounds

    def to_json(self) -> str:
        return json.dumps(self._data)

    def with_repaired_names(self) -> "Hypergraph":
        """Copy of this hypergraph with :func:`repair_bsp_tree_names` applied to the tree."""
        return Hypergraph(
            repair_bsp_tree_names(copy.deepcopy(self._data)),
            bounds=self._bounds,
        )

    def __repr__(self) -> str:
        name = self._data.get("name", "?")
        return f"Hypergraph(name={name!r}, leaves={self.leaf_count})"

    # -- Layer 1: pure-Python tree operations --------------------------------

    @property
    def leaves(self) -> List[dict]:
        """All leaf nodes (final=True)."""
        result = []

        def _traverse(node):
            if node.get("final", False):
                result.append(node)
            for child in node.get("children", []):
                _traverse(child)

        _traverse(self._data)
        return result

    @property
    def leaf_count(self) -> int:
        return len(self.leaves)

    def has_connectivity(self) -> bool:
        """True if at least one leaf has a non-empty 'connected' list."""
        for leaf in self.leaves:
            if leaf.get("connected"):
                return True
        return False

    @property
    def access_graph(self) -> dict:
        """Extract ``{nodes, edges}`` access graph from leaf connectivity."""
        leaves = self.leaves
        if not leaves:
            return {"nodes": {}, "edges": []}

        leaf_name_to_id = {
            leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)
        }

        nodes = {}
        for leaf in leaves:
            lid = leaf_name_to_id[leaf["name"]]
            nodes[lid] = leaf.get("mergeid", "Unknown")

        edges = []
        edges_set: set = set()
        for leaf in leaves:
            lid = leaf_name_to_id[leaf["name"]]
            for connected_name in leaf.get("connected", []):
                if connected_name in leaf_name_to_id:
                    cid = leaf_name_to_id[connected_name]
                    edge = tuple(sorted([lid, cid], key=lambda x: int(x)))
                    if edge not in edges_set:
                        edges_set.add(edge)
                        edges.append([edge[0], edge[1]])

        return {"nodes": nodes, "edges": edges}

    def access_graph_components(self) -> list:
        """Connected components of the access graph.

        Returns a list of sets, each containing node-id strings that
        belong to the same connected component.  A fully connected
        access graph returns a single-element list.
        """
        import networkx as nx
        from .metrics import accessgraph_dict_to_nx

        g = accessgraph_dict_to_nx(self.access_graph)
        if g.number_of_nodes() == 0:
            return []
        return list(nx.connected_components(g))

    def is_access_graph_connected(self) -> bool:
        """True if the access graph is a single connected component."""
        return len(self.access_graph_components()) <= 1

    # -- validation ----------------------------------------------------------

    def validate(self, accessgraph=None, debug=False, quick=False):
        """Structural validation.

        Returns:
            (is_valid, passed_checks, failed_checks, ged_info)
            When *quick=True*, passed_checks is always [] and ged_info is None.
        """
        if quick:
            return self._validate_quick(self._data)

        import networkx as nx

        passed: list = []
        failed: list = []

        if not isinstance(self._data, dict):
            failed.append("❌ Root is not a dictionary")
            return False, passed, failed, None
        else:
            passed.append("✅ Root is a dictionary")

        if "name" not in self._data:
            failed.append("❌ Missing 'name' property in root")
        else:
            passed.append(f"✅ Root has 'name' property: '{self._data['name']}'")

        if "children" not in self._data:
            failed.append("❌ Missing 'children' property in root")
        else:
            passed.append("✅ Root has 'children' property")

        leaf_node_names: set = set()
        total_nodes = 0
        total_leaf_nodes = 0
        total_internal_nodes = 0

        def collect_leaf_names(node):
            nonlocal total_nodes, total_leaf_nodes, total_internal_nodes
            total_nodes += 1
            if node.get("final", False):
                leaf_node_names.add(node.get("name", ""))
                total_leaf_nodes += 1
            else:
                total_internal_nodes += 1
            for child in node.get("children", []):
                collect_leaf_names(child)

        collect_leaf_names(self._data)

        passed.append(f"✅ Total nodes in tree: {total_nodes}")
        passed.append(f"✅ Leaf nodes (rooms): {total_leaf_nodes}")
        passed.append(f"✅ Internal nodes (splits): {total_internal_nodes}")

        if accessgraph:
            num_ag = len(accessgraph.get("nodes", {}))
            if total_leaf_nodes == num_ag:
                passed.append(f"✅ Leaf node count matches access graph: {total_leaf_nodes} == {num_ag}")
            else:
                failed.append(f"❌ Leaf node count mismatch: hypergraph has {total_leaf_nodes} leaves, access graph has {num_ag} nodes")

        def validate_node(node, path="root"):
            required = ["name", "area", "angle", "mergeid", "final", "children"]
            for field in required:
                if field not in node:
                    failed.append(f"❌ {path}: Missing required field '{field}'")
                else:
                    passed.append(f"✅ {path}: Has required field '{field}'")

            if "angle" in node:
                angle = node["angle"]
                valid_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
                tolerance = 0.1
                is_valid_angle = any(abs(angle - v) < tolerance for v in valid_angles)
                if is_valid_angle:
                    passed.append(f"✅ {path}: Angle {angle:.4f} is horizontal/vertical")
                else:
                    failed.append(f"❌ {path}: Angle {angle:.4f} is not horizontal/vertical (should be 0, π/2, π, or 3π/2)")

            if node.get("final", False):
                if "connected" not in node:
                    failed.append(f"❌ {path}: Leaf node missing 'connected' field")
                else:
                    conn_list = node.get("connected", [])
                    passed.append(f"✅ {path}: Has 'connected' field with {len(conn_list)} connections")

                if node.get("children", []):
                    failed.append(f"❌ {path}: Leaf node (final=true) should have empty children")
                else:
                    passed.append(f"✅ {path}: Leaf node has empty children array")

                connected = node.get("connected", [])
                node_name = node.get("name", "")

                if node_name in connected:
                    failed.append(f"❌ {path}: Self-loop detected - node '{node_name}' references itself in connected array")
                else:
                    passed.append(f"✅ {path}: No self-loops (node doesn't reference itself)")

                for conn in connected:
                    if conn not in leaf_node_names:
                        failed.append(f"❌ {path}: Connected references '{conn}' which is not a leaf node")
                    else:
                        passed.append(f"✅ {path}: Connection to '{conn}' is valid (leaf node)")
                    if "_" in conn and not conn.startswith("root"):
                        failed.append(f"❌ {path}: Connected contains '{conn}' which looks like a mergeid")

            if not node.get("final", False):
                children = node.get("children", [])
                if len(children) != 2:
                    failed.append(f"❌ {path}: Non-leaf node must have exactly 2 children, has {len(children)}")
                else:
                    passed.append(f"✅ {path}: Non-leaf node has exactly 2 children")
                for i, child in enumerate(children):
                    child_suffix = "L" if i == 0 else "R"
                    validate_node(child, path + child_suffix)

        validate_node(self._data)

        ged_info = None
        if accessgraph and not debug:
            logger.info("Computing Graph Edit Distance...")
            from .metrics import compute_ged_from_hypergraph

            ged, _ = compute_ged_from_hypergraph(accessgraph, self._data, "validate")
            if ged is not None:
                ged_info = {"ged": ged}
                passed.append(f"✅ GED computed: {ged:.2f}")
            else:
                passed.append("⚠️ GED computation failed")
        elif accessgraph and debug:
            def extract_leaves(hg):
                lvs = []
                def trav(n):
                    if n.get("final", False):
                        lvs.append(n)
                    for c in n.get("children", []):
                        trav(c)
                trav(hg)
                return lvs

            lvs = extract_leaves(self._data)

            ag_graph = nx.Graph()
            for node_id, room_type in accessgraph.get("nodes", {}).items():
                ag_graph.add_node(node_id, room_type=room_type)
            for edge in accessgraph.get("edges", []):
                ag_graph.add_edge(edge[0], edge[1])

            hg_graph = nx.Graph()
            leaf_map = {leaf["name"]: leaf for leaf in lvs}
            for leaf in lvs:
                hg_graph.add_node(leaf["name"], room_type=leaf.get("mergeid", "Unknown"))
                for cn in leaf.get("connected", []):
                    if cn in leaf_map:
                        hg_graph.add_edge(leaf["name"], cn)

            logger.info("\n" + "=" * 80)
            logger.info("🔍 DEBUG MODE: GRAPH INFORMATION")
            logger.info("=" * 80)
            logger.info("\n📊 ACCESS GRAPH:")
            logger.info(f"   Nodes: {ag_graph.number_of_nodes()}")
            logger.info(f"   Edges: {ag_graph.number_of_edges()}")
            logger.info(f"\n   Node details:")
            for node, data in ag_graph.nodes(data=True):
                neighbors = list(ag_graph.neighbors(node))
                logger.info(f"     Node '{node}': room_type={data['room_type']}, degree={ag_graph.degree(node)}, neighbors={neighbors}")
            logger.info(f"\n   Edge list:")
            for u, v in ag_graph.edges():
                logger.info(f"     {u} <-> {v}")

            logger.info("\n📊 HYPERGRAPH (Leaf Node Connectivity):")
            logger.info(f"   Leaf Nodes: {hg_graph.number_of_nodes()}")
            logger.info(f"   Edges: {hg_graph.number_of_edges()}")
            logger.info(f"\n   Leaf node details:")
            for node, data in hg_graph.nodes(data=True):
                neighbors = list(hg_graph.neighbors(node))
                logger.info(f"     Node '{node}': room_type={data['room_type']}, degree={hg_graph.degree(node)}, neighbors={neighbors}")
            logger.info(f"\n   Edge list:")
            for u, v in hg_graph.edges():
                logger.info(f"     {u} <-> {v}")

            logger.info("\n" + "=" * 80)
            logger.info("⏸️  DEBUG MODE: Skipping GED computation")
            logger.info("=" * 80 + "\n")
            passed.append("⏸️  Debug mode: GED computation skipped")

        is_valid = len(failed) == 0
        return is_valid, passed, failed, ged_info

    @staticmethod
    def _validate_quick(node, _depth=0):
        """Fast recursive structural validation for hot loops."""
        if not isinstance(node, dict):
            return False, [], [f"Node at depth {_depth} is {type(node).__name__}, not dict"], None

        failed = []
        missing = {"name", "area", "children"} - set(node.keys())
        if missing:
            failed.append(f"Missing keys at depth {_depth}: {missing}")

        children = node.get("children", [])
        if not isinstance(children, list):
            return False, [], [f"children is {type(children).__name__}, not list"], None

        if node.get("final", False):
            return len(failed) == 0, [], failed, None

        if len(children) == 0:
            failed.append(f"Non-final node at depth {_depth} has no children")
            return False, [], failed, None

        for i, child in enumerate(children):
            if isinstance(child, str):
                failed.append(f"Child {i} at depth {_depth} is string '{child}', not dict")
                return False, [], failed, None
            child_valid, _, child_failed, _ = Hypergraph._validate_quick(child, _depth + 1)
            failed.extend(child_failed)
            if not child_valid:
                return False, [], failed, None

        return len(failed) == 0, [], failed, None

    # -- transformers (return new Hypergraph) --------------------------------

    def with_ratio_areas(self) -> "Hypergraph":
        """Return a new Hypergraph with areas converted to parent-relative ratios."""
        data = copy.deepcopy(self._data)

        def _convert(node):
            parent_area = node.get("area", 0)
            for child in node.get("children", []):
                _convert(child)
                if parent_area > 0:
                    child["area"] = child["area"] / parent_area
            if node.get("name") == "root":
                node["area"] = 1.0

        _convert(data)
        return Hypergraph(data, bounds=self._bounds)

    def without_tree_node_mesh(self) -> "Hypergraph":
        """Return a new Hypergraph with 'treeNodeMesh' fields stripped."""
        data = copy.deepcopy(self._data)

        def _remove(node):
            if "treeNodeMesh" in node:
                del node["treeNodeMesh"]
            for child in node.get("children", []):
                _remove(child)

        _remove(data)
        return Hypergraph(data, bounds=self._bounds)

    # -- comparison ----------------------------------------------------------

    def compare(self, other: "Hypergraph", verbose: bool = False) -> tuple:
        """Compare structural/semantic equality with another Hypergraph.

        Returns:
            (is_equal, report_dict)
        """
        hg1 = self._data
        hg2 = other._data if isinstance(other, Hypergraph) else other

        report = {
            "equal": True,
            "node_count_equal": True,
            "topology_equal": True,
            "connected_equal": True,
            "mergeid_equal": True,
            "node_count_1": 0,
            "node_count_2": 0,
            "differences": [],
        }

        def count_nodes(node):
            c = 1
            for child in node.get("children", []):
                c += count_nodes(child)
            return c

        def compare_nodes(n1, n2, path="root"):
            name1 = n1.get("name", "")
            name2 = n2.get("name", "")
            if name1 != name2:
                report["topology_equal"] = False
                report["differences"].append(f"Node name mismatch at {path}: '{name1}' vs '{name2}'")
            m1 = n1.get("mergeid", "")
            m2 = n2.get("mergeid", "")
            if m1 != m2:
                report["mergeid_equal"] = False
                report["differences"].append(f"Merge ID mismatch at {path}: '{m1}' vs '{m2}'")
            f1 = n1.get("final", False)
            f2 = n2.get("final", False)
            if f1 != f2:
                report["topology_equal"] = False
                report["differences"].append(f"Final flag mismatch at {path}: {f1} vs {f2}")
            c1 = sorted(n1.get("connected", []))
            c2 = sorted(n2.get("connected", []))
            if c1 != c2:
                report["connected_equal"] = False
                report["differences"].append(f"Connected list mismatch at {path}: {c1} vs {c2}")
            ch1 = n1.get("children", [])
            ch2 = n2.get("children", [])
            if len(ch1) != len(ch2):
                report["topology_equal"] = False
                report["differences"].append(f"Number of children mismatch at {path}: {len(ch1)} vs {len(ch2)}")
                return
            for i, (child1, child2) in enumerate(zip(ch1, ch2)):
                child_path = f"{path}/{child1.get('name', f'child{i}')}"
                compare_nodes(child1, child2, child_path)

        report["node_count_1"] = count_nodes(hg1)
        report["node_count_2"] = count_nodes(hg2)
        if report["node_count_1"] != report["node_count_2"]:
            report["node_count_equal"] = False
            report["differences"].append(f"Node count mismatch: {report['node_count_1']} vs {report['node_count_2']}")

        compare_nodes(hg1, hg2)

        report["equal"] = (
            report["node_count_equal"]
            and report["topology_equal"]
            and report["connected_equal"]
            and report["mergeid_equal"]
        )

        if verbose:
            print("=" * 60)
            print("HYPERGRAPH COMPARISON REPORT")
            print("=" * 60)
            print(f"Overall Result: {'✓ EQUAL' if report['equal'] else '✗ NOT EQUAL'}")
            print(f"\nNode Count: {report['node_count_1']} vs {report['node_count_2']} - {'✓' if report['node_count_equal'] else '✗'}")
            print(f"Topology:   {'✓ EQUAL' if report['topology_equal'] else '✗ DIFFERENT'}")
            print(f"Connected:  {'✓ EQUAL' if report['connected_equal'] else '✗ DIFFERENT'}")
            print(f"Merge IDs:  {'✓ EQUAL' if report['mergeid_equal'] else '✗ DIFFERENT'}")
            if report["differences"]:
                print(f"\nFound {len(report['differences'])} difference(s):")
                for i, diff in enumerate(report["differences"], 1):
                    print(f"  {i}. {diff}")
            else:
                print("\n✓ No differences found - hypergraphs are identical!")
            print("=" * 60)

        return report["equal"], report

    # -- Layer 3: RGL geometry operations (lazy-loaded) ----------------------

    @classmethod
    def _init_rgl(cls):
        """Lazily initialise and cache ``(RGL, Collections)``."""
        if cls._rgl_cache is None:
            import clr
            from System import Collections as SysCollections
            from hypergraph.api.lib.tools import RGL as RGLModule
            cls._rgl_cache = (RGLModule, SysCollections)
        return cls._rgl_cache

    @classmethod
    def init_rgl(cls):
        """Public entry point to pre-initialise RGL.

        Returns ``(RGL, Collections)`` and caches for subsequent calls
        to :meth:`to_mesh` and :meth:`to_apartment`.
        """
        return cls._init_rgl()

    @staticmethod
    def _polygon_to_boundary_mesh(polygon, RGL, Collections):
        """Convert a list of ``(x, y)`` tuples to an ``RGL.NMesh``."""
        points_list = Collections.Generic.List[RGL.Vec3d]()
        for x, y in polygon:
            points_list.Add(RGL.Vec3d(x=float(x), y=float(y), z=0.0))
        bf = RGL.NFace(points_list)
        nfl = Collections.Generic.List[RGL.NFace]()
        nfl.Add(bf)
        return RGL.NMesh(nfl)

    def to_mesh(self, bounds=None, polygon=None, boundary_mesh=None,
                already_ratios=False, RGL=None, Collections=None):
        """Subdivide this hypergraph into a room mesh via RGL.

        Provide exactly ONE boundary source:
          *bounds*      -- dict with 'corners' key (from dataset files)
          *polygon*     -- list of (x, y) tuples
          *boundary_mesh* -- pre-built RGL.NMesh

        Args:
            already_ratios: If True, skip area-to-ratio conversion.
            RGL: Pre-initialized RGL module (optional, lazy-loads if None).
            Collections: Pre-initialized System.Collections (optional).

        Returns:
            RGL.NMesh with room faces.
        """
        if RGL is None or Collections is None:
            RGL, Collections = self._init_rgl()

        if already_ratios:
            hg_ratio = self._data
        elif self._data.get("area") == 1.0:
            hg_ratio = self._data
        else:
            hg_ratio = self.with_ratio_areas().data

        if boundary_mesh is not None:
            pass
        elif bounds is not None:
            corners = bounds.get("corners", [])
            polygon = [(c["X"], c["Y"]) for c in corners]
            boundary_mesh = self._polygon_to_boundary_mesh(polygon, RGL, Collections)
        elif polygon is not None:
            boundary_mesh = self._polygon_to_boundary_mesh(polygon, RGL, Collections)
        else:
            raise ValueError("Provide one of: bounds, polygon, or boundary_mesh")

        data_node = RGL.DataNode.deserializeDataNode(json.dumps(hg_ratio))
        return RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)

    def mesh_face_labels(self, mesh, distinct=None) -> list:
        """Build display labels for each face in an assembled RGL mesh.

        Uses ``unique_id`` metadata on each face to map it back to the
        corresponding BSP leaf / access-graph node, then produces a
        human-readable label (e.g. ``living1``, ``kitchen``).

        Args:
            mesh: RGL NMesh returned by :meth:`to_mesh`.
            distinct: Optional dict mapping access-graph node-id to
                label string (as returned by
                :meth:`HypergraphPlotter.make_distinct_labels`).
                When *None*, labels are built automatically from
                :pyattr:`access_graph`.

        Returns:
            List of label strings, one per face in ``mesh.faceList``.
        """
        leaf_name_to_ag_id = {
            leaf["name"]: str(i + 1)
            for i, leaf in enumerate(self.leaves)
        }

        if distinct is None:
            ag = self.access_graph
            ag_rooms = {
                "nodes": {k: v for k, v in ag.get("nodes", {}).items()
                          if k != "outside"},
                "edges": [e for e in ag.get("edges", [])
                          if "outside" not in e],
            }
            distinct = HypergraphPlotter.make_distinct_labels(ag_rooms)

        n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
        labels = []
        for i in range(n):
            face = mesh.faceList[i]
            uid = getattr(face, 'unique_id', '')
            ag_id = leaf_name_to_ag_id.get(uid, '')
            labels.append(
                distinct.get(ag_id, getattr(face, 'merge_id', 'unknown')))
        return labels

    def mesh_entry_door_face_idx(self, mesh):
        """See :func:`utils.visualization_helpers.mesh_entry_door_face_idx`."""
        from utils.visualization_helpers import mesh_entry_door_face_idx as _impl
        return _impl(self, mesh)

    def mesh_door_face_pairs(self, mesh, *, access_graph=None, bounds=None) -> list:
        """See :func:`utils.visualization_helpers.mesh_door_face_pairs`."""
        from utils.visualization_helpers import mesh_door_face_pairs as _impl
        return _impl(self, mesh, access_graph=access_graph, bounds=bounds)

    def _compute_mesh_adjacency(self, bounds, *, scale_factor=1.05):
        """Return (mesh_adj, id_to_mergeid, node_ids) or None if no leaves."""
        leaves = self.leaves
        if not leaves:
            return None
        leaf_name_to_id = {leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)}
        node_ids = set(leaf_name_to_id.values())
        id_to_mergeid = {str(i + 1): leaves[i].get("mergeid", "Unknown") for i in range(len(leaves))}
        mesh = self.to_mesh(bounds=bounds)
        n_faces = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
        if n_faces <= 0:
            return set(), id_to_mergeid, node_ids
        polys, face_node = _build_mesh_polygons(mesh, leaf_name_to_id, scale_factor)
        return _polygon_adjacency_pairs_by_id(polys, face_node), id_to_mergeid, node_ids

    def correct_accessgraph(self, bounds: dict, *, scale_factor: float = 1.05) -> dict:
        """Return corrected access graph using mesh adjacency.

        Keeps predicted edges that are mesh-adjacent; adds mesh edges as bridges
        until the graph is connected.  Reasoning is on node ids, not room types.
        """
        out = self._compute_mesh_adjacency(bounds, scale_factor=scale_factor)
        if out is None:
            return {"nodes": {}, "edges": []}
        mesh_adj, id_to_mergeid, node_ids = out

        ag = self.access_graph if isinstance(self.access_graph, dict) else {}

        def _ep(u, v):
            return tuple(sorted((str(u), str(v)), key=int))

        raw = [(str(e[0]), str(e[1])) for e in (ag.get("edges") or [])
               if isinstance(e, (list, tuple)) and len(e) == 2
               and str(e[0]) in node_ids and str(e[1]) in node_ids and str(e[0]) != str(e[1])]

        if not mesh_adj:
            return {
                "nodes": {k: id_to_mergeid[k] for k in node_ids if k in id_to_mergeid},
                "edges": [[p[0], p[1]] for p in sorted({_ep(u, v) for u, v in raw})],
            }

        parent = {n: n for n in node_ids}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        def _union(x, y):
            px, py = _find(x), _find(y)
            if px == py: return False
            parent[px] = py; return True

        seen: set = set()
        final = []
        for u, v in raw:
            ep = _ep(u, v)
            if ep in mesh_adj and ep not in seen:
                seen.add(ep); final.append(ep); _union(u, v)

        for u, v in sorted(mesh_adj, key=lambda e: (int(e[0]), int(e[1]))):
            if u in node_ids and v in node_ids and _union(u, v):
                final.append(_ep(u, v))

        out_seen: set = set()
        out_edges = []
        for ep in final:
            if ep not in out_seen:
                out_seen.add(ep); out_edges.append([ep[0], ep[1]])
        out_edges.sort(key=lambda e: (int(e[0]), int(e[1])))

        return {
            "nodes": {nid: id_to_mergeid[nid] for nid in sorted(node_ids, key=int)},
            "edges": out_edges,
        }
    def to_apartment(self, polygon, apt_id="generated",
                     RGL=None, Collections=None, verbose=False):
        """Convert this hypergraph + polygon boundary to an RGL Apartment.

        Builds the Apartment shell via ``ReadApartmentsFromJson`` (which
        populates bounds, facade, circulation, adjacency, and metadata),
        then **replaces** the room mesh with the output of :meth:`to_mesh`
        so that the geometry is identical regardless of whether the caller
        uses ``to_mesh`` or ``to_apartment``.

        Args:
            polygon: list of ``(x, y)`` tuples defining the apartment boundary.
            apt_id: identifier string embedded in the apartment record.
            RGL: Pre-initialized RGL module (optional, lazy-loads if None).
            Collections: Pre-initialized Collections (optional).
            verbose: If True, print apartment info after loading.

        Returns:
            ``(apartment, apt_id)`` -- an RGL Apartment object and its id.
        """
        import os
        import tempfile

        if RGL is None or Collections is None:
            RGL, Collections = self._init_rgl()

        corners = [
            {"X": float(x), "Y": float(y), "Z": 0, "Mag": math.sqrt(x * x + y * y)}
            for x, y in polygon
        ]
        facade = [
            {"start": corners[i], "end": corners[(i + 1) % len(corners)]}
            for i in range(len(corners))
        ]
        entry = {
            "database": "generated",
            "id": apt_id,
            "area": 0,
            "bedrooms": 0,
            "bathrooms": 0,
            "bounds": json.dumps({"corners": corners}),
            "facade": json.dumps(facade),
            "circulation": "[]",
            "split": json.dumps(self._data),
            "country": "None",
            "city": "None",
            "name": apt_id,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump([entry], tmp)
            tmp_path = tmp.name
        apartments = RGL.BuildingSolver.ALibrary.ReadApartmentsFromJson(tmp_path)
        os.unlink(tmp_path)

        apt = apartments[0]

        rooms_mesh = self.to_mesh(polygon=polygon, RGL=RGL, Collections=Collections)
        apt.rooms = rooms_mesh
        apt.rooms.UpdateAdjacencyListWithFaceStrings()

        if verbose:
            print(f"Loaded: {apt_id} - {apt.rooms.faceList.Count} rooms, "
                  f"{apt.bounds.Area:.1f}m\u00b2")
        return apt, apt_id

    # -- class methods (constructors) ----------------------------------------

    @classmethod
    def from_qwen_response(cls, response, *, allow_raw_json_fallback=False,
                           silent=False) -> "Optional[Hypergraph]":
        """Extract a Hypergraph from a Qwen model response string.

        Returns None if no valid hypergraph JSON is found.
        """
        def _strip_fences(s: str) -> str:
            s = s.strip()
            if s.startswith("```json"):
                s = s[7:]
            elif s.startswith("```"):
                s = s[3:]
            if s.endswith("```"):
                s = s[:-3]
            return s.strip()

        assistant_marker = "<|im_start|>assistant"
        assistant_idx = response.find(assistant_marker)

        if assistant_idx != -1:
            json_start = assistant_idx + len(assistant_marker)
            json_section = response[json_start:]
            end_marker = "<|im_end|>"
            end_idx = json_section.find(end_marker)
            if end_idx != -1:
                json_section = json_section[:end_idx]
            json_section = _strip_fences(json_section)
            obj = None
            used_first_object_only = False
            try:
                obj = json.loads(json_section)
            except json.JSONDecodeError as e:
                obj = _parse_first_json_object(json_section)
                if obj is not None:
                    used_first_object_only = True
                elif not silent:
                    logger.warning(f"Found assistant section but failed to parse JSON: {e}")
                    logger.debug(f"Attempted to parse: {json_section[:200]}...")

            if obj is not None and isinstance(obj, dict) and "name" in obj and "children" in obj:
                if not silent:
                    if used_first_object_only:
                        logger.info(
                            "✅ Extracted hypergraph JSON from assistant section "
                            "(first JSON object only; trailing model text ignored)"
                        )
                    else:
                        logger.info("✅ Successfully extracted JSON from assistant section")
                return cls(obj)
            if obj is not None and isinstance(obj, dict) and not silent:
                logger.warning("Parsed JSON but missing required fields ('name' or 'children')")

        if allow_raw_json_fallback:
            json_section = _strip_fences(response)
            obj = None
            try:
                obj = json.loads(json_section)
            except json.JSONDecodeError:
                obj = _parse_first_json_object(json_section)
            if isinstance(obj, dict) and "name" in obj and "children" in obj:
                if not silent:
                    logger.info("✅ Successfully extracted JSON from raw response")
                return cls(obj)

        if not silent:
            logger.error("❌ Could not find valid JSON in response")
        return None

    @classmethod
    def from_gpt_response(cls, response) -> "Optional[Hypergraph]":
        """Extract a Hypergraph from a GPT model response string.

        Returns None if no valid hypergraph JSON is found.
        """
        json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(json_block_pattern, response, re.DOTALL)

        def _unwrap_root(obj):
            if isinstance(obj, dict) and "root" in obj and isinstance(obj["root"], dict):
                logger.info("⚠️  Found nested 'root' wrapper, extracting...")
                obj = obj["root"]
                obj["name"] = "root"
            return obj

        for match in matches:
            try:
                obj = _unwrap_root(json.loads(match.strip()))
                if isinstance(obj, dict) and "name" in obj and "children" in obj:
                    logger.info("✅ Extracted JSON from markdown code block")
                    return cls(obj)
            except json.JSONDecodeError:
                continue

        potential_jsons = []
        stack: list = []
        start_idx = None
        for i, char in enumerate(response):
            if char == "{":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        potential_jsons.append(response[start_idx : i + 1])

        potential_jsons.sort(key=len, reverse=True)
        for candidate in potential_jsons:
            try:
                obj = _unwrap_root(json.loads(candidate))
                if isinstance(obj, dict) and "name" in obj and "children" in obj:
                    logger.info("✅ Extracted JSON from response body")
                    return cls(obj)
            except json.JSONDecodeError:
                continue

        try:
            obj = _unwrap_root(json.loads(response.strip()))
            if isinstance(obj, dict) and "name" in obj and "children" in obj:
                logger.info("✅ Parsed entire response as JSON")
                return cls(obj)
        except json.JSONDecodeError:
            pass

        logger.warning("❌ Could not extract valid JSON from response")
        logger.debug(f"Response preview: {response[:500]}...")
        return None

    @classmethod
    def from_file(cls, path) -> "Hypergraph":
        """Load from a dataset .txt file.

        Expects at least a ``"hypergraph"`` key.  If a ``"bounds"`` key is
        present it is parsed and stored on the returned instance (accessible
        via the :pyattr:`bounds` property).
        """
        with open(path, "r") as f:
            data = json.load(f)
        hg_raw = data.get("hypergraph", "")
        if isinstance(hg_raw, str):
            hg_raw = json.loads(hg_raw) if hg_raw else {}

        bounds_raw = data.get("bounds")
        if isinstance(bounds_raw, str):
            bounds_raw = json.loads(bounds_raw) if bounds_raw else None

        return cls(hg_raw, bounds=bounds_raw)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class HypergraphPlotter:
    """Visualises hypergraph structures and apartment layouts.

    Handles:
    - Hierarchical graph structures with subdivision and adjacency edges
    - Spatial partitioning and room connectivity
    - Complex apartment layouts with rooms, boundaries, and circulation

    Plotting dependencies (``matplotlib``, ``networkx``, ``numpy``) are
    imported lazily so the rest of ``hypergraph.py`` stays lightweight.
    """

    ROOM_COLORS = {
        # Capitalized (MSDPlotter format)
        'Bedroom': '#1f77b4',
        'Livingroom': '#e6550d',
        'Kitchen': '#fd8d3c',
        'Dining': '#fdae6b',
        'Corridor': '#fdd0a2',
        'Stairs': '#72246c',
        'Storeroom': '#5254a3',
        'Bathroom': '#6b6ecf',
        'Balcony': '#BEBEBE',
        # Lowercase (Hypergraph Dataset format)
        'bed': '#5B448C',
        'living': '#ED6D87',
        'kitchen': '#9A5387',
        'bath': '#FFE953',
        'foyer': '#ED6D87',
        'extra': '#FF9FBB',
        'dining': '#fdae6b',
        'corridor': '#fdd0a2',
        'bathroom': '#FFE953',
        'balcony': '#BEBEBE',
        'storage': '#5254a3',
        'storeroom': '#5254a3',
        'stairs': '#72246c',
        'outside': '#FF0000',
    }

    ROOM_COLOR_MAP = ROOM_COLORS  # legacy alias

    MESH_SLIVER_FRAC = 1e-3  # faces < 0.1 % of total mesh area are Clipper noise

    # Pairs of room types that should render as a single open zone -- no wall
    # corridor and no door between them, just a thin black delineation line
    # along their shared edge. Pairs are direction-independent.  Each room in
    # the zone keeps its own colored fill and label; only the wall corridor
    # between them is suppressed.  Same-type entries (e.g. ``frozenset({'foyer'})``)
    # cover same-type adjacencies like foyer-foyer or living-living.
    OPEN_ZONE_TYPE_PAIRS = frozenset({
        frozenset({'foyer', 'kitchen'}),
        frozenset({'foyer', 'living'}),
        frozenset({'living', 'kitchen'}),
        frozenset({'living'}),
        frozenset({'foyer'}),
    })

    def __init__(self, color_map=None):
        self.color_map = color_map if color_map is not None else self.ROOM_COLORS

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _mesh_min_area(face_list):
        """Return the minimum visible face area for *face_list*.

        Sums every face's area, then returns
        ``total * MESH_SLIVER_FRAC``.  Faces below this value are
        Clipper numerical artifacts and should be skipped when rendering.
        """
        total = sum(getattr(face_list[i], 'Area', 0)
                    for i in range(face_list.Count))
        return total * HypergraphPlotter.MESH_SLIVER_FRAC

    @staticmethod
    def create_color_map_with_roomtype(graph):
        """Return a list of colours for *graph* nodes based on room type."""
        color_map = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            is_final = node_data.get('final', False)
            mergeid = node_data.get('mergeid', None)
            if is_final:
                color_map.append(
                    HypergraphPlotter.ROOM_COLORS.get(mergeid, '#FF6B6B') if mergeid else '#FF6B6B'
                )
            else:
                if node == 'root' or mergeid == 'root':
                    color_map.append('#505050')
                else:
                    color_map.append('#D3D3D3')
        return color_map

    @staticmethod
    def count_room_nodes(graph):
        """Count subdivision vs room nodes and their degree distributions."""
        numSubdNodes = 0
        numRoomNodes = 0
        degreeSubd = []
        degreeRoom = []
        degreeDict = graph.degree()
        for node in graph.nodes():
            mergeid = graph.nodes[node].get('mergeid', None)
            val = degreeDict[node]
            if node == 'root':
                numSubdNodes += 1
                degreeSubd.append(val)
            elif mergeid and mergeid in HypergraphPlotter.ROOM_COLORS:
                numRoomNodes += 1
                degreeRoom.append(val)
            else:
                numSubdNodes += 1
                degreeSubd.append(val)
        return [numSubdNodes, numRoomNodes, degreeSubd, degreeRoom]

    @staticmethod
    def add_nodes_edges(graph, data):
        """Recursively add nodes and edges from hierarchical *data*."""
        name = data["name"]
        graph.add_node(name, area=data["area"], angle=data["angle"], mergeid=data["mergeid"])
        for child in data["children"]:
            graph.add_node(child["name"], area=child["area"],
                           angle=child["angle"], mergeid=child["mergeid"])
            graph.add_edge(name, child["name"])
            HypergraphPlotter.add_nodes_edges(graph, child)

    @staticmethod
    def add_hypergraph(graph, data):
        """Recursively add hypergraph structure with subdivision and adjacency edges."""
        if 'name' not in data:
            return
        node_name = data['name']
        graph.add_node(node_name, **data)
        if 'connected' in data:
            for connected_node in data['connected']:
                graph.add_edge(node_name, connected_node, type='adjacency')
        if 'children' in data:
            for child in data['children']:
                graph.add_edge(node_name, child['name'], type='subdiv')
                HypergraphPlotter.add_hypergraph(graph, child)

    # -- main plotting methods -----------------------------------------------

    def plot_hypergraph(self, json_data, title=None, figsize=(10, 8),
                        node_scale=1):
        """Plot a hypergraph from JSON data. Returns the NetworkX graph."""
        import networkx as nx
        import matplotlib.pyplot as plt

        if isinstance(json_data, str):
            data = json.loads(json_data)
        elif isinstance(json_data, dict):
            data = json_data
        else:
            raise ValueError("json_data must be either a JSON string or a dictionary")

        graph = nx.Graph()
        self.add_hypergraph(graph, data)

        color_map = self.create_color_map_with_roomtype(graph)

        subdiv_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'subdiv']
        adj_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'adjacency']

        subgraph = graph.edge_subgraph(subdiv_edges)
        layout = nx.kamada_kawai_layout(subgraph)

        plt.figure(figsize=figsize)

        root_node = None
        for node, nd in graph.nodes(data=True):
            if nd.get('mergeid', None) == 'root':
                root_node = node
                break

        if root_node is not None:
            distances = nx.shortest_path_length(graph, target=root_node)
            max_distance = max(distances.values()) if distances else 0

            for u, v, edge_data in graph.edges(data=True):
                edge_type = edge_data.get('type', None)
                if edge_type == 'subdiv':
                    lw_u = 3 + (max_distance - distances.get(u, 0)) * 2
                    lw_v = 3 + (max_distance - distances.get(v, 0)) * 2
                    nx.draw_networkx_edges(graph, pos=layout, edgelist=[(u, v)],
                                           edge_color='#E6E6E6',
                                           width=max(lw_u, lw_v))
                elif edge_type == 'adjacency':
                    nx.draw_networkx_edges(graph, pos=layout, edgelist=[(u, v)],
                                           edge_color='black', width=1.5, style='dotted')

        node_sizes = [nd.get('area', 0.5) * node_scale for _, nd in graph.nodes(data=True)]
        nx.draw_networkx_nodes(graph, pos=layout, node_color=color_map,
                               edgecolors='black', linewidths=0.5, node_size=node_sizes)

        num_subdiv = len(subdiv_edges)
        num_adj = len(adj_edges)
        total_area = sum(nd.get('area', 0) for _, nd in graph.nodes(data=True))

        hfont = {'fontname': 'sans-serif'}
        if title:
            plt.title(title, **hfont)
        plt.xlabel(f"N Subdivision: {num_subdiv}\nN Adjacency: {num_adj}\n"
                   f"Total Area: {round(total_area, 2)} m²\nNodes: {len(graph.nodes())}",
                   **hfont)
        plt.box(False)
        plt.tight_layout()
        plt.show()
        return graph

    @staticmethod
    def _layout_by_component(g, components):
        """Lay out each connected component independently, then tile them.

        Produces much cleaner results than a single global layout because
        disconnected components no longer compete for the same space.
        """
        import networkx as nx

        def _layout_one(sub):
            if sub.number_of_nodes() == 1:
                n = list(sub.nodes())[0]
                return {n: (0.0, 0.0)}
            return nx.kamada_kawai_layout(sub, scale=2.0)

        if len(components) <= 1:
            return _layout_one(g)

        pos = {}
        comp_boxes = []
        for comp in sorted(components, key=len, reverse=True):
            sub = g.subgraph(comp)
            sub_pos = _layout_one(sub)

            if len(sub_pos) > 1:
                xs = [p[0] for p in sub_pos.values()]
                ys = [p[1] for p in sub_pos.values()]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                cx = (max(xs) + min(xs)) / 2
                cy = (max(ys) + min(ys)) / 2
            else:
                w, h, cx, cy = 0.0, 0.0, 0.0, 0.0

            centred = {n: (p[0] - cx, p[1] - cy) for n, p in sub_pos.items()}
            comp_boxes.append((centred, w, h))

        gap = 1.0
        x_cursor = 0.0
        for centred, w, h in comp_boxes:
            offset_x = x_cursor + max(w, 0.5) / 2
            for n, (px, py) in centred.items():
                pos[n] = (px + offset_x, py)
            x_cursor += max(w, 0.5) + gap

        return pos

    @staticmethod
    def make_distinct_labels(ag_dict: dict) -> dict[str, str]:
        """Build distinct labels like ``foyer1``, ``foyer2`` for each node.

        When a room type appears only once, the label is just the type
        (e.g. ``living``).  When it appears more than once, a 1-based
        index is appended (e.g. ``bed1``, ``bed2``).

        Args:
            ag_dict: Access-graph dict with ``nodes`` mapping
                     node-id -> room-type.

        Returns:
            Dict mapping node-id -> distinct label string.
        """
        from collections import Counter

        nodes = ag_dict.get("nodes", {})
        type_counts: Counter = Counter(nodes.values())
        type_seen: dict[str, int] = {}
        labels: dict[str, str] = {}
        for nid in sorted(nodes, key=lambda x: int(x)):
            rtype = nodes[nid]
            if type_counts[rtype] > 1:
                type_seen[rtype] = type_seen.get(rtype, 0) + 1
                labels[nid] = f"{rtype}{type_seen[rtype]}"
            else:
                labels[nid] = rtype
        return labels

    def plot_access_graph_on_axis(self, ag_dict, ax, title=None,
                                  components=None, node_labels=None):
        """Plot an access graph on a specific *ax*.

        Args:
            ag_dict: Access-graph dict with ``nodes`` and ``edges`` keys
                     (as returned by :pyattr:`Hypergraph.access_graph`).
            ax: Matplotlib axis to draw on.
            title: Optional axis title.
            components: Optional list of node-id sets (connected
                        components).  Used to lay out each component
                        independently for a cleaner graph.
            node_labels: Optional dict mapping node-id -> label string.
                         When provided, labels are drawn on the nodes.
        """
        import networkx as nx
        from .metrics import accessgraph_dict_to_nx

        g = accessgraph_dict_to_nx(ag_dict)
        if g.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No nodes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.axis("off")
            return

        if components is None:
            components = list(nx.connected_components(g))

        node_fill = [
            self.ROOM_COLORS.get(g.nodes[n].get("label", ""), "#CCCCCC")
            for n in g.nodes()
        ]

        pos = self._layout_by_component(g, components)

        nx.draw_networkx_edges(
            g, pos, edge_color='#b8b5b4', width=4.0,
            style='dotted', ax=ax)

        nx.draw_networkx_nodes(
            g, pos, node_color=node_fill,
            edgecolors='blue', linewidths=0.0,
            node_size=1000, ax=ax)

        if node_labels is not None:
            nx.draw_networkx_labels(
                g, pos, labels=node_labels,
                font_size=14, font_weight="bold", ax=ax)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    def display_single_graph(self, dfX, i):
        """Display a single graph from a DataFrame row."""
        import matplotlib.pyplot as plt

        graph = dfX['graph'][i]
        self.display_graph(
            graph, dfX['graph_detail_subd'][i], dfX['graph_detail_adj'][i],
            dfX['id'][i], dfX['city'][i], dfX['area'][i], dfX['bedrooms'][i],
            plt.subplots(figsize=(10, 10))[1],
        )
        plt.show()

    def display_graph(self, graph, subd, adj, id_val, city, area, bedrooms, ax,
                      node_scale=1):
        """Display a graph with detailed information on *ax*."""
        import networkx as nx

        color_map = self.create_color_map_with_roomtype(graph)
        subdiv_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'subdiv']
        subgraph = graph.edge_subgraph(subdiv_edges)
        layout = nx.kamada_kawai_layout(subgraph)

        root_node = None
        for node, data in graph.nodes(data=True):
            if data.get('mergeid', None) == 'root':
                root_node = node
                break

        if root_node is not None:
            distances = nx.shortest_path_length(graph, target=root_node)
            max_distance = max(distances.values())
            for u, v, edge_data in graph.edges(data=True):
                edge_type = edge_data.get('type', None)
                if edge_type == 'subdiv':
                    lw_u = 3 + (max_distance - distances[u]) * 2
                    lw_v = 3 + (max_distance - distances[v]) * 2
                    nx.draw_networkx_edges(graph, pos=layout, edgelist=[(u, v)],
                                           edge_color='#E6E6E6',
                                           width=max(lw_u, lw_v), ax=ax)
                elif edge_type == 'adjacency':
                    nx.draw_networkx_edges(graph, pos=layout, edgelist=[(u, v)],
                                           edge_color='black', width=1.5,
                                           style='dotted', ax=ax)

        node_sizes = [nd['area'] * node_scale for _, nd in graph.nodes(data=True)]
        nx.draw_networkx_nodes(graph, pos=layout, node_color=color_map,
                               edgecolors='black', linewidths=0.5,
                               node_size=node_sizes, ax=ax)

        ax.set_title(f"N Subdivision: {subd} \n N Adjacency: {adj}\nID: {id_val}\n "
                     f"City: {city}\nArea: {round(area, 2)} m2\nBedrooms: {bedrooms}")
        ax.axis('off')

    # -- apartment layout methods --------------------------------------------

    def plot_complex_apartment(self, apartment, title="Complex Apartment Layout", figsize=(12, 8)):
        """Plot a complex apartment layout. Returns ``(fig, ax)``."""
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        boundary_corners = []
        for i in range(apartment.bounds.getPoints().Count):
            corner = apartment.bounds.getPoints()[i]
            boundary_corners.append((corner.X, corner.Y))

        if boundary_corners:
            bx = [pt[0] for pt in boundary_corners] + [boundary_corners[0][0]]
            by = [pt[1] for pt in boundary_corners] + [boundary_corners[0][1]]
            ax.plot(bx, by, 'k-', linewidth=3, label='Apartment Boundary')

        total_area = 0
        room_counts = {}
        room_types_plotted = set()

        if hasattr(apartment, 'rooms') and apartment.rooms:
            min_area = self._mesh_min_area(apartment.rooms.faceList)
            for i in range(apartment.rooms.faceList.Count):
                face = apartment.rooms.faceList[i]
                area_val = getattr(face, 'Area', 0)
                if area_val < min_area:
                    continue
                room_type = getattr(face, 'merge_id', 'unknown')
                total_area += area_val
                room_counts[room_type] = room_counts.get(room_type, 0) + 1

                room_corners = []
                for j in range(face.getPoints().Count):
                    c = face.getPoints()[j]
                    room_corners.append((c.X, c.Y))

                if room_corners and len(room_corners) >= 3:
                    rx = [pt[0] for pt in room_corners]
                    ry = [pt[1] for pt in room_corners]
                    color = self.color_map.get(room_type, '#CCCCCC')
                    legend_label = f'{room_type}' if room_type not in room_types_plotted else ""
                    room_types_plotted.add(room_type)
                    ax.fill(rx, ry, color=color, alpha=0.8,
                            edgecolor='black', linewidth=1.5, label=legend_label)
                    cx, cy = np.mean(rx), np.mean(ry)
                    ax.text(cx, cy, f'{room_type}\n{area_val:.1f}m²',
                            ha='center', va='center', fontsize=18, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        if hasattr(apartment, 'facade') and apartment.facade:
            for i in range(apartment.facade.Count):
                fl = apartment.facade[i]
                ax.plot([fl.start.X, fl.end.X], [fl.start.Y, fl.end.Y],
                        'r-', linewidth=4, alpha=0.7, label='Facade' if i == 0 else "")

        if hasattr(apartment, 'circulation') and apartment.circulation:
            for i in range(apartment.circulation.Count):
                cl = apartment.circulation[i]
                ax.plot([cl.start.X, cl.end.X], [cl.start.Y, cl.end.Y],
                        'b-', linewidth=4, alpha=0.7, label='Circulation' if i == 0 else "")

        if hasattr(apartment, 'doors') and apartment.doors:
            dx, dy = [], []
            for i in range(apartment.doors.Count):
                door = apartment.doors[i]
                dx.append(door.X); dy.append(door.Y)
            if dx and dy:
                ax.scatter(dx, dy, c='orange', s=100, marker='o',
                           edgecolors='black', linewidth=2, label='Doors', zorder=5)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)

        summary_text = f"Rooms: {len(room_counts)}\nTotal Area: {total_area:.1f}m²"
        room_summary = ", ".join([f"{cnt}×{rm}" for rm, cnt in room_counts.items()])
        if room_summary:
            summary_text += f"\n{room_summary}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 0.85))

        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_complex_apartment_on_axis(self, apartment, ax, title="Apartment Layout"):
        """Plot apartment on a specific axis (for subplots)."""
        import numpy as np

        boundary_corners = []
        for i in range(apartment.bounds.getPoints().Count):
            corner = apartment.bounds.getPoints()[i]
            boundary_corners.append((corner.X, corner.Y))

        if boundary_corners:
            bx = [pt[0] for pt in boundary_corners] + [boundary_corners[0][0]]
            by = [pt[1] for pt in boundary_corners] + [boundary_corners[0][1]]
            ax.plot(bx, by, 'k-', linewidth=2)

        total_area = 0
        if hasattr(apartment, 'rooms') and apartment.rooms:
            min_area = self._mesh_min_area(apartment.rooms.faceList)
            for i in range(apartment.rooms.faceList.Count):
                face = apartment.rooms.faceList[i]
                area_val = getattr(face, 'Area', 0)
                if area_val < min_area:
                    continue
                room_type = getattr(face, 'merge_id', 'unknown')
                total_area += area_val
                room_corners = []
                for j in range(face.getPoints().Count):
                    c = face.getPoints()[j]
                    room_corners.append((c.X, c.Y))
                if room_corners and len(room_corners) >= 3:
                    rx = [pt[0] for pt in room_corners]
                    ry = [pt[1] for pt in room_corners]
                    color = self.color_map.get(room_type, '#CCCCCC')
                    ax.fill(rx, ry, color=color, alpha=0.8,
                            edgecolor='black', linewidth=1)
                    ax.text(np.mean(rx), np.mean(ry), room_type,
                            ha='center', va='center', fontsize=16, fontweight='bold')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(0.02, 0.98, f"Area: {total_area:.1f}m²", transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))

    def plot_apartment_comparison(self, apartments, titles=None, figsize=(15, 5)):
        """Plot multiple apartments side by side. Returns ``(fig, axes)``."""
        import matplotlib.pyplot as plt

        n = len(apartments)
        if titles is None:
            titles = [f"Apartment {i+1}" for i in range(n)]
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        for i, (apt, t) in enumerate(zip(apartments, titles)):
            self.plot_complex_apartment_on_axis(apt, axes[i], t)
        plt.tight_layout()
        plt.show()
        return fig, axes

    def simple_plot_generated_mesh(self, mesh):
        """Plot a generated mesh coloured by room type. Returns ``(fig, ax)``."""
        from shapely.geometry import Polygon
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        min_area = self._mesh_min_area(mesh.faceList)
        for i in range(mesh.faceList.Count):
            face = mesh.faceList[i]
            if getattr(face, 'Area', 0) < min_area:
                continue
            room_type = getattr(face, 'merge_id', 'unknown')
            points = face.getPoints()
            corners = [(points[j].X, points[j].Y) for j in range(points.Count)]
            if len(corners) < 3:
                continue
            poly = Polygon(corners)
            x, y = poly.exterior.xy
            color = self.color_map.get(room_type, '#D3D3D3')
            ax.fill(x, y, color=color, edgecolor='black', linewidth=0.5, alpha=1)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_mesh_on_axis(self, mesh, ax, boundary_polygon=None,
                          title="Floor Plan", face_labels=None,
                          show_title=True, show_area=True, show_frame=True,
                          wall_inset=None, door_pairs=None,
                          entry_door_face_idx=None,
                          entry_door_segment=None):
        """See :func:`utils.visualization_helpers.plot_mesh_on_axis`."""
        from utils.visualization_helpers import plot_mesh_on_axis as _impl
        return _impl(
            self, mesh, ax,
            boundary_polygon=boundary_polygon,
            title=title,
            face_labels=face_labels,
            show_title=show_title,
            show_area=show_area,
            show_frame=show_frame,
            wall_inset=wall_inset,
            door_pairs=door_pairs,
            entry_door_face_idx=entry_door_face_idx,
            entry_door_segment=entry_door_segment,
        )

