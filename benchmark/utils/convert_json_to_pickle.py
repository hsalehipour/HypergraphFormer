#!/usr/bin/env python3
"""
Convert House-Diffusion JSON files to HouseGAN++ pickle format.

Source: per-sample JSON files produced by raster_to_json / batch_raster_to_json,
        each containing {room_type, boxes, edges, ed_rm}.

Target: a single pickle file (list of dicts) loadable by
        benchmark.src.dataset_loader.Hypergraph1kDataset with keys:
            id       : str   -- sample identifier (filename stem)
            polygons : list[list[tuple[int,int]]]  -- one polygon per node,
                       coordinates in [0, target_size] (default 64)
            room_types: list[int] -- RPLAN room type IDs (1-18, doors=17)
            triples  : list[list[int]] -- [k, rel, l] for every pair k<l,
                       rel in {1, -1}

The geometry pipeline mirrors convert_json_to_npz._reader() and _build_graph()
so polygons and adjacency are consistent with the House-Diffusion preprocessing.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reader -- identical to convert_json_to_npz._reader()
# ---------------------------------------------------------------------------

def _reader(filepath):
    """Read a House-Diffusion JSON and return centred arrays."""
    with open(filepath) as f:
        info = json.load(f)
    rms_bbs = np.asarray(info['boxes'])
    fp_eds = np.array(info['edges'], dtype=float)
    rms_type = info['room_type']
    eds_to_rms = info['ed_rm']

    rms_bbs = rms_bbs / 256.0
    fp_eds = fp_eds / 256.0
    fp_eds = fp_eds[:, :4]

    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    return rms_type, fp_eds, rms_bbs, eds_to_rms


# ---------------------------------------------------------------------------
# Sequence / polygon helpers -- identical to convert_json_to_npz
# ---------------------------------------------------------------------------

def _make_sequence(edges):
    """Chain edges into closed polygons (vertex sequences)."""
    polys = []
    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False
    while len(e_visited) < len(edges):
        if not find_next:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = not find_next
        else:
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if v_curr == tuple(e[:2]):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif v_curr == tuple(e[2:]):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(edges[0][:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    polys.append(seq_tracker)
    return polys


def _build_graph(rms_type, fp_eds, eds_to_rms, out_size=64):
    """
    Build adjacency graph and rasterised room masks.

    Returns (nodes, triples, room_masks) where:
        nodes      : np.array of room type IDs
        triples    : np.array  [[k, 1/-1, l], ...] for all k<l
        room_masks : np.array  [num_nodes, out_size, out_size]
    """
    nodes = rms_type
    triples = []
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adj = any(
                    True for e_map in eds_to_rms
                    if (l in e_map) and (k in e_map)
                )
                triples.append([k, 1 if is_adj else -1, l])

    eds_to_rms_tmp = [[e[0]] for e in eds_to_rms]
    rms_masks = []
    im_size = 256
    fp_mk = np.zeros((out_size, out_size))

    for k in range(len(nodes)):
        eds = [l for l, e_map in enumerate(eds_to_rms_tmp) if k in e_map]
        rm_im = Image.new('L', (im_size, im_size))
        dr = ImageDraw.Draw(rm_im)
        poly = _make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
        poly = [(im_size * x, im_size * y) for x, y in poly]
        if len(poly) >= 2:
            dr.polygon(poly, fill='white')
        else:
            continue
        rm_im = rm_im.resize((out_size, out_size))
        rm_arr = np.array(rm_im)
        inds = np.where(rm_arr > 0)
        rm_arr[inds] = 1.0
        rms_masks.append(rm_arr)
        if rms_type[k] != 15 and rms_type[k] != 17:
            fp_mk[inds] = k + 1

    for k in range(len(nodes)):
        if rms_type[k] != 15 and rms_type[k] != 17:
            rm_arr = np.zeros((out_size, out_size))
            inds = np.where(fp_mk == k + 1)
            rm_arr[inds] = 1.0
            rms_masks[k] = rm_arr

    return np.array(nodes), np.array(triples), np.array(rms_masks)


# ---------------------------------------------------------------------------
# Mask → polygon extraction (contour → vertex list in [0, target_size])
# ---------------------------------------------------------------------------

def _mask_to_polygon(mask, target_size=64):
    """
    Extract a polygon (list of (x, y) tuples) from a binary mask.

    Pipeline: mask → uint8 → findContours → scale to target_size.
    Returns None if no valid contour is found.
    """
    import cv2 as cv

    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    contours, _ = cv.findContours(
        (m * 255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    pts = contour[:, 0, :]  # (N, 2)
    if len(pts) < 3:
        return None
    scale_x = target_size / w
    scale_y = target_size / h
    poly = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in pts]
    return poly


# ---------------------------------------------------------------------------
# Single-sample conversion
# ---------------------------------------------------------------------------

def _convert_one(json_path, target_size=64):
    """
    Convert a single House-Diffusion JSON to a HouseGAN++ pickle sample dict.

    Returns dict {id, polygons, room_types, triples} or None on failure.
    """
    name = Path(json_path).stem
    try:
        rms_type, fp_eds, rms_bbs, eds_to_rms = _reader(str(json_path))
    except Exception:
        return None

    try:
        graph_nodes, graph_triples, rooms_mks = _build_graph(
            rms_type, fp_eds, eds_to_rms, out_size=target_size
        )
    except Exception:
        return None

    if len(rooms_mks) != len(graph_nodes):
        return None

    polygons = []
    for mask in rooms_mks:
        poly = _mask_to_polygon(mask, target_size=target_size)
        if poly is None:
            return None
        polygons.append(poly)

    return {
        'id': name,
        'polygons': polygons,
        'room_types': graph_nodes.tolist(),
        'triples': graph_triples.tolist(),
    }


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert(json_dir, output_path, target_set=None):
    """
    Convert all JSONs in *json_dir* to a single HouseGAN++ pickle file.

    Args:
        json_dir:    Directory containing {id}.json files (and optionally list.txt).
        output_path: Output .pkl path.
        target_set:  If not None, keep only samples with this many non-door rooms.
    """
    json_dir = Path(json_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_path = json_dir / 'list.txt'
    if list_path.exists():
        with open(list_path) as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        filenames = sorted(p.name for p in json_dir.glob('*.json'))

    print(f"Found {len(filenames)} JSON files in {json_dir}")

    samples = []
    skipped = defaultdict(int)

    for fname in tqdm(filenames, desc="Converting", unit="sample"):
        fpath = json_dir / fname
        if not fpath.exists():
            skipped['missing'] += 1
            continue

        result = _convert_one(fpath)
        if result is None:
            skipped['error'] += 1
            continue

        if target_set is not None:
            num_rooms = sum(1 for rt in result['room_types'] if rt != 17)
            if num_rooms != target_set:
                skipped['filtered'] += 1
                continue

        samples.append(result)

    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"Saved {len(samples)} samples to {output_path}")
    if skipped:
        parts = [f"{k}={v}" for k, v in skipped.items()]
        print(f"  Skipped: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert House-Diffusion JSON files to HouseGAN++ pickle format",
    )
    parser.add_argument(
        "--json_dir", type=str, required=True,
        help="Directory containing {id}.json files (e.g. dataset_housediffusion/test)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output pickle path (e.g. Hypergraph_to_RPLAN/test/test_data.pkl)",
    )
    parser.add_argument(
        "--target_set", type=int, default=None,
        help="Keep only samples with this many non-door rooms (default: all)",
    )
    parser.add_argument(
        "--target_size", type=int, default=64,
        help="Polygon coordinate space (default: 64, matching HouseGAN++)",
    )
    args = parser.parse_args()

    convert(args.json_dir, args.output, target_set=args.target_set)


if __name__ == "__main__":
    main()
