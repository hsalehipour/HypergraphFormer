#!/usr/bin/env python3
"""
Convert RPLAN JSON files (from Housegan-data-reader) to NPZ format.

Replicates the preprocessing pipeline of rplanhg_datasets.py:
    reader() -> centering -> build_graph() -> contour extraction
    -> feature encoding -> mask construction -> NPZ

Produces three files required by RPlanhgDataset at eval time:
    rplan_eval_<target>.npz       -- ground-truth eval data
    rplan_train_<target>_cndist.npz -- corner-number distribution
    rplan_eval_<target>_syn.npz   -- synthesized conditioning

Minimal dependencies: numpy, opencv-python, Pillow, tqdm (no torch).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


_get_one_hot = lambda x, z: np.eye(z)[x]


# ---------------------------------------------------------------------------
# Helpers faithfully copied from rplanhg_datasets.py
# ---------------------------------------------------------------------------

def _reader(filepath):
    with open(filepath) as f:
        info = json.load(f)
    rms_bbs = np.asarray(info['boxes'])
    fp_eds = info['edges']
    rms_type = info['room_type']
    eds_to_rms = info['ed_rm']
    rms_bbs = np.array(rms_bbs) / 256.0
    fp_eds = np.array(fp_eds, dtype=float) / 256.0
    fp_eds = fp_eds[:, :4]
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    return rms_type, fp_eds, rms_bbs, eds_to_rms


def _make_sequence(edges):
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
    triples = []
    nodes = rms_type
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adj = any(True for e_map in eds_to_rms if (l in e_map) and (k in e_map))
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
# Main conversion
# ---------------------------------------------------------------------------

def _collect_cnumber_dist(json_dir, target_set=None):
    """Collect corner-number distribution from a JSON directory (typically train)."""
    json_dir = Path(json_dir)
    list_path = json_dir / 'list.txt'
    if list_path.exists():
        with open(list_path) as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        filenames = [f.name for f in sorted(json_dir.glob('*.json'))]

    print(f"[cndist] Scanning {len(filenames)} JSONs in {json_dir} ...")

    cnumber_dist = defaultdict(list)
    max_num_points = 100
    for fname in tqdm(filenames, desc="Building cndist"):
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        try:
            rms_type, fp_eds, rms_bbs, eds_to_rms = _reader(str(fpath))
        except Exception:
            continue
        fp_size = len([x for x in rms_type if x != 15 and x != 17])
        if target_set is not None and fp_size != target_set:
            continue
        try:
            graph_nodes, graph_edges, rooms_mks = _build_graph(rms_type, fp_eds, eds_to_rms)
        except Exception:
            continue
        total_corners = 0
        for room_mask, room_type in zip(rooms_mks, graph_nodes):
            room_mask = room_mask.astype(np.uint8)
            room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
            contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            n_corners = len(contours[0][:, 0, :])
            hd_type = {15: 11, 17: 12, 16: 13}.get(room_type, room_type) if room_type > 10 else room_type
            cnumber_dist[hd_type].append(n_corners)
            total_corners += n_corners
        if total_corners > max_num_points:
            continue

    print(f"[cndist] Collected distribution for {sum(len(v) for v in cnumber_dist.values())} rooms "
          f"across {len(cnumber_dist)} room types")
    return dict(cnumber_dist)


def convert(json_dir, output_dir, set_name='eval', target_set=None, train_json_dir=None):
    """Convert RPLAN JSON files to NPZ format.

    Args:
        train_json_dir: Directory of train-split JSONs. Used to build the
            corner-number distribution (cndist) required for synthesized
            conditioning at eval time.  If None and set_name=='eval', the
            cndist is (incorrectly) built from the eval data itself.
    """
    target_set_str = "all" if target_set is None else str(target_set)
    max_num_points = 100

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    list_path = json_dir / 'list.txt'
    if list_path.exists():
        with open(list_path) as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        filenames = [f.name for f in sorted(json_dir.glob('*.json'))]

    print(f"Found {len(filenames)} JSON files in {json_dir}")

    # ---- Phase 1: Read JSONs ------------------------------------------------
    subgraphs = []
    sample_ids = []
    skipped_ids = []
    for fname in tqdm(filenames, desc="Reading JSONs"):
        fpath = json_dir / fname
        sid = fname.replace('.json', '')
        if not fpath.exists():
            skipped_ids.append((sid, 'missing'))
            continue
        try:
            rms_type, fp_eds, rms_bbs, eds_to_rms = _reader(str(fpath))
        except Exception:
            skipped_ids.append((sid, 'read_error'))
            continue
        fp_size = len([x for x in rms_type if x != 15 and x != 17])
        if target_set is not None and fp_size != target_set:
            skipped_ids.append((sid, 'filtered'))
            continue
        subgraphs.append([rms_type, rms_bbs, fp_eds, eds_to_rms])
        sample_ids.append(sid)

    print(f"Loaded {len(subgraphs)} samples (after filtering)")

    # ---- Phase 2: Build graphs & extract contours ----------------------------
    org_houses = []
    org_graphs = []
    valid_ids = []
    for idx, graph_data in enumerate(tqdm(subgraphs, desc="Building graphs")):
        rms_type = graph_data[0]
        fp_eds = np.array(graph_data[2])
        eds_to_rms = graph_data[3]
        try:
            graph_nodes, graph_edges, rooms_mks = _build_graph(rms_type, fp_eds, eds_to_rms)
        except Exception:
            skipped_ids.append((sample_ids[idx], 'graph_error'))
            continue
        house = []
        for room_mask, room_type in zip(rooms_mks, graph_nodes):
            room_mask = room_mask.astype(np.uint8)
            room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
            contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                house.append([np.zeros((0, 2), dtype=int), room_type])
                continue
            house.append([contours[0][:, 0, :], room_type])
        org_houses.append(house)
        org_graphs.append(graph_edges)
        valid_ids.append(sample_ids[idx])

    # ---- Phase 3: Encode features & build masks -----------------------------
    houses = []
    graphs = []
    door_masks = []
    self_masks = []
    gen_masks = []
    final_ids = []
    cnumber_dist = defaultdict(list)
    kept_org_houses = []
    kept_org_graphs = []

    MAX_ROOMS = 32
    MAX_CORNERS = 32
    MAX_GRAPH_EDGES = 200
    for h, graph, sid in tqdm(zip(org_houses, org_graphs, valid_ids),
                              total=len(org_houses), desc="Encoding features"):
        if len(h) >= MAX_ROOMS:
            skipped_ids.append((sid, 'too_many_rooms'))
            continue
        if any(len(room[0]) >= MAX_CORNERS for room in h):
            skipped_ids.append((sid, 'too_many_corners'))
            continue
        if len(graph) > MAX_GRAPH_EDGES:
            skipped_ids.append((sid, 'too_many_graph_edges'))
            continue
        house = []
        corner_bounds = []
        num_points = 0
        for i, room in enumerate(h):
            if room[1] > 10:
                room[1] = {15: 11, 17: 12, 16: 13}[room[1]]
            room[0] = np.reshape(room[0], [len(room[0]), 2]) / 256.0 - 0.5
            room[0] = room[0] * 2
            cnumber_dist[room[1]].append(len(room[0]))
            num_room_corners = len(room[0])
            if num_room_corners < 1:
                continue
            rtype = np.repeat(np.array([_get_one_hot(room[1], 25)]), num_room_corners, 0)
            room_index = np.repeat(np.array([_get_one_hot(len(house) + 1, 32)]), num_room_corners, 0)
            corner_index = np.array([_get_one_hot(x, 32) for x in range(num_room_corners)])
            padding_mask = np.expand_dims(np.repeat(1, num_room_corners), 1)
            connections = np.array([[ci, (ci + 1) % num_room_corners] for ci in range(num_room_corners)])
            connections += num_points
            corner_bounds.append([num_points, num_points + num_room_corners])
            num_points += num_room_corners
            room_data = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
            house.append(room_data)

        if not house:
            skipped_ids.append((sid, 'empty_house'))
            continue
        house_layouts = np.concatenate(house, 0)
        if len(house_layouts) > max_num_points:
            skipped_ids.append((sid, 'too_many_points'))
            continue

        padding = np.zeros((max_num_points - len(house_layouts), 94))
        gen_mask = np.ones((max_num_points, max_num_points))
        gen_mask[:len(house_layouts), :len(house_layouts)] = 0
        house_layouts = np.concatenate((house_layouts, padding), 0)

        door_mask = np.ones((max_num_points, max_num_points))
        self_mask = np.ones((max_num_points, max_num_points))
        for i in range(len(corner_bounds)):
            for j in range(len(corner_bounds)):
                if i == j:
                    self_mask[corner_bounds[i][0]:corner_bounds[i][1],
                              corner_bounds[j][0]:corner_bounds[j][1]] = 0
                elif (any(np.equal([i, 1, j], graph).all(1))
                      or any(np.equal([j, 1, i], graph).all(1))):
                    door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                              corner_bounds[j][0]:corner_bounds[j][1]] = 0

        houses.append(house_layouts)
        graphs.append(graph)
        door_masks.append(door_mask)
        self_masks.append(self_mask)
        gen_masks.append(gen_mask)
        final_ids.append(sid)
        kept_org_houses.append(h)
        kept_org_graphs.append(graph)

    print(f"Successfully encoded {len(houses)} samples")

    # ---- Save eval NPZ -------------------------------------------------------
    houses_array = np.array(houses)
    graphs_array = np.empty(len(graphs), dtype=object)
    for i, g in enumerate(graphs):
        graphs_array[i] = g
    ids_array = np.array(final_ids, dtype=object)

    npz_path = output_dir / f'rplan_{set_name}_{target_set_str}.npz'
    np.savez_compressed(
        npz_path,
        houses=houses_array,
        graphs=graphs_array,
        door_masks=np.array(door_masks),
        self_masks=np.array(self_masks),
        gen_masks=np.array(gen_masks),
        ids=ids_array,
    )
    print(f"Saved {len(houses)} samples to {npz_path}")

    if skipped_ids:
        skip_path = output_dir / f'rplan_{set_name}_{target_set_str}_skipped.txt'
        with open(skip_path, 'w') as f:
            for sid, reason in skipped_ids:
                f.write(f"{sid}\t{reason}\n")
        print(f"Skipped {len(skipped_ids)} samples -> {skip_path}")

    # ---- Build / save cndist --------------------------------------------------
    # For eval: prefer cndist from training data if train_json_dir is provided.
    if set_name == 'eval' and train_json_dir is not None:
        train_cnumber_dist = _collect_cnumber_dist(train_json_dir, target_set)
    else:
        train_cnumber_dist = dict(cnumber_dist)

    cndist_path = output_dir / f'rplan_train_{target_set_str}_cndist.npz'
    np.savez_compressed(cndist_path, cnumber_dist=train_cnumber_dist)
    print(f"Saved corner distribution to {cndist_path}")

    # ---- For eval: build synthesized conditioning ----------------------------
    # Randomized corner counts, zeroed coordinates, same graph structure.
    # Matches rplanhg_datasets.py lines 258-325.
    if set_name == 'eval':
        syn_houses = []
        syn_graphs = []
        syn_door_masks = []
        syn_self_masks = []
        syn_gen_masks = []

        for h, graph in tqdm(zip(kept_org_houses, kept_org_graphs),
                             total=len(kept_org_houses), desc="Building syn data"):
            house = []
            corner_bounds = []
            num_points = 0
            num_room_corners_total = [
                train_cnumber_dist[room[1]][random.randint(0, len(train_cnumber_dist[room[1]]) - 1)]
                for room in h
            ]
            while np.sum(num_room_corners_total) >= max_num_points:
                num_room_corners_total = [
                    train_cnumber_dist[room[1]][random.randint(0, len(train_cnumber_dist[room[1]]) - 1)]
                    for room in h
                ]

            skip_sample = False
            for i, room in enumerate(h):
                num_room_corners = num_room_corners_total[i]
                if num_room_corners < 1 or num_room_corners >= MAX_CORNERS:
                    skip_sample = True
                    break
                rtype = np.repeat(np.array([_get_one_hot(room[1], 25)]), num_room_corners, 0)
                room_index = np.repeat(np.array([_get_one_hot(len(house) + 1, MAX_ROOMS)]), num_room_corners, 0)
                corner_index = np.array([_get_one_hot(x, MAX_CORNERS) for x in range(num_room_corners)])
                padding_mask = np.expand_dims(np.repeat(1, num_room_corners), 1)
                connections = np.array([[ci, (ci + 1) % num_room_corners] for ci in range(num_room_corners)])
                connections += num_points
                corner_bounds.append([num_points, num_points + num_room_corners])
                num_points += num_room_corners
                room_data = np.concatenate(
                    (np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections), 1
                )
                house.append(room_data)
            if skip_sample:
                continue

            house_layouts = np.concatenate(house, 0)
            if np.sum([len(room[0]) for room in h]) > max_num_points:
                continue
            padding = np.zeros((max_num_points - len(house_layouts), 94))
            gen_mask = np.ones((max_num_points, max_num_points))
            gen_mask[:len(house_layouts), :len(house_layouts)] = 0
            house_layouts = np.concatenate((house_layouts, padding), 0)

            door_mask = np.ones((max_num_points, max_num_points))
            self_mask = np.ones((max_num_points, max_num_points))

            living_room_index = 0
            for i, room in enumerate(h):
                if room[1] == 1:
                    living_room_index = i
                    break

            for i in range(len(corner_bounds)):
                is_connected = False
                for j in range(len(corner_bounds)):
                    if i == j:
                        self_mask[corner_bounds[i][0]:corner_bounds[i][1],
                                  corner_bounds[j][0]:corner_bounds[j][1]] = 0
                    elif (any(np.equal([i, 1, j], graph).all(1))
                          or any(np.equal([j, 1, i], graph).all(1))):
                        door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                                  corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        is_connected = True
                if not is_connected:
                    door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                              corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

            syn_houses.append(house_layouts)
            syn_graphs.append(graph)
            syn_door_masks.append(door_mask)
            syn_self_masks.append(self_mask)
            syn_gen_masks.append(gen_mask)

        syn_npz_path = output_dir / f'rplan_{set_name}_{target_set_str}_syn.npz'
        syn_graphs_array = np.empty(len(syn_graphs), dtype=object)
        for i, g in enumerate(syn_graphs):
            syn_graphs_array[i] = g
        np.savez_compressed(
            syn_npz_path,
            houses=np.array(syn_houses),
            graphs=syn_graphs_array,
            door_masks=np.array(syn_door_masks),
            self_masks=np.array(syn_self_masks),
            gen_masks=np.array(syn_gen_masks),
        )
        print(f"Saved {len(syn_houses)} syn samples to {syn_npz_path}")

    return str(npz_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RPLAN JSON files to NPZ format"
    )
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory containing {id}.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for NPZ files")
    parser.add_argument("--set_name", type=str, default="eval",
                        help="Dataset split name (default: eval)")
    parser.add_argument("--target_set", type=int, default=None,
                        help="Filter by room count (default: None = all)")
    parser.add_argument("--train_json_dir", type=str, default=None,
                        help="Directory of train-split JSONs (for cndist); "
                             "required for correct eval syn generation")
    args = parser.parse_args()

    convert(args.json_dir, args.output_dir,
            set_name=args.set_name, target_set=args.target_set,
            train_json_dir=args.train_json_dir)


if __name__ == "__main__":
    main()
