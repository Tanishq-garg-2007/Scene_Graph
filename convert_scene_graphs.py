"""
Preprocessed Scene Graph to Minimized SLM Representation Converter
==================================================================
Converts massive, dense scene graphs (~3,000 pairwise inferred edges, hash IDs)
into the optimized, minimal representation for Small Language Models:
  1. Semantic IDs: Canonical, readable IDs (e.g., Man_1, Van_1, Clock_1).
  2. Relational Edges: Explicit interactions (wearing, behind, on) with target attributes.
  3. Minimal Spatial Relationships: Non-redundant 1D topological chains
     (transitive reduction) for left-to-right ordering and vertical elevation.

Usage:
  python convert_scene_graphs.py --input preprocessed_graphs-20261022T004957Z-1-001.zip --output converted_graphs/
  python convert_scene_graphs.py --input path/to/1.json --output 1_converted.json
"""

import os
import sys
import json
import zipfile
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple

# Surface & terrain labels that span ambient visual backgrounds rather than discrete objects
AMBIENT_LABELS: Set[str] = {
    "street", "road", "sidewalk", "pavement", "ground", "sky", "shade", 
    "floor", "grass", "wall", "ceiling", "carpet", "dirt"
}

# Clothing, accessories, and wearable items attached to an agent
WEARABLE_LABELS: Set[str] = {
    "shirt", "pants", "jacket", "shoes", "sneakers", "glasses", "hat", 
    "cap", "watch", "tie", "coat", "hoodie", "jeans", "shorts", "boots", 
    "sunglasses", "belt", "sweater", "vest", "t-shirt", "tshirt", "glove", "gloves"
}

# Secondary body parts that add visual noise rather than semantic reasoning value
BODY_PART_LABELS: Set[str] = {
    "chin", "back", "arm", "hand", "leg", "foot", "feet", "head", 
    "face", "hair", "eye", "eyes", "nose", "mouth", "ear", "ears"
}

# Known Visual Genome synonym clusters for alias resolution
SYNONYM_GROUPS: List[Set[str]] = [
    {"man", "guy", "boy"},
    {"woman", "girl", "lady"},
    {"car", "automobile", "auto"}
]

def are_synonyms(label1: str, label2: str) -> bool:
    l1, l2 = label1.lower(), label2.lower()
    if l1 == l2:
        return True
    for grp in SYNONYM_GROUPS:
        if l1 in grp and l2 in grp:
            return True
    return False

def compute_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    if not box1 or not box2 or "x" not in box1 or "x" not in box2:
        return 0.0
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1.get("w", 0), box2["x"] + box2.get("w", 0))
    y2 = min(box1["y"] + box1.get("h", 0), box2["y"] + box2.get("h", 0))
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = box1.get("w", 0) * box1.get("h", 0)
    area2 = box2.get("w", 0) * box2.get("h", 0)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def convert_scene_graph(sg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a single raw preprocessed scene graph into the minimal SLM format.
    Returns a dictionary containing structured components and prompt-ready text.
    """
    objects = sg_data.get("objects", [])
    annotated_rels = sg_data.get("annotated_relationships", [])
    
    # -------------------------------------------------------------
    # 1. Alias Resolution (Only merge synonyms with strong visual overlap)
    # -------------------------------------------------------------
    alias_map: Dict[Any, Any] = {}
    kept_objects: List[Dict[str, Any]] = []

    def label_priority(o):
        lbl = o.get("label", "").lower()
        if lbl in {"man", "woman", "person"}:
            return 2
        return 1

    sorted_for_dedup = sorted(objects, key=label_priority, reverse=True)

    for o in sorted_for_dedup:
        o_id = o["object_id"]
        o_lbl = o.get("label", "")
        o_bbox = o.get("bbox") or o.get("bounding_box") or {}
        
        is_syn_dup = False
        for kept in kept_objects:
            k_lbl = kept.get("label", "")
            k_bbox = kept.get("bbox") or kept.get("bounding_box") or {}
            if are_synonyms(o_lbl, k_lbl):
                iou = compute_iou(o_bbox, k_bbox)
                center_dist = (
                    (o["center"]["x"] - kept["center"]["x"])**2 + 
                    (o["center"]["y"] - kept["center"]["y"])**2
                )**0.5
                if iou > 0.65 or center_dist < 10:
                    alias_map[o_id] = kept["object_id"]
                    is_syn_dup = True
                    break
        if not is_syn_dup:
            kept_objects.append(o)
            alias_map[o_id] = o_id

    # -------------------------------------------------------------
    # 2. Semantic ID Assignment
    # -------------------------------------------------------------
    label_counts: Dict[str, int] = defaultdict(int)
    id_to_semantic: Dict[Any, str] = {}
    id_to_obj: Dict[Any, Dict[str, Any]] = {}

    sorted_kept = sorted(kept_objects, key=lambda o: (o.get("label", ""), o["center"]["x"]))

    for obj in sorted_kept:
        raw_id = obj["object_id"]
        label = obj.get("label", "").strip()
        clean_label = "".join(word.capitalize() for word in label.split())
        label_counts[clean_label] += 1
        semantic_id = f"{clean_label}_{label_counts[clean_label]}"
        id_to_semantic[raw_id] = semantic_id
        id_to_obj[raw_id] = obj

    def format_node_with_attrs(raw_id: Any) -> str:
        canon_id = alias_map.get(raw_id, raw_id)
        obj = id_to_obj.get(canon_id)
        if not obj:
            return f"[{id_to_semantic.get(canon_id, str(canon_id))}]"
        sem_id = id_to_semantic[canon_id]
        attrs = [a for a in obj.get("attributes", []) if a]
        if attrs:
            seen = set()
            clean_attrs = [a for a in attrs if not (a in seen or seen.add(a))]
            return f"[{sem_id}: {', '.join(clean_attrs)}]"
        return f"[{sem_id}]"

    # -------------------------------------------------------------
    # 3. Relational Edges Extraction
    # -------------------------------------------------------------
    wear_predicates = {"wears", "wearing", "has", "in"}
    relational_edges: List[Tuple[Any, str, Any]] = []
    attached_child_ids: Set[Any] = set()
    seen_rel_signatures: Set[Tuple[Any, str, Any]] = set()

    for r in annotated_rels:
        raw_sub = r.get("subject_id")
        raw_obj = r.get("object_id")
        sub_id = alias_map.get(raw_sub, raw_sub)
        obj_id = alias_map.get(raw_obj, raw_obj)
        
        if sub_id == obj_id or sub_id not in id_to_semantic or obj_id not in id_to_semantic:
            continue

        pred = r.get("predicate", "").strip().lower()
        sub_label = id_to_obj[sub_id].get("label", "").lower()
        obj_label = id_to_obj[obj_id].get("label", "").lower()

        # Filter out minor body parts
        if obj_label in BODY_PART_LABELS or sub_label in BODY_PART_LABELS:
            continue

        # Canonicalize wearing relations
        if (pred in wear_predicates or pred == "wearing") and (
            obj_label in WEARABLE_LABELS or sub_label in {"man", "woman", "person", "boy", "girl"}
        ):
            norm_pred = "wearing"
            sig = (sub_id, norm_pred, obj_id)
            if sig not in seen_rel_signatures:
                seen_rel_signatures.add(sig)
                relational_edges.append((sub_id, norm_pred, obj_id))
                attached_child_ids.add(obj_id)
        elif pred in {"on", "on top of", "along", "behind", "next to", "near", "holding", "parked on", "attached to"}:
            norm_pred = "on" if pred in {"on", "on top of", "parked on"} else pred
            sig = (sub_id, norm_pred, obj_id)
            # Avoid redundant reverse edges (e.g., Shirt on Man when Man wearing Shirt is present)
            rev_wear_sig = (obj_id, "wearing", sub_id)
            if sig not in seen_rel_signatures and rev_wear_sig not in seen_rel_signatures:
                seen_rel_signatures.add(sig)
                relational_edges.append((sub_id, norm_pred, obj_id))

    rel_lines = []
    for sub_id, pred, obj_id in relational_edges:
        sub_node = f"[{id_to_semantic[sub_id]}]"
        obj_node = format_node_with_attrs(obj_id)
        rel_lines.append(f"- {sub_node} --[{pred}]--> {obj_node}")

    # -------------------------------------------------------------
    # 4. Minimal Spatial Relationships (Linear Topological Chain)
    # -------------------------------------------------------------
    # Exclude wearables, body parts, ambient surfaces, and minor attachment features
    spatial_anchors = []
    ignored_spatial_labels = WEARABLE_LABELS | BODY_PART_LABELS | AMBIENT_LABELS | {"headlight", "back", "windows", "shade"}

    for obj in kept_objects:
        raw_id = obj["object_id"]
        label = obj.get("label", "").strip().lower()
        if raw_id in attached_child_ids or label in ignored_spatial_labels:
            continue
        spatial_anchors.append(obj)

    # Sort left-to-right by horizontal center
    sorted_horizontal = sorted(spatial_anchors, key=lambda o: o["center"]["x"])
    horiz_nodes = [format_node_with_attrs(o["object_id"]) for o in sorted_horizontal]
    horiz_chain_str = " -> ".join(horiz_nodes) if horiz_nodes else "None"

    # Vertical Elevation (Top-to-Bottom / Highest-to-Lowest: smaller y = higher elevation)
    sorted_vertical = sorted(spatial_anchors, key=lambda o: o["center"]["y"])
    vert_nodes = [format_node_with_attrs(o["object_id"]) for o in sorted_vertical]
    vert_chain_str = " -> ".join(vert_nodes) if vert_nodes else "None"

    # -------------------------------------------------------------
    # 5. Format Text Output
    # -------------------------------------------------------------
    prompt_text = "### Relational Edges:\n"
    prompt_text += "\n".join(rel_lines) if rel_lines else "None"
    prompt_text += "\n\n### Spatial Ordering Chains:\n"
    prompt_text += f"- Horizontal Ordering (Left-to-Right Chain):\n  {horiz_chain_str}\n\n"
    prompt_text += f"- Vertical Ordering (Top-to-Bottom / Highest-to-Lowest Chain):\n  {vert_chain_str}"

    return {
        "image_id": sg_data.get("image_id"),
        "prompt_text": prompt_text,
        "semantic_id_mapping": id_to_semantic,
        "relational_edges": [
            {"subject": id_to_semantic[s], "predicate": p, "object": id_to_semantic[o]}
            for s, p, o in relational_edges
        ],
        "horizontal_ordering": [
            id_to_semantic[o["object_id"]] for o in sorted_horizontal
        ]
    }


def process_zip(zip_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = [f for f in z.namelist() if f.endswith(".json") and not f.startswith("__MACOSX")]
        print(f"Found {len(file_list)} scene graphs in {zip_path}...")
        
        for idx, filename in enumerate(file_list, 1):
            try:
                raw_data = json.loads(z.read(filename).decode("utf-8"))
                converted = convert_scene_graph(raw_data)
                
                base_name = os.path.basename(filename)
                out_path = os.path.join(output_dir, base_name)
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(converted, out_f, indent=2)
                
                if idx % 10 == 0 or idx == len(file_list):
                    print(f"[{idx}/{len(file_list)}] Converted {base_name}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

def process_file(json_path: str, output_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    converted = convert_scene_graph(raw_data)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(converted, out_f, indent=2)
    print(f"Saved converted scene graph to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert massive scene graphs into minimal SLM representation.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input json file, directory, or zip archive.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output json file or directory.")
    args = parser.parse_args()

    if args.input.endswith(".zip"):
        process_zip(args.input, args.output)
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.endswith(".json"):
                process_file(os.path.join(args.input, fname), os.path.join(args.output, fname))
    else:
        process_file(args.input, args.output)

if __name__ == "__main__":
    main()
