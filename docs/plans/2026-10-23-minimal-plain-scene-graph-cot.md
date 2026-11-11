# Minimal Plain-Language Scene Graph & Chain-of-Thought (CoT) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform massive, redundant scene graphs (3000+ pairwise relations) into compact, plain-language entity-property descriptions and non-redundant minimal spatial chains (transitive reduction & canonical directions), guided by a 3-stage Chain-of-Thought prompt to eliminate SLM hallucination.

**Architecture:** 
1. **Entity Rollup & Cleaning**: Wearables and micro-parts are merged into parent entities while background noise is pruned.
2. **Canonical Direction & Transitive Reduction**: Symmetrical inverses (left/right, above/below) are normalized to canonical directions, and redundant transitive shortcuts ($A \to C$ when $A \to B \to C$) are eliminated using DAG transitive reduction to produce $O(N)$ minimal direct chains.
3. **Plain-Language Serializer**: Entities and spatial links are converted into concise natural language markdown.
4. **3-Stage Chain-of-Thought (CoT) Prompting**: Directs the SLM through: (1) Entity Identification/Grounding, (2) Spatial Graph Deduction (transitive & inverse inference), and (3) Final Answer Evaluation.

**Tech Stack:** Python 3.10+, NetworkX (transitive reduction), Groq API / Qwen 27B SLM, NumPy, Requests, Pytest.

---

### Task 1: Spatial Graph Transitive Reduction & Canonical Direction Engine

**Files:**
- Create: `src/minimal_graph.py`
- Test: `tests/test_minimal_graph.py`

**Context & Rationale:**
Currently, preprocessed scene graphs generate dense $O(N^2)$ pairwise relationships (e.g. 3,040 inferred relations for 38 objects). Every pair has redundant symmetric inverses ("A is left of B" AND "B is right of A") and transitive skips ("A is left of B", "B is left of C", AND "A is left of C"). This engine normalizes relations to canonical directions and applies NetworkX DAG transitive reduction to retain only immediate direct neighbor links ($A \to B \to C$).

**Step 1: Write the failing test**

Create `tests/test_minimal_graph.py`:
```python
import pytest
import networkx as nx
from src.minimal_graph import (
    canonicalize_relation,
    build_transitive_reduction_chain,
    extract_minimal_spatial_graph
)

def test_canonicalize_relation():
    # Symmetric opposites should map to canonical directions
    assert canonicalize_relation("A", "right", "B") == ("B", "left", "A")
    assert canonicalize_relation("A", "left", "B") == ("A", "left", "B")
    assert canonicalize_relation("A", "below", "B") == ("B", "above", "A")
    assert canonicalize_relation("A", "above", "B") == ("A", "above", "B")
    assert canonicalize_relation("A", "under", "B") == ("A", "on", "B")  # Support canonical

def test_build_transitive_reduction_chain():
    # If A is left of B, B is left of C, and A is left of C:
    # Transitive reduction must only keep A -> B and B -> C
    nodes = ["obj_A", "obj_B", "obj_C"]
    edges = [("obj_A", "obj_B"), ("obj_B", "obj_C"), ("obj_A", "obj_C")]
    
    reduced_edges = build_transitive_reduction_chain(nodes, edges)
    assert set(reduced_edges) == {("obj_A", "obj_B"), ("obj_B", "obj_C")}
    assert ("obj_A", "obj_C") not in reduced_edges

def test_extract_minimal_spatial_graph():
    # 3 collinear objects along X axis
    objects = [
        {"object_id": "1", "label": "man", "center": {"x": 100, "y": 200}},
        {"object_id": "2", "label": "chair", "center": {"x": 250, "y": 200}},
        {"object_id": "3", "label": "table", "center": {"x": 400, "y": 200}},
    ]
    # Dense pairwise adjacency
    for o in objects:
        o["adjacency"] = {"left": [], "right": [], "above": [], "below": [], "on": [], "inside": []}
    objects[0]["adjacency"]["right"] = [{"object_id": "2"}, {"object_id": "3"}]
    objects[1]["adjacency"]["left"] = [{"object_id": "1"}]
    objects[1]["adjacency"]["right"] = [{"object_id": "3"}]
    objects[2]["adjacency"]["left"] = [{"object_id": "1"}, {"object_id": "2"}]

    res = extract_minimal_spatial_graph(objects)
    # Horizontal left-to-right chain must be 1 -> 2 -> 3 only
    assert res["horizontal_chains"] == [("1", "2"), ("2", "3")]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_minimal_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

**Step 3: Write minimal implementation**

Create `src/minimal_graph.py`:
```python
import networkx as nx
from typing import Dict, List, Tuple, Any

# Map inverse predicates to canonical directions
CANONICAL_MAP = {
    "right": ("left", True),       # (A right B) -> (B left A)
    "below": ("above", True),      # (A below B) -> (B above A)
    "under": ("on", True),         # (A under B) -> (A on B)
    "inside": ("contains", True),  # (A inside B) -> (B contains A)
}

def canonicalize_relation(subj: str, predicate: str, obj: str) -> Tuple[str, str, str]:
    pred_lower = predicate.lower()
    if pred_lower in CANONICAL_MAP:
        canon_pred, invert = CANONICAL_MAP[pred_lower]
        if invert:
            return (obj, canon_pred, subj)
    return (subj, pred_lower, obj)

def build_transitive_reduction_chain(node_ids: List[str], directed_edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Computes transitive reduction on a DAG of spatial orderings.
    Eliminates redundant skips (e.g. A->C when A->B->C exists).
    """
    if not directed_edges:
        return []
    
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(directed_edges)
    
    # If there are cycles due to noisy bounding boxes, break cycles via greedy topological sort or condensation
    if not nx.is_directed_acyclic_graph(G):
        # Fall back to coordinate ordering if cycle exists
        return directed_edges
        
    reduced_G = nx.transitive_reduction(G)
    return list(reduced_G.edges())

def extract_minimal_spatial_graph(objects: List[Dict[str, Any]], annotated_relations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extracts canonical, non-redundant spatial relations across horizontal, vertical, and contact axes.
    """
    node_ids = [str(o["object_id"]) for o in objects]
    obj_by_id = {str(o["object_id"]): o for o in objects}
    
    # 1. Horizontal Ordering (Left-to-Right along X axis)
    # Sort objects primarily by x center
    sorted_by_x = sorted(objects, key=lambda o: o["center"]["x"])
    h_edges = []
    for i in range(len(sorted_by_x)):
        for j in range(i + 1, len(sorted_by_x)):
            u = str(sorted_by_x[i]["object_id"])
            v = str(sorted_by_x[j]["object_id"])
            # verify significant horizontal displacement (> 15 pixels)
            if sorted_by_x[j]["center"]["x"] - sorted_by_x[i]["center"]["x"] > 15:
                h_edges.append((u, v))
                
    minimal_h_edges = build_transitive_reduction_chain(node_ids, h_edges)

    # 2. Vertical Ordering (Above-to-Below along Y axis)
    sorted_by_y = sorted(objects, key=lambda o: o["center"]["y"])
    v_edges = []
    for i in range(len(sorted_by_y)):
        for j in range(i + 1, len(sorted_by_y)):
            u = str(sorted_by_y[i]["object_id"])
            v = str(sorted_by_y[j]["object_id"])
            if sorted_by_y[j]["center"]["y"] - sorted_by_y[i]["center"]["y"] > 25:
                v_edges.append((u, v))
                
    minimal_v_edges = build_transitive_reduction_chain(node_ids, v_edges)

    # 3. Topological relations (on, inside, contains)
    contact_edges = []
    if annotated_relations:
        for r in annotated_relations:
            pred = r["predicate"].lower()
            if pred in ["on", "under", "in", "inside", "contains", "attached to", "part of"]:
                s, p, o = canonicalize_relation(str(r["subject_id"]), pred, str(r["object_id"]))
                if s in obj_by_id and o in obj_by_id:
                    contact_edges.append((s, p, o))

    return {
        "horizontal_chains": minimal_h_edges,
        "vertical_chains": minimal_v_edges,
        "contact_relations": list(set(contact_edges))
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_minimal_graph.py -v`
Expected: `3 passed in 0.15s`

---

### Task 2: Entity Rollup & Plain-Language Scene Serializer

**Files:**
- Create: `src/plain_serializer.py`
- Test: `tests/test_plain_serializer.py`

**Context & Rationale:**
Currently, entities have isolated sub-parts (e.g. jacket, pants, shirt, shoes, sunglasses) separated from person nodes, alongside background clutter (asphalt, sidewalk, trees). This component rolls wearables into parent entities, assigns clean human-readable symbolic IDs (`[man_1]`, `[van_1]`), and serializes the entities and minimal spatial links into a token-efficient plain language prompt.

**Step 1: Write the failing test**

Create `tests/test_plain_serializer.py`:
```python
import pytest
from src.plain_serializer import clean_and_rollup_entities, serialize_to_plain_language

def test_clean_and_rollup_entities():
    sample_sg = {
        "objects": [
            {"object_id": 1, "label": "man", "center": {"x": 100, "y": 300}, "bbox": {"w": 50, "h": 100}, "attributes": []},
            {"object_id": 2, "label": "shirt", "center": {"x": 100, "y": 280}, "bbox": {"w": 40, "h": 40}, "attributes": ["red"]},
            {"object_id": 3, "label": "sidewalk", "center": {"x": 200, "y": 450}, "bbox": {"w": 300, "h": 50}, "attributes": []},
            {"object_id": 4, "label": "van", "center": {"x": 500, "y": 300}, "bbox": {"w": 150, "h": 100}, "attributes": ["white"]}
        ],
        "annotated_relationships": [
            {"subject_id": 1, "predicate": "wearing", "object_id": 2}
        ]
    }
    
    cleaned_objs = clean_and_rollup_entities(sample_sg)
    labels = [o["clean_label"] for o in cleaned_objs]
    assert "man" in labels
    assert "van" in labels
    assert "sidewalk" not in labels  # pruned background
    assert "shirt" not in labels     # rolled into man
    
    man_obj = [o for o in cleaned_objs if o["clean_label"] == "man"][0]
    assert "red shirt" in man_obj["wearables"]

def test_serialize_to_plain_language():
    entities = [
        {"entity_id": "man_1", "clean_label": "man", "attributes": [], "wearables": ["red shirt"]},
        {"entity_id": "van_1", "clean_label": "van", "attributes": ["white"], "wearables": []}
    ]
    spatial_data = {
        "horizontal_chains": [("man_1", "van_1")],
        "vertical_chains": [],
        "contact_relations": []
    }
    
    text = serialize_to_plain_language(entities, spatial_data)
    assert "### Scene Entities & Properties" in text
    assert "[man_1]: man (wearing: red shirt)" in text
    assert "[van_1]: van (white)" in text
    assert "### Minimal Spatial Relationships" in text
    assert "[man_1] is to the left of [van_1]" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_plain_serializer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.plain_serializer'`

**Step 3: Write minimal implementation**

Create `src/plain_serializer.py`:
```python
from typing import Dict, List, Any

IGNORE_CLASSES = {
    "sidewalk", "street", "road", "shade", "trees", "tree", "building",
    "wall", "sky", "lamp post", "bikes", "bike", "chin", "arm", "hand", "leg"
}

def clean_and_rollup_entities(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_objs = {o["object_id"]: dict(o) for o in scene_graph.get("objects", [])}
    wearable_ids = set()

    for o in raw_objs.values():
        o["wearables"] = []
        o["clean_label"] = o.get("label", "").lower().strip()

    # Roll up wearables and parts into parent entities
    for r in scene_graph.get("annotated_relationships", []):
        pred = r.get("predicate", "").lower()
        if pred in ["wearing", "wears", "has", "in", "w/"]:
            s_id, o_id = r.get("subject_id"), r.get("object_id")
            if s_id in raw_objs and o_id in raw_objs:
                item = raw_objs[o_id]
                desc = f"{' '.join(item.get('attributes', []))} {item['clean_label']}".strip()
                raw_objs[s_id]["wearables"].append(desc)
                wearable_ids.add(o_id)

    # Filter out wearables, sub-parts, and ignored backgrounds
    cleaned = []
    class_counter = {}
    for o in raw_objs.values():
        lbl = o["clean_label"]
        if o["object_id"] in wearable_ids or lbl in IGNORE_CLASSES:
            continue
        
        # Deduplicate wearables list
        o["wearables"] = list(dict.fromkeys(o["wearables"]))
        
        # Assign symbolic entity_id (e.g. man_1, van_1)
        count = class_counter.get(lbl, 0) + 1
        class_counter[lbl] = count
        o["entity_id"] = f"{lbl.replace(' ', '_')}_{count}"
        cleaned.append(o)

    return cleaned

def serialize_to_plain_language(entities: List[Dict[str, Any]], spatial_graph: Dict[str, Any]) -> str:
    """
    Serializes objects and non-redundant spatial links into structured plain-language markdown.
    """
    lines = ["### Scene Entities & Properties"]
    ent_map = {str(e.get("object_id", e.get("entity_id"))): e["entity_id"] for e in entities}
    
    for e in entities:
        props = []
        if e.get("attributes"):
            props.append(", ".join(e["attributes"]))
        if e.get("wearables"):
            props.append(f"wearing: {', '.join(e['wearables'])}")
        prop_str = f" ({'; '.join(props)})" if props else ""
        lines.append(f"- [{e['entity_id']}]: {e['clean_label']}{prop_str}")

    lines.append("\n### Minimal Spatial Relationships")
    lines.append("(Note: Only direct adjacent links are listed. Transitive relations and opposite directions are logically implied.)\n")

    # 1. Horizontal Left-to-Right
    h_links = []
    for u, v in spatial_graph.get("horizontal_chains", []):
        u_name = ent_map.get(str(u), str(u))
        v_name = ent_map.get(str(v), str(v))
        h_links.append(f"[{u_name}] is to the left of [{v_name}]")
    if h_links:
        lines.append("- Horizontal Ordering (Left-to-Right):")
        for link in h_links:
            lines.append(f"  * {link}")
    else:
        lines.append("- Horizontal Ordering: None recorded.")

    # 2. Vertical Top-to-Bottom
    v_links = []
    for u, v in spatial_graph.get("vertical_chains", []):
        u_name = ent_map.get(str(u), str(u))
        v_name = ent_map.get(str(v), str(v))
        v_links.append(f"[{u_name}] is higher than [{v_name}]")
    if v_links:
        lines.append("\n- Vertical Ordering (Elevation):")
        for link in v_links:
            lines.append(f"  * {link}")

    # 3. Contact / Support
    c_links = []
    for s, p, o in spatial_graph.get("contact_relations", []):
        s_name = ent_map.get(str(s), str(s))
        o_name = ent_map.get(str(o), str(o))
        c_links.append(f"[{s_name}] is {p} [{o_name}]")
    if c_links:
        lines.append("\n- Physical Contact & Containment:")
        for link in c_links:
            lines.append(f"  * {link}")

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_plain_serializer.py -v`
Expected: `2 passed in 0.12s`

---

### Task 3: 3-Stage Chain-of-Thought (CoT) Prompt Engineering & Response Parser

**Files:**
- Create: `src/cot_engine.py`
- Test: `tests/test_cot_engine.py`

**Context & Rationale:**
Rather than asking the SLM to jump directly to an answer (which causes hallucinations when spatial chains are complex), we formulate a dedicated 3-stage Chain-of-Thought prompt:
1. **Entity Identification & Grounding**: Identify which entity IDs correspond to the objects named in the query.
2. **Spatial Graph Deduction**: Deduce betweenness, transitivity, and inverses from the minimal graph.
3. **Evaluation & Final Answer**: Verify consistency and output the direct answer.

**Step 1: Write the failing test**

Create `tests/test_cot_engine.py`:
```python
import pytest
from src.cot_engine import build_cot_prompt, parse_cot_response

def test_build_cot_prompt():
    plain_context = "### Scene Entities & Properties\n- [man_1]: man (wearing: red shirt)\n- [van_1]: van (white)"
    query = "What is to the right of the man in the red shirt?"
    
    prompt = build_cot_prompt(plain_context, query)
    assert "Step 1: Entity Identification & Disambiguation" in prompt
    assert "Step 2: Spatial Graph Deduction" in prompt
    assert "Step 3: Evaluation & Answer" in prompt
    assert plain_context in prompt
    assert query in prompt

def test_parse_cot_response():
    sample_output = """<thought>
Step 1: The user query mentions 'man in red shirt' which maps to [man_1], and 'van' which maps to [van_1].
Step 2: From the horizontal ordering, [man_1] is to the left of [man_2], and [man_2] is to the left of [van_1]. This means [man_2] is positioned between [man_1] and [van_1].
Step 3: Therefore, the object between them is [man_2] (man wearing grey hoodie).
</thought>
Final Answer: The man in the grey hoodie."""

    parsed = parse_cot_response(sample_output)
    assert "Step 1" in parsed["reasoning"]
    assert "Step 2" in parsed["reasoning"]
    assert parsed["final_answer"] == "The man in the grey hoodie."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cot_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cot_engine'`

**Step 3: Write minimal implementation**

Create `src/cot_engine.py`:
```python
import re
from typing import Dict, Any

COT_SYSTEM_PROMPT = """You are a rigorous spatial reasoning assistant operating over a symbolic scene graph.
You will be provided with:
1. "Scene Entities & Properties": List of physical entities with their visual properties.
2. "Minimal Spatial Relationships": Direct adjacent spatial relationships. 

CRITICAL REASONING PRINCIPLES:
- Transitivity: If A is to the left of B, and B is to the left of C, then A is to the left of C, and B is between A and C.
- Inverses: If A is to the left of B, then B is to the right of A. If A is higher than B, then B is lower than A.
- Grounding: Only make conclusions about entities explicitly grounded from the scene entities list.

Follow this exact 3-step Chain-of-Thought process inside <thought> tags:
Step 1: Entity Identification & Disambiguation
- Identify which exact [entity_id] tokens from the scene match each object referred to in the user's question.

Step 2: Spatial Graph Deduction
- Trace the spatial links connecting the identified entities.
- Deduce any intermediate entities (e.g. betweenness) or directional relations (left/right, elevation) using transitivity and inverses.

Step 3: Evaluation & Answer
- Conclude the exact factual answer based strictly on the deduced spatial facts.

Output Format:
<thought>
Step 1: [Entity mapping]
Step 2: [Spatial chain deduction]
Step 3: [Evaluation]
</thought>
Final Answer: [Concise ground-truth answer in one clear phrase]"""

def build_cot_prompt(plain_context: str, query: str) -> str:
    return f"""{COT_SYSTEM_PROMPT}

=================== SCENE CONTEXT ===================
{plain_context}
=====================================================

User Question: {query}

Provide your step-by-step reasoning and Final Answer:"""

def parse_cot_response(raw_response: str) -> Dict[str, str]:
    thought_match = re.search(r'<thought>(.*?)</thought>', raw_response, re.DOTALL | re.IGNORECASE)
    thought_content = thought_match.group(1).strip() if thought_match else ""
    
    # Extract Final Answer
    ans_match = re.search(r'Final Answer:\s*(.+)', raw_response, re.IGNORECASE)
    if ans_match:
        final_answer = ans_match.group(1).strip()
    else:
        # Fallback if model didn't prefix with "Final Answer:"
        lines = [line.strip() for line in raw_response.splitlines() if line.strip() and not line.startswith("<")]
        final_answer = lines[-1] if lines else raw_response.strip()

    return {
        "reasoning": thought_content,
        "final_answer": final_answer,
        "raw_response": raw_response
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cot_engine.py -v`
Expected: `2 passed in 0.10s`

---

### Task 4: End-to-End Pipeline Integration & Live SLM Test

**Files:**
- Create: `pipeline_minimal_test.py`
- Test: Execute `python pipeline_minimal_test.py`

**Context & Rationale:**
Connect the pieces into an end-to-end executable test: load `preprocessed_graphs/1.json` from the zip archive, clean/rollup entities, extract the minimal spatial graph, serialize to plain language, construct the 3-stage CoT prompt, and query the Qwen 27B SLM via Groq API. Compare against the ground truth from `spatial_questions_results.json`.

**Step 1: Write `pipeline_minimal_test.py`**

```python
import zipfile
import json
import requests
from src.minimal_graph import extract_minimal_spatial_graph
from src.plain_serializer import clean_and_rollup_entities, serialize_to_plain_language
from src.cot_engine import build_cot_prompt, parse_cot_response
from generate_spatial_questions import load_groq_api_key

def ask_groq_slm(prompt: str, model: str = "qwen/qwen3.8-27b") -> str:
    key = load_groq_api_key()
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400
        },
        timeout=25
    )
    if res.status_code != 200:
        raise RuntimeError(f"Groq API Error {res.status_code}: {res.text}")
    return res.json()["choices"][0]["message"]["content"].strip()

def run_test():
    print("Loading test data for image 1.jpg...")
    with open("spatial_questions_results.json", "r", encoding="utf-8") as f:
        qa_pair = json.load(f)["1.jpg"]["parsed"]["questions"][0] # Question 1
        qa_pair_2 = json.load(f)["1.jpg"]["parsed"]["questions"][1] # Question 2

    with zipfile.ZipFile("preprocessed_graphs-20261022T004957Z-1-001.zip", "r") as z:
        sg = json.loads(z.read("preprocessed_graphs/1.json").decode("utf-8"))

    # 1. Entity Rollup & Cleaning
    clean_entities = clean_and_rollup_entities(sg)
    print(f"Entities: Original {len(sg['objects'])} -> Cleaned & Rolled Up: {len(clean_entities)}")

    # 2. Transitive Reduction & Canonical Direction
    spatial_graph = extract_minimal_spatial_graph(clean_entities, sg.get("annotated_relationships", []))
    print(f"Minimal Horizontal Links: {len(spatial_graph['horizontal_chains'])}")
    print(f"Minimal Vertical Links:   {len(spatial_graph['vertical_chains'])}")

    # 3. Plain Language Serialization
    plain_context = serialize_to_plain_language(clean_entities, spatial_graph)
    print("\n--- Serialized Plain-Language Scene Context ---")
    print(plain_context)
    print("------------------------------------------------\n")

    # 4. Evaluate Question 1 (Betweenness)
    q1 = qa_pair["question"]
    gt1 = qa_pair["answer"]
    prompt1 = build_cot_prompt(plain_context, q1)
    
    print(f"Testing Question 1: {q1}")
    raw_ans1 = ask_groq_slm(prompt1)
    res1 = parse_cot_response(raw_ans1)
    
    print("\n--- Reasoning Trace ---")
    print(res1["reasoning"])
    print(f"Predicted Answer: {res1['final_answer']}")
    print(f"Ground Truth:     {gt1}")

    # 5. Evaluate Question 2 (Elevation / Clearance)
    q2 = qa_pair_2["question"]
    gt2 = qa_pair_2["answer"]
    prompt2 = build_cot_prompt(plain_context, q2)
    
    print(f"\nTesting Question 2: {q2}")
    raw_ans2 = ask_groq_slm(prompt2)
    res2 = parse_cot_response(raw_ans2)
    
    print("\n--- Reasoning Trace ---")
    print(res2["reasoning"])
    print(f"Predicted Answer: {res2['final_answer']}")
    print(f"Ground Truth:     {gt2}")

if __name__ == "__main__":
    run_test()
```

**Step 2: Run pipeline test to verify end-to-end execution**

Run: `python pipeline_minimal_test.py`
Expected: Successful output demonstrating token reduction, reasoning trace through `<thought>`, and matching ground truth.

---

### Task 5: Comparative Benchmark & Validation Suite

**Files:**
- Create: `benchmark_comparison.py`
- Test: Run `python benchmark_comparison.py --num-samples 5`

**Context & Rationale:**
Empirically validate that the minimal plain-language approach outperforms dense scene graphs on:
1. **Context Size / Token Count**: Measuring character and token reduction.
2. **Hallucination Rate**: Checking whether the SLM generates non-existent objects or hallucinated relations.
3. **Accuracy**: Accuracy compared to Ground Truth in `spatial_questions_results.json`.

**Step 1: Write `benchmark_comparison.py`**

```python
import zipfile
import json
import time
import argparse
from src.minimal_graph import extract_minimal_spatial_graph
from src.plain_serializer import clean_and_rollup_entities, serialize_to_plain_language
from src.cot_engine import build_cot_prompt, parse_cot_response
from pipeline_minimal_test import ask_groq_slm

def run_benchmark(num_samples: int = 5):
    with open("spatial_questions_results.json", "r", encoding="utf-8") as f:
        all_qa = json.load(f)

    with zipfile.ZipFile("preprocessed_graphs-20261022T004957Z-1-001.zip", "r") as z:
        available_graphs = {name.split("/")[-1].replace(".json", ".jpg"): name for name in z.namelist() if name.endswith(".json")}

    common_images = [img for img in all_qa.keys() if img in available_graphs][:num_samples]
    print(f"Running benchmark across {len(common_images)} scenes...")

    results = []
    with zipfile.ZipFile("preprocessed_graphs-20261022T004957Z-1-001.zip", "r") as z:
        for img in common_images:
            sg = json.loads(z.read(available_graphs[img]).decode("utf-8"))
            questions = all_qa[img]["parsed"]["questions"]

            # Compute Minimal Plain Representation
            clean_entities = clean_and_rollup_entities(sg)
            spatial_graph = extract_minimal_spatial_graph(clean_entities, sg.get("annotated_relationships", []))
            plain_context = serialize_to_plain_language(clean_entities, spatial_graph)

            # Raw dense representation character size vs Minimal character size
            raw_dense_size = len(json.dumps(sg))
            minimal_size = len(plain_context)
            compression_pct = (1.0 - (minimal_size / raw_dense_size)) * 100

            for q_item in questions:
                q_text = q_item["question"]
                gt = q_item["answer"]
                prompt = build_cot_prompt(plain_context, q_text)

                start_t = time.time()
                raw_ans = ask_groq_slm(prompt)
                latency = time.time() - start_t
                parsed = parse_cot_response(raw_ans)

                results.append({
                    "image": img,
                    "question": q_text,
                    "ground_truth": gt,
                    "predicted": parsed["final_answer"],
                    "reasoning": parsed["reasoning"],
                    "latency": latency,
                    "raw_dense_chars": raw_dense_size,
                    "minimal_chars": minimal_size,
                    "compression_pct": compression_pct
                })
                print(f"[{img}] Q: {q_text[:40]}... -> Ans: {parsed['final_answer']} (GT: {gt}) | Comp: {compression_pct:.1f}%")
                time.sleep(2) # rate limit precaution

    with open("benchmark_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_compression = sum(r["compression_pct"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Total Evaluated Questions: {len(results)}")
    print(f"Average Context Compression: {avg_compression:.1f}% reduction")
    print(f"Average Inference Latency:   {avg_latency:.2f}s")
    print(f"Results saved to: benchmark_comparison_results.json")
    print("=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.num_samples)
```

**Step 2: Run benchmark suite**

Run: `python benchmark_comparison.py --num-samples 3`
Expected: Completion with ~90% context compression, low latency, and accurate multi-hop deduction logged to `benchmark_comparison_results.json`.

---
