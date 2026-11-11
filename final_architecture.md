# Final Architecture: Grounded Spatial VQA & Scene Graph Pipeline

## 1. Core Architecture Overview
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Methodology**: Embedded Structured Chain-of-Thought (CoT) + Grounded Scene Graph (Single-Pass, Zero-Crop).
- **Core Mechanism**: Autoregressive cross-attention steering via **Generalized Spatial & Relational Tracing** (`spatial_relation_trace`) and bounded landmark context arrays (`max 3 items`), generating normalized 2D bounding boxes `[ymin, xmin, ymax, xmax]` (0-1000) alongside directed relational tuples.

---

## 2. Model & Decoding Hyperparameters

| Parameter | Value | Technical Justification |
| :--- | :--- | :--- |
| `torch_dtype` | `torch.float16` | Optimal performance on Nvidia T4 / Ampere GPUs without precision loss. |
| `do_sample` | `False` (Greedy) | Enforces deterministic visual factuality; prevents creative hallucinations. |
| `repetition_penalty` | `1.05` | **Mandatory**: Prevents single-token looping in uniform image regions. |
| `max_pixels` | `1280 * 28 * 28` | High-resolution ViT patch retention without downsampling small objects. |
| `max_new_tokens` | `1000` | Ample headroom for complete JSON completion. |

---

## 3. Prompts Given to the VLM (Fully Generalized)

### System Prompt
```text
You are an elite spatial vision architect specializing in grounded visual reasoning and scene graph generation.
You never guess from generic room context. You strictly trace physical contacts, containment, and spatial relationships as requested by the query.
You return ONLY valid JSON matching the exact schema.
```

### User Prompt Template
```text
Examine the high-resolution image to resolve this visual query:
QUERY: "{question}"

INSPECTION PROTOCOL:
1. Focal Landmark: Locate the primary object or entity relevant to the query.
2. Physical Contact Tracing: Identify every adjacent object that directly touches the focal entity. Trace all contact boundaries, including surfaces below, walls behind, lateral neighbors in contact, and enclosing containers.
3. Context Landmarks: Identify up to 3 major adjacent landmarks relevant to the query.
4. Visible Attributes & Text: Inspect fine-grained colors, states, and legible inscriptions if relevant (or "none").
5. Bounding Box Anchoring: Estimate normalized coordinates [ymin, xmin, ymax, xmax] (0-1000) for every detected object node.

Output strictly valid JSON with this schema:
{
  "query": "{question}",
  "micro_inspection_cot": {
    "focal_landmark": "<identification and location of primary query entity>",
    "spatial_relation_trace": "<exact physical contact boundary, touching neighbor, container, or support surface relevant to the query>",
    "nearby_landmarks": ["<list up to 3 major adjacent landmarks>"],
    "visual_evidence_and_text": "<exact visual features, colors, states, or legible text>"
  },
  "grounded_scene_graph": {
    "nodes": [
      {
        "id": "obj_1",
        "name": "<precise noun>",
        "box_2d": [0, 0, 0, 0],
        "attributes": ["..."],
        "visible_text": "<text or none>"
      }
    ],
    "edges": [
      {
        "subject": "obj_1",
        "predicate": "touches | rests_on | leans_against | supported_by | inside | contains | mounted_on | attached_to | holding | wearing | adjacent_to",
        "object": "obj_2"
      }
    ]
  },
  "verified_answer": "<direct, non-tautological answer to the query citing the exact physical evidence>"
}
```

---

## 4. Real Tested Sample (Cluttered Workstation)

- **Input Image**: Cluttered Desk Workstation (`media_1793594833091_5a2837f9.jpg`)
- **Query**: `"What exactly is supporting the Apple monitor stand, and is there any text visible on that supporting object?"`

### VLM Single-Pass Output:
```json
{
  "query": "What exactly is supporting the Apple monitor stand, and is there any text visible on that supporting object?",
  "micro_inspection_cot": {
    "focal_landmark": "Apple monitor stand",
    "spatial_relation_trace": "book directly beneath the monitor stand foot",
    "nearby_landmarks": ["sheet music", "keyboard", "lamp"],
    "visual_evidence_and_text": "white spine with printed text: THE COM CARCASH QUIA"
  },
  "grounded_scene_graph": {
    "nodes": [
      {
        "id": "obj_1",
        "name": "Apple monitor stand",
        "box_2d": [245, 71, 786, 239],
        "attributes": ["supporting", "metal"],
        "visible_text": "none"
      },
      {
        "id": "obj_2",
        "name": "book",
        "box_2d": [429, 234, 619, 318],
        "attributes": ["supporting", "white spine"],
        "visible_text": "THE COM CARCASH QUIA"
      }
    ],
    "edges": [
      {
        "subject": "obj_1",
        "predicate": "supported_by",
        "object": "obj_2"
      }
    ]
  },
  "verified_answer": "The Apple monitor stand is supported by a book, and there is text visible on that supporting object."
}
```

---

## 5. Primary Codebase Files
- **Execution Script**: [`qwen_spatial_vqa.py`](file:///c:/Users/tanis/OneDrive/Documents/Ai-Ml/Fine%20Tunning/qwen_spatial_vqa.py) (Production pipeline implementation)
- **Empirical Report**: [`report.md`](file:///c:/Users/tanis/OneDrive/Documents/Ai-Ml/Fine%20Tunning/report.md) (Full benchmark history and findings across 48+ test runs)

