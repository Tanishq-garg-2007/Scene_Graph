import os
import io
import json
import base64
import zipfile

SELECTED_IMAGES = [
    '167.jpg', '29.jpg', '7.jpg', '193.jpg', '73.jpg', '65.jpg', '59.jpg', '36.jpg',
    '192.jpg', '27.jpg', '177.jpg', '142.jpg', '23.jpg', '154.jpg', '111.jpg', '9.jpg',
    '8.jpg', '24.jpg', '57.jpg', '62.jpg', '132.jpg', '158.jpg', '201.jpg', '146.jpg'
]

with open('spatial_questions_results.json') as f:
    data = json.load(f)

benchmark_items = []
for idx, img in enumerate(SELECTED_IMAGES, 1):
    item = data[img]
    qs = item.get('parsed', {}).get('questions', [])
    q_chosen = qs[idx % len(qs)] if qs else None
    if q_chosen:
        benchmark_items.append({
            'idx': idx,
            'image_file': img,
            'question': q_chosen.get('question'),
            'ground_truth': q_chosen.get('answer'),
            'feature': q_chosen.get('relation_or_feature', 'spatial')
        })

# Create in-memory zip of images
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for img in SELECTED_IMAGES:
        z.write(os.path.join('images', img), img)

zip_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
print(f"Packaged {len(SELECTED_IMAGES)} images. Base64 zip length: {len(zip_b64)}")

kernel_script = f'''# 24-Query Spatial Reasoning Benchmark for Qwen2.5-VL-3B-Instruct
import os, sys, time, json, re, base64, io, zipfile, subprocess, gc
import torch
from PIL import Image

print("[INFO] Upgrading transformers and installing qwen-vl-utils...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate"])

import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print(f"[INFO] PyTorch: {{torch.__version__}}, Transformers: {{transformers.__version__}}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
print(f"[INFO] GPU: {{torch.cuda.get_device_name(0)}} ({{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB)")

# Unpack images
ZIP_B64 = "{zip_b64}"
IMG_DIR = "/tmp/benchmark_images"
os.makedirs(IMG_DIR, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP_B64))) as z:
    z.extractall(IMG_DIR)
print(f"[INFO] Unpacked {{len(os.listdir(IMG_DIR))}} images to {{IMG_DIR}}")

BENCHMARK_ITEMS = {json.dumps(benchmark_items, indent=2)}

SYSTEM_PROMPT = """You are an elite spatial vision architect specializing in grounded visual reasoning and scene graph generation.
You never guess from generic room context. You strictly trace physical contacts, containment, and spatial relationships as requested by the query.
You return ONLY valid JSON matching the exact schema."""

USER_PROMPT_TEMPLATE = """Examine the high-resolution image to resolve this visual query:
QUERY: "{{question}}"

INSPECTION PROTOCOL:
1. Focal Landmark: Locate the primary object or entity relevant to the query.
2. Physical Contact Tracing: Identify every adjacent object that directly touches the focal entity. Trace all contact boundaries, including surfaces below, walls behind, lateral neighbors in contact, and enclosing containers.
3. Context Landmarks: Identify up to 3 major adjacent landmarks relevant to the query.
4. Visible Attributes & Text: Inspect fine-grained colors, states, and legible inscriptions if relevant (or "none").
5. Bounding Box Anchoring: Estimate normalized coordinates [ymin, xmin, ymax, xmax] (0-1000) for every detected object node.

Output strictly valid JSON with this schema:
{{{{
  "query": "{{question}}",
  "micro_inspection_cot": {{{{
    "focal_landmark": "<identification and location of primary query entity>",
    "spatial_relation_trace": "<exact physical contact boundary, touching neighbor, container, or support surface relevant to the query>",
    "nearby_landmarks": ["<list up to 3 major adjacent landmarks>"],
    "visual_evidence_and_text": "<exact visual features, colors, states, or legible text>"
  }}}},
  "grounded_scene_graph": {{{{
    "nodes": [
      {{{{
        "id": "obj_1",
        "name": "<precise noun>",
        "box_2d": [0, 0, 0, 0],
        "attributes": ["..."],
        "visible_text": "<text or none>"
      }}}}
    ],
    "edges": [
      {{{{
        "subject": "obj_1",
        "predicate": "touches | rests_on | leans_against | supported_by | inside | contains | mounted_on | attached_to | holding | wearing | adjacent_to",
        "object": "obj_2"
      }}}}
    ]
  }}}},
  "verified_answer": "<direct, non-tautological answer to the query citing the exact physical evidence>"
}}}}"""

def extract_json(raw_text):
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{{")
    end = cleaned.rfind("}}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"[INFO] Loading model {{MODEL_ID}}...")
t_load = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print(f"[INFO] Model loaded in {{time.time() - t_load:.2f}}s")

results = []
correct_count = 0
valid_json_count = 0

print("\\n" + "=" * 80)
print(f"STARTING 24-QUERY EMPIRICAL BENCHMARK")
print("=" * 80)

for item in BENCHMARK_ITEMS:
    idx = item["idx"]
    img_name = item["image_file"]
    query_text = item["question"]
    ground_truth = item["ground_truth"]
    
    img_path = os.path.join(IMG_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1280, 1280))
    
    prompt = USER_PROMPT_TEMPLATE.format(question=query_text)
    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{
            "role": "user",
            "content": [
                {{"type": "image", "image": image}},
                {{"type": "text", "text": prompt}}
            ]
        }}
    ]
    
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True).to("cuda")
    
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=650,
            do_sample=False,
            repetition_penalty=1.05
        )
    gen_time = time.time() - t0
    gen_tokens = out_ids[:, inputs.input_ids.shape[1]:]
    num_tokens = gen_tokens.shape[1]
    tok_per_sec = round(num_tokens / max(gen_time, 0.001), 2)
    
    decoded = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
    parsed = extract_json(decoded)
    
    is_valid_json = parsed is not None
    if is_valid_json:
        valid_json_count += 1
        v_answer = parsed.get("verified_answer", "").strip()
        nodes = parsed.get("grounded_scene_graph", {{}}).get("nodes", [])
        edges = parsed.get("grounded_scene_graph", {{}}).get("edges", [])
    else:
        v_answer = decoded
        nodes = []
        edges = []
        
    record = {{
        "idx": idx,
        "image": img_name,
        "question": query_text,
        "ground_truth": ground_truth,
        "verified_answer": v_answer,
        "latency_seconds": round(gen_time, 2),
        "tokens": num_tokens,
        "tokens_per_sec": tok_per_sec,
        "json_valid": is_valid_json,
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "parsed_json": parsed,
        "raw_output": decoded
    }}
    results.append(record)
    
    print(f"\\n[{{idx:02d}}/24] Image: {{img_name}} | Latency: {{gen_time:.2f}}s | Tokens: {{num_tokens}} ({{tok_per_sec}} tok/s) | JSON Valid: {{is_valid_json}}")
    print(f"  Q:  {{query_text}}")
    print(f"  GT: {{ground_truth}}")
    print(f"  A:  {{v_answer}}")

print("\\n" + "=" * 80)
print(f"BENCHMARK COMPLETED: 24/24 QUERIES EXECUTED")
print(f"Valid JSON Rate: {{valid_json_count}}/24 ({{valid_json_count/24*100:.1f}}%)")
print("=" * 80)

with open("benchmark_24_results.json", "w") as f:
    json.dump({{
        "model": MODEL_ID,
        "total_queries": len(results),
        "valid_json_count": valid_json_count,
        "results": results
    }}, f, indent=2)

print("[INFO] Results saved to benchmark_24_results.json")
'''

os.makedirs('kaggle_benchmark_24', exist_ok=True)
with open('kaggle_benchmark_24/kernel_benchmark_24.py', 'w', encoding='utf-8') as f:
    f.write(kernel_script)

metadata = {
    "id": "tanishqgarg2007/qwen-24-query-spatial-benchmark",
    "title": "Qwen 24 Query Spatial Benchmark",
    "code_file": "kernel_benchmark_24.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

with open('kaggle_benchmark_24/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Kaggle package created successfully in kaggle_benchmark_24/")

