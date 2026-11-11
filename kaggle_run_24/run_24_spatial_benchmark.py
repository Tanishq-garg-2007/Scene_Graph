# 24-Query Empirical Spatial Reasoning Benchmark on Kaggle GPU
# Model: Qwen/Qwen2.5-VL-3B-Instruct
# Methodology: Embedded Structured CoT + Physical Contact Tracing (qwen_spatial_vqa.py)

import os, sys, time, json, re, subprocess, gc
import torch
from PIL import Image

print("[INFO] Upgrading transformers and installing qwen-vl-utils...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate"])

import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print(f"[INFO] PyTorch: {torch.__version__}, Transformers: {transformers.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

# Locate dataset directory
DATASET_DIR = "/kaggle/input/spatial-benchmark-24-images"
if not os.path.exists(DATASET_DIR) or not any(f.endswith('.jpg') for f in os.listdir(DATASET_DIR)):
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.endswith('.jpg') for f in files):
            DATASET_DIR = root
            break
print(f"[INFO] Using dataset directory: {DATASET_DIR}")

BENCHMARK_ITEMS = [
  {"idx": 1, "image_file": "167.jpg", "question": "Is the metal staircase positioned higher or lower than the wooden cabinet?", "ground_truth": "The metal staircase is positioned higher than the wooden cabinet."},
  {"idx": 2, "image_file": "29.jpg", "question": "What object is located between the woman in the white tank top and the woman in the red shirt?", "ground_truth": "The white boat labeled \"BLUE ART\"."},
  {"idx": 3, "image_file": "7.jpg", "question": "Is the whiteboard positioned higher or lower than the desk?", "ground_truth": "Higher"},
  {"idx": 4, "image_file": "193.jpg", "question": "What object is located between the rows of seats and the large screen?", "ground_truth": "The stage."},
  {"idx": 5, "image_file": "73.jpg", "question": "Is the white plastic chair in the foreground positioned higher or lower than the table surface?", "ground_truth": "The chair is positioned lower than the table surface."},
  {"idx": 6, "image_file": "65.jpg", "question": "What vehicle is positioned between the white van and the grey car?", "ground_truth": "A man (wearing a grey sweater) is standing between the white van and the grey car."},
  {"idx": 7, "image_file": "59.jpg", "question": "Is the \"EROSKI center\" sign positioned higher or lower than the \"ELITE RESTAURANTE\" sign?", "ground_truth": "Lower"},
  {"idx": 8, "image_file": "36.jpg", "question": "What object is located between the dark green hutch and the refrigerator?", "ground_truth": "There is no object between them; they are positioned directly adjacent to each other."},
  {"idx": 9, "image_file": "192.jpg", "question": "Is the yellow taxi sign on the roof of the center car positioned higher or lower than the blue \"TAXI\" street sign on the right?", "ground_truth": "The yellow taxi sign on the roof is positioned lower than the blue \"TAXI\" street sign."},
  {"idx": 10, "image_file": "27.jpg", "question": "What object is located between the table and the wall?", "ground_truth": "The radiator"},
  {"idx": 11, "image_file": "177.jpg", "question": "Is the dental light positioned higher or lower than the dental chair?", "ground_truth": "Higher"},
  {"idx": 12, "image_file": "142.jpg", "question": "What object is located between the fountain and the row of benches?", "ground_truth": "A black lamp post"},
  {"idx": 13, "image_file": "23.jpg", "question": "Is the blue parking sign positioned higher or lower than the top of the black car?", "ground_truth": "Higher"},
  {"idx": 14, "image_file": "154.jpg", "question": "What object is located between the red cup and the white toaster?", "ground_truth": "The coffee pot"},
  {"idx": 15, "image_file": "111.jpg", "question": "Is the spatula positioned higher or lower than the orange?", "ground_truth": "Higher"},
  {"idx": 16, "image_file": "9.jpg", "question": "What object is located between the corkboard on the wall and the window?", "ground_truth": "Bookshelf"},
  {"idx": 17, "image_file": "8.jpg", "question": "Is the white mug positioned higher or lower than the keyboard?", "ground_truth": "The white mug is positioned higher than the keyboard."},
  {"idx": 18, "image_file": "24.jpg", "question": "What vehicle is parked directly in front of the red car, and what type of vehicle is parked in front of that one?", "ground_truth": "A white van is parked in front of the red car, and a row of motorcycles is parked in front of the white van."},
  {"idx": 19, "image_file": "57.jpg", "question": "Is the man in the suit positioned higher or lower than the parked car?", "ground_truth": "Lower"},
  {"idx": 20, "image_file": "62.jpg", "question": "What object is located between the large planter and the beige car?", "ground_truth": "A metal pole (or street lamp post)."},
  {"idx": 21, "image_file": "132.jpg", "question": "Is the street lamp positioned higher or lower than the car?", "ground_truth": "Higher"},
  {"idx": 22, "image_file": "158.jpg", "question": "What object is located between the orange flower on the left and the pink flower in the background?", "ground_truth": "The red flower (with the butterfly on it)."},
  {"idx": 23, "image_file": "201.jpg", "question": "Is the blue trash bin positioned to the left or right of the green trash bin?", "ground_truth": "The blue trash bin is to the left of the green trash bin."},
  {"idx": 24, "image_file": "146.jpg", "question": "What object is located between the motorcycle and the broom?", "ground_truth": "The wall"}
]

SYSTEM_PROMPT = """You are an elite spatial vision architect specializing in grounded visual reasoning and scene graph generation.
You never guess from generic room context. You strictly trace physical contacts, containment, and spatial relationships as requested by the query.
You return ONLY valid JSON matching the exact schema."""

USER_PROMPT_TEMPLATE = """Examine the high-resolution image to resolve this visual query:
QUERY: "{question}"

INSPECTION PROTOCOL:
1. Focal Landmark: Locate the primary object or entity relevant to the query.
2. Physical Contact Tracing: Identify every adjacent object that directly touches the focal entity. Trace all contact boundaries, including surfaces below, walls behind, lateral neighbors in contact, and enclosing containers.
3. Context Landmarks: Identify up to 3 major adjacent landmarks relevant to the query.
4. Visible Attributes & Text: Inspect fine-grained colors, states, and legible inscriptions if relevant (or "none").
5. Bounding Box Anchoring: Estimate normalized coordinates [ymin, xmin, ymax, xmax] (0-1000) for every detected object node.

Output strictly valid JSON with this schema:
{{
  "query": "{question}",
  "micro_inspection_cot": {{
    "focal_landmark": "<identification and location of primary query entity>",
    "spatial_relation_trace": "<exact physical contact boundary, touching neighbor, container, or support surface relevant to the query>",
    "nearby_landmarks": ["<list up to 3 major adjacent landmarks>"],
    "visual_evidence_and_text": "<exact visual features, colors, states, or legible text>"
  }},
  "grounded_scene_graph": {{
    "nodes": [
      {{
        "id": "obj_1",
        "name": "<precise noun>",
        "box_2d": [0, 0, 0, 0],
        "attributes": ["..."],
        "visible_text": "<text or none>"
      }}
    ],
    "edges": [
      {{
        "subject": "obj_1",
        "predicate": "touches | rests_on | leans_against | supported_by | inside | contains | mounted_on | attached_to | holding | wearing | adjacent_to",
        "object": "obj_2"
      }}
    ]
  }},
  "verified_answer": "<direct, non-tautological answer to the query citing the exact physical evidence>"
}}"""

def extract_json(raw_text):
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"[INFO] Loading model {MODEL_ID}...")
t_load = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print(f"[INFO] Model loaded successfully in {time.time() - t_load:.2f}s")

results = []
valid_json_count = 0

print("\n" + "=" * 80)
print("EXECUTING 24-QUERY BENCHMARK ON KAGGLE TESLA T4")
print("=" * 80)

for item in BENCHMARK_ITEMS:
    idx = item["idx"]
    img_name = item["image_file"]
    query_text = item["question"]
    ground_truth = item["ground_truth"]
    
    img_path = os.path.join(DATASET_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"[WARN] File {img_name} not found in {DATASET_DIR}")
        continue
        
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1280, 1280))
    
    prompt = USER_PROMPT_TEMPLATE.format(question=query_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
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
        nodes = parsed.get("grounded_scene_graph", {}).get("nodes", [])
        edges = parsed.get("grounded_scene_graph", {}).get("edges", [])
    else:
        v_answer = decoded
        nodes = []
        edges = []
        
    record = {
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
    }
    results.append(record)
    
    print(f"\n[{idx:02d}/24] Image: {img_name} | Latency: {gen_time:.2f}s | Tokens: {num_tokens} ({tok_per_sec} tok/s) | JSON Valid: {is_valid_json}")
    print(f"  Q:  {query_text}")
    print(f"  GT: {ground_truth}")
    print(f"  A:  {v_answer}")

print("\n" + "=" * 80)
print(f"BENCHMARK COMPLETED: {len(results)}/24 QUERIES EXECUTED")
print(f"Valid JSON Rate: {valid_json_count}/{len(results)} ({valid_json_count/len(results)*100:.1f}%)")
print("=" * 80)

with open("benchmark_24_results.json", "w") as f:
    json.dump({
        "model": MODEL_ID,
        "total_queries": len(results),
        "valid_json_count": valid_json_count,
        "results": results
    }, f, indent=2)

print("[INFO] Results saved to benchmark_24_results.json successfully.")

