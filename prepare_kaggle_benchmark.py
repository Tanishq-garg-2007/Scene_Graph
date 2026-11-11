import base64
import json

def get_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img1_b64 = get_b64("images/1.jpg")
img3_b64 = get_b64("images/3.jpg")

kernel_code = f'''# Live Kaggle GPU Benchmark for Qwen Models
import os, sys, time, json, re, base64, io, subprocess, gc
import torch
from PIL import Image

print("[INFO] Upgrading transformers and installing qwen-vl-utils...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate"])

import transformers
print(f"[INFO] Using PyTorch {{torch.__version__}}, Transformers {{transformers.__version__}}")
print(f"[INFO] GPU: {{torch.cuda.get_device_name(0)}} ({{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB)")

MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3.5-2B"
]

IMAGES = {{
    "3.jpg": "{img3_b64}",
    "1.jpg": "{img1_b64}"
}}

TEST_CASES = [
    {{
        "id": "T1_PHONE_SUBSTRATE",
        "image_key": "3.jpg",
        "query": "What object is the corded telephone resting directly on, and what item is located immediately to its left?"
    }},
    {{
        "id": "T2_HOODIE_MAN_CONTACT",
        "image_key": "1.jpg",
        "query": "Look at the man on the right wearing the grey hoodie: what color footwear is he wearing, and what surface is he standing directly on?"
    }}
]

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

def decode_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

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

def load_model_and_processor(model_id):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    loaders = []
    if "Qwen2.5-VL" in model_id or "Qwen2_5_VL" in model_id:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            loaders.append(Qwen2_5_VLForConditionalGeneration)
        except ImportError:
            pass
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForCausalLM
        loaders.append(AutoModelForCausalLM)
    except ImportError:
        pass
    try:
        from transformers import AutoModel
        loaders.append(AutoModel)
    except ImportError:
        pass

    last_err = None
    for loader in loaders:
        try:
            model = loader.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
            )
            print(f"[INFO] Loaded {{model_id}} with {{loader.__name__}}")
            return model, processor
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load {{model_id}}: {{last_err}}")

results = {{}}

for model_id in MODELS:
    print(f"\\n{'='*70}\\nBENCHMARKING MODEL: {{model_id}}\\n{'='*70}")
    model_record = {{"tests": [], "errors": []}}
    
    t_load_start = time.time()
    try:
        model, processor = load_model_and_processor(model_id)
        load_duration = time.time() - t_load_start
        model_record["load_time_seconds"] = round(load_duration, 2)
        print(f"[INFO] {{model_id}} loaded successfully in {{load_duration:.2f}}s")
    except Exception as e:
        err_msg = f"Failed to load {{model_id}}: {{str(e)}}"
        print(f"[ERROR] {{err_msg}}")
        model_record["errors"].append(err_msg)
        results[model_id] = model_record
        gc.collect()
        torch.cuda.empty_cache()
        continue
        
    for tc in TEST_CASES:
        test_id = tc["id"]
        query_text = tc["query"]
        img = decode_image(IMAGES[tc["image_key"]])
        img.thumbnail((1280, 1280))
        
        prompt = USER_PROMPT_TEMPLATE.format(question=query_text)
        messages = [
            {{"role": "system", "content": SYSTEM_PROMPT}},
            {{
                "role": "user",
                "content": [
                    {{"type": "image", "image": img}},
                    {{"type": "text", "text": prompt}}
                ]
            }}
        ]
        
        torch.cuda.reset_peak_memory_stats()
        t_gen_start = time.time()
        try:
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=[img], return_tensors="pt", padding=True).to("cuda")
            
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    do_sample=False,
                    repetition_penalty=1.05
                )
            gen_duration = time.time() - t_gen_start
            gen_tokens = out_ids[:, inputs.input_ids.shape[1]:]
            num_tokens = gen_tokens.shape[1]
            tok_per_sec = round(num_tokens / max(gen_duration, 0.001), 2)
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            
            decoded = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
            parsed = extract_json(decoded)
            
            test_res = {{
                "test_id": test_id,
                "latency_seconds": round(gen_duration, 2),
                "generated_tokens": num_tokens,
                "tokens_per_second": tok_per_sec,
                "peak_vram_gb": peak_vram_gb,
                "json_valid": parsed is not None,
                "parsed_output": parsed,
                "raw_output": decoded
            }}
            model_record["tests"].append(test_res)
            print(f"[RESULT] {{test_id}} | Latency: {{gen_duration:.2f}}s | Tokens: {{num_tokens}} ({{tok_per_sec}} tok/s) | VRAM: {{peak_vram_gb}}GB | JSON Valid: {{parsed is not None}}")
            if parsed:
                print(f"Verified Answer: {{parsed.get('verified_answer')}}")
        except Exception as e:
            err_msg = f"Inference failed on {{test_id}}: {{str(e)}}"
            print(f"[ERROR] {{err_msg}}")
            model_record["tests"].append({{"test_id": test_id, "error": err_msg}})
            
    results[model_id] = model_record
    
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\n[COMPLETE] Benchmark execution finished. Results saved to benchmark_results.json")
'''

with open("kaggle_kernel.py", "w", encoding="utf-8") as f:
    f.write(kernel_code)

print("kaggle_kernel.py generated successfully.")

