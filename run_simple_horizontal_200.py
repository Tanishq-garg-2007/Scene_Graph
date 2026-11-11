"""
Batch Evaluation of Horizontal Chain Reasoning on Qwen 0.8B across all 200 images.
Clean, minimal format:
Input: Relational Edges + Clean Horizontal Chain + Question 1
Output JSON: { "1.jpg": { "question": "...", "response": "..." } }
"""

import os
import re
import sys
import json
import time
import requests

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:0.8b"
OUTPUT_FILE = "simple_horizontal_results_200.json"

SYSTEM_PROMPT = """You are an expert spatial reasoning assistant.
Solve the spatial question using Chain of Thought:
1. Identify Entity A and Entity B from the question and relational edges.
2. In the horizontal chain, extract the complete sub-sequence of ALL entities located between Entity A and Entity B.
3. Final Answer: List EVERY intermediate entity found in that sub-sequence."""

def natural_sort_key(filename: str):
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def run_all():
    print("=" * 70)
    print("STARTING HORIZONTAL REASONING RUN FOR ALL 200 IMAGES")
    print(f"Model: {MODEL} | Max Tokens: 1024 | Output: {OUTPUT_FILE}")
    print("=" * 70)

    with open("spatial_questions_results.json", "r", encoding="utf-8") as f:
        questions_db = json.load(f)

    image_keys = sorted(list(questions_db.keys()), key=natural_sort_key)
    total_images = len(image_keys)

    # Load existing checkpoint if resuming
    results = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as rf:
                results = json.load(rf)
            print(f"Loaded existing checkpoint: {len(results)} images already completed.")
        except Exception:
            results = {}

    start_total_time = time.time()
    processed_this_run = 0

    for idx, img_key in enumerate(image_keys, 1):
        if img_key in results:
            print(f"[{idx}/{total_images}] [CACHED] {img_key}")
            continue

        img_id = re.sub(r"\D", "", img_key)
        converted_path = f"converted_graphs/{img_id}.json"

        if not os.path.exists(converted_path):
            print(f"[{idx}/{total_images}] Warning: {converted_path} not found. Skipping.")
            continue

        with open(converted_path, "r", encoding="utf-8") as gf:
            graph_data = json.load(gf)

        parts = graph_data["prompt_text"].split("### Spatial Ordering Chains:")
        edges = parts[0].strip()
        horiz_chain = " -> ".join(graph_data["horizontal_ordering"])

        q_list = questions_db[img_key].get("parsed", {}).get("questions", [])
        if not q_list:
            print(f"[{idx}/{total_images}] Warning: No questions for {img_key}. Skipping.")
            continue
        
        question = q_list[0]["question"]

        user_prompt = f"""{edges}

### Horizontal Ordering Chain (Left-to-Right):
{horiz_chain}

Question: {question}"""

        t0 = time.time()
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024
                }
            }, timeout=120).json()
            
            duration = round(time.time() - t0, 2)
            raw_response = res.get("message", {}).get("content", "").strip()
            eval_tokens = res.get("eval_count", 0)

            # Store minimal format: only question and response
            results[img_key] = {
                "question": question,
                "response": raw_response
            }

            # Checkpoint immediately to disk
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
                json.dump(results, out, indent=2)

            processed_this_run += 1
            print(f"[{idx}/{total_images}] {img_key} ({duration}s, {eval_tokens} tok)")

        except Exception as e:
            print(f"[{idx}/{total_images}] Error on {img_key}: {e}")
            time.sleep(3)

    total_time = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 70)
    print("COMPLETED ALL 200 IMAGES!")
    print(f"Total Completed in JSON: {len(results)}/{total_images}")
    print(f"Total Time: {total_time}s")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    run_all()
