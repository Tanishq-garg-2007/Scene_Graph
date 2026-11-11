"""
Batch Spatial Question Generator using Groq-hosted Qwen 27B Vision Model.

Focus: Multi-entity relationship chains (A -> B -> C) and structural clearances / elevation.
Features:
- Incremental checkpointing (saves after every image to spatial_questions_results.json)
- Automatic rate-limit handling & backoff retry on HTTP 429
- Natural numeric image sorting (1.jpg, 2.jpg, ... 200.jpg)
- Resume capability on interruption
"""

import os
import re
import sys
import time
import json
import base64
import argparse
import requests
from typing import Dict, Any, List

DEFAULT_MODEL = "qwen/qwen3.8-27b"
RESULTS_FILE = "spatial_questions_results.json"
IMAGES_DIR = "images"

PROMPT_TEMPLATE = """You are an expert evaluator assessing spatial reasoning and scene graphs in multimodal AI.
Analyze the image and generate exactly 2 clear, direct, and straightforward spatial questions:

1. Question 1 (Multi-Entity Chain: A -> B -> C):
   Ask a direct question connecting 3 objects (e.g., 'What object is positioned between X and Y?', or 'Starting from object A, which object is directly next to it, and what is behind that object?'). Keep the question simple and easy to understand.

2. Question 2 (Structural Clearance / Relative Position):
   Ask a direct question testing vertical elevation, clearance, or relative distance between two key objects (e.g., 'Is object A positioned higher or lower than object B?', or 'What object is directly underneath/above X?').

Requirements:
- Keep questions simple, concise, and direct (avoid long or complicated sentence structures).
- Provide an exact, unambiguous ground-truth answer.

Output Format:
### Question 1:
- **Question:** [Direct question text]
- **Relation Chain:** [A -> B -> C]
- **Answer:** [Clear factual answer]

### Question 2:
- **Question:** [Direct question text]
- **Spatial Feature:** [Clearance / Elevation / Alignment]
- **Answer:** [Clear factual answer]"""


def load_groq_api_key() -> str:
    """Loads Groq API key from environment or .ENV file."""
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("Groq_API_KEY")
    if key and key.strip():
        return key.strip()

    for env_name in [".ENV", ".env"]:
        if os.path.exists(env_name):
            with open(env_name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Groq_API_KEY=") or line.startswith("GROQ_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"\'')
                        if val:
                            return val
    raise ValueError("Groq API key not found in environment or .ENV file.")


def natural_sort_key(filename: str):
    """Sort filenames like 1.jpg, 2.jpg, 10.jpg, 100.jpg in correct numerical order."""
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def encode_image(image_path: str) -> str:
    """Base64 encodes an image."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    """Returns mime type based on extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def query_groq_with_retry(image_path: str, api_key: str, model: str = DEFAULT_MODEL, max_retries: int = 5) -> str:
    """Sends vision request to Groq API with robust exponential backoff on rate limits."""
    b64_data = encode_image(image_path)
    mime_type = get_mime_type(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    url = "https://api.groq.com/openai/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            elif res.status_code == 429:
                # Rate limit encountered
                wait_time = 20 * (attempt + 1)
                try:
                    err_json = res.json().get("error", {})
                    msg = err_json.get("message", "")
                    match = re.search(r"try again in ([\d\.]+)s", msg)
                    if match:
                        wait_time = float(match.group(1)) + 2.0
                except Exception:
                    pass
                print(f"  [Rate Limit 429] Waiting {wait_time:.1f}s before retry (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  [Error {res.status_code}] {res.text}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Groq API Error {res.status_code}: {res.text}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  [Network Error] {e}. Retrying in 10s...")
            time.sleep(10)

    raise RuntimeError(f"Failed to process image {image_path} after {max_retries} attempts.")


def parse_qa_output(raw_text: str) -> Dict[str, Any]:
    """Helper to structure the raw LLM output into clean JSON fields."""
    parsed = {"raw": raw_text, "questions": []}
    
    # Simple regex extraction for Q1 and Q2
    q_blocks = re.split(r'###\s*Question\s*\d+:', raw_text, flags=re.IGNORECASE)
    for i, block in enumerate(q_blocks[1:], 1):
        q_match = re.search(r'\*\*Question:\*\*\s*(.+)', block)
        ans_match = re.search(r'\*\*Answer:\*\*\s*(.+)', block)
        chain_match = re.search(r'\*\*(?:Relation Chain|Spatial Feature):\*\*\s*(.+)', block)
        
        parsed["questions"].append({
            "id": i,
            "question": q_match.group(1).strip() if q_match else "",
            "relation_or_feature": chain_match.group(1).strip() if chain_match else "",
            "answer": ans_match.group(1).strip() if ans_match else ""
        })
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Generate 2 tough spatial questions per image using Groq Qwen Vision.")
    parser.add_argument("--images-dir", type=str, default=IMAGES_DIR, help="Path to images directory.")
    parser.add_argument("--output", type=str, default=RESULTS_FILE, help="Path to output JSON results.")
    parser.add_argument("--image", type=str, default=None, help="Process a single specific image instead of batch.")
    parser.add_argument("--delay", type=float, default=12.0, help="Base delay between requests in seconds (to respect 1000 OTPM rate limit).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to process.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model identifier.")

    args = parser.parse_args()

    api_key = load_groq_api_key()
    print(f"Loaded Groq API key: {api_key[:6]}...{api_key[-4:]}")

    # Single image mode
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        output = query_groq_with_retry(args.image, api_key, model=args.model)
        print("\n--- Output ---")
        print(output)
        return

    # Batch mode
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' not found.")
        sys.exit(1)

    image_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    image_files.sort(key=natural_sort_key)

    if args.limit:
        image_files = image_files[:args.limit]

    total_images = len(image_files)
    print(f"Found {total_images} images in '{args.images_dir}'.")

    # Load existing checkpoint if available
    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing checkpoint: {len(results)} images already processed.")
        except Exception as e:
            print(f"Warning: Could not read existing results file ({e}). Starting fresh.")

    # Process images
    processed_count = 0
    for idx, img_name in enumerate(image_files, 1):
        if img_name in results:
            continue

        img_path = os.path.join(args.images_dir, img_name)
        print(f"\n[{idx}/{total_images}] Processing {img_name}...")

        try:
            raw_text = query_groq_with_retry(img_path, api_key, model=args.model)
            parsed = parse_qa_output(raw_text)
            
            results[img_name] = {
                "image": img_name,
                "parsed": parsed,
                "raw_output": raw_text
            }

            # Save checkpoint after every image
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            print(f"  -> Successfully generated 2 questions. Checkpoint saved.")
            processed_count += 1

            # Respect rate limit
            time.sleep(args.delay)

        except Exception as e:
            print(f"  [Failed] Could not process {img_name}: {e}")

    print(f"\n[DONE] Completed batch processing! Total processed in this run: {processed_count}. Total in {args.output}: {len(results)}.")


if __name__ == "__main__":
    main()
