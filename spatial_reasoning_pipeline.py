"""
Spatial Reasoning Pipeline for Small Language Models (SLM)
===========================================================
This script implements the optimized, verified pipeline for visual-spatial reasoning
over minimized scene graphs using local Ollama SLMs (specifically qwen3.5:0.8b).

Key Architecture:
1. Clean Semantic IDs: (e.g. Man_1, Man_2, Van_1, Clock_1) replacing long hashes.
2. Relational Edges: Explicit node-to-node relationships (e.g. wearing, on, behind).
3. Minimal Spatial Relationships: Non-redundant linear topological chains (transitive reduction).
4. Single-Call 3-Step Chain-of-Thought: Controlled reasoning (Entity Matching -> Spatial Chain -> Final Answer)
   with think=False to prevent sub-1B models from entering infinite internal reflection loops.
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:0.8b"

COT_SYSTEM_PROMPT = """You are an expert spatial reasoning assistant.
Solve the question using this 4-step Chain of Thought:
Step 1 - Entity Matching: Identify entity IDs matching the question (match people via clothing/accessories in Relational Edges, and match objects/vehicles directly).
Step 2 - Chain Selection: For "between" questions, choose Horizontal Ordering (the chain is strictly Left-to-Right, where A -> B means A is to the left of B). For height/elevation questions, choose Vertical Ordering (the chain is strictly Highest-to-Lowest, where A -> B means A is higher than B).
Step 3 - Path Tracing: Trace the segment between Entity 1 and Entity 2 along the chosen chain and list the intermediate entities between them.
Step 4 - Final Answer: State the final concise answer."""

class SpatialReasoningEngine:
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def query(self, scene_context: str, question: str, max_tokens: int = 2048, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Executes a spatial reasoning query against the local SLM in a single call.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Scene Graph:\n{scene_context}\n\nQuestion: {question}"}
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        t0 = time.time()
        res = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=240)
        t1 = time.time()
        
        if res.status_code != 200:
            raise RuntimeError(f"Ollama request failed with status {res.status_code}: {res.text}")

        data = res.json()
        response_text = data.get("message", {}).get("content", "").strip()
        
        return {
            "question": question,
            "response": response_text,
            "eval_count": data.get("eval_count", 0),
            "duration_seconds": round(t1 - t0, 2),
            "model": self.model
        }


def load_scene_graph(image_id: int) -> str:
    """
    Loads the dynamically converted scene graph for any image from converted_graphs/.
    Every graph follows the exact same standard format:
      1. Relational Edges
      2. Minimal Spatial Relationships (Horizontal Ordering & Vertical Elevation)
    """
    json_path = f"converted_graphs/{image_id}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompt_text"]


def run_benchmark():
    engine = SpatialReasoningEngine()
    
    # Dynamically load the generated scene graphs for Image 1, 2, and 3
    scene_1 = load_scene_graph(1)
    scene_2 = load_scene_graph(2)
    scene_3 = load_scene_graph(3)

    test_cases = [
        {
            "id": "Image 1 - Test 1: Elevation",
            "scene": scene_1,
            "question": "Is the green clock positioned higher or lower than the heads of the two men standing on the sidewalk?",
            "ground_truth": "Higher"
        },
        {
            "id": "Image 1 - Test 2: Multi-hop Attribute Reasoning",
            "scene": scene_1,
            "question": "What color jacket is worn by the person standing to the right of the man in the red shirt?",
            "ground_truth": "gray / grey"
        },
        {
            "id": "Image 2 - Test 1: Surface & Context",
            "scene": scene_2,
            "question": "What is Man_1 wearing on his back according to the relational edges?",
            "ground_truth": "Backpack_1: silver"
        },
        {
            "id": "Image 3 - Test 1: Betweenness (Desk items)",
            "scene": scene_3,
            "question": "What object is located between Keyboard_1 and Telephone_1 on the desk?",
            "ground_truth": "Mouse_1 (the mouse)"
        }
    ]

    print("=" * 80)
    print("RUNNING PIPELINE WITH DYNAMICALLY CONVERTED SCENE GRAPHS (Budget: 1024)")
    print("All graphs loaded dynamically from converted_graphs/*.json (Zero Hardcoding)")
    print("=" * 80)

    for case in test_cases:
        print(f"\n[+] {case['id']}")
        print(f"QUESTION:     {case['question']}")
        print(f"EXPECTED:     {case['ground_truth']}")
        
        result = engine.query(case["scene"], case["question"], max_tokens=1024)
        
        print(f"DURATION:     {result['duration_seconds']}s | TOKENS: {result['eval_count']}")
        print("\n--- Model Response ---")
        print(result["response"])
        print("-" * 80)

if __name__ == "__main__":
    run_benchmark()
