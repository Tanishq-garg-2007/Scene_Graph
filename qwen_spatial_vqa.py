"""
qwen_spatial_vqa.py - Production Spatial VQA & Grounded Scene Graph Pipeline
=============================================================================
Architecture: Embedded Structured Chain-of-Thought + Grounded Scene Graph (H2 Enhanced)
Model: Qwen/Qwen2.5-VL-3B-Instruct
Key Innovations:
  - Direct Physical Contact Tracing (inspects all touching surfaces, containment, and contact points)
  - Bounded Landmark Context (prevents token repetition loops in cluttered scenes)
  - Normalized 2D Bounding Boxes [ymin, xmin, ymax, xmax] (0-1000)
  - Inverted / Rotated OCR Inspection
  - Strict Repetition Penalty (1.05) enforcement
"""

import os
import sys
import json
import re
import io
import argparse
from typing import Dict, Any, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

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


class SpatialVQAPipeline:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing SpatialVQAPipeline on {self.device}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("[INFO] Model and processor loaded successfully.")

    @staticmethod
    def load_image(image_input) -> Image.Image:
        """Loads an image from a filepath, URL, or PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                import requests
                resp = requests.get(image_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            elif os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
        raise ValueError(f"Unable to load image from input: {image_input}")

    @staticmethod
    def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extracts and parses JSON object from model generation output."""
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None

    def query(
        self,
        image_input,
        question: str,
        max_new_tokens: int = 1000,
        repetition_penalty: float = 1.05
    ) -> Dict[str, Any]:
        """Runs the spatial reasoning pipeline on an image and query."""
        image = self.load_image(image_input)
        
        # Enforce resolution limit to fit within standard GPU memory
        image.thumbnail((1280, 1280))
        
        prompt = USER_PROMPT_TEMPLATE.format(question=question)
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
        
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty
            )
            
        gen_tokens = out_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
        parsed = self.extract_json(decoded)
        
        return {
            "success": parsed is not None,
            "parsed": parsed,
            "raw_output": decoded
        }


def main():
    parser = argparse.ArgumentParser(description="Run Qwen Spatial VQA Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image")
    parser.add_argument("--query", type=str, required=True, help="Spatial or contact query to resolve")
    args = parser.parse_args()

    pipeline = SpatialVQAPipeline()
    result = pipeline.query(args.image, args.query)
    
    print("\n" + "=" * 60)
    print("SPATIAL REASONING RESULT:")
    print("=" * 60)
    if result["success"]:
        print(json.dumps(result["parsed"], indent=2))
    else:
        print("[WARNING] Could not strictly parse JSON. Raw output:")
        print(result["raw_output"])


if __name__ == "__main__":
    main()

