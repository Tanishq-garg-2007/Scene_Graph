import time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3.5-2B"
]

def benchmark(image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1280, 1280))
    
    for mid in MODELS:
        print(f"\n--- Model: {mid} ---")
        t0 = time.time()
        try:
            processor = AutoProcessor.from_pretrained(mid)
            model = AutoModelForVision2Seq.from_pretrained(
                mid, torch_dtype=torch.float16, device_map="cuda"
            )
            load_time = time.time() - t0
            
            inputs = processor(
                text=[f"<|im_start|>user\n<|image_pad|>{prompt}<|im_end|>\n<|im_start|>assistant\n"],
                images=[image],
                return_tensors="pt"
            ).to("cuda")
            
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=300, do_sample=False, repetition_penalty=1.05)
            gen_time = time.time() - t1
            
            text = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"Load: {load_time:.1f}s | Gen: {gen_time:.1f}s | VRAM: {vram:.1f}GB")
            print(f"Output:\n{text[:200]}...")
        except Exception as e:
            print(f"Failed: {e}")
        finally:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    benchmark("images/1.jpg", "Identify the focal object and its physical contacts as JSON.")

