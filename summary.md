# Affordance-Aware Multimodal Reasoning for Indoor Fire Escape Assistance
## Comprehensive Project & Dataset Summary

---

## 1. Project Overview & Repository
- **Project Title:** Affordance-Aware Multimodal Reasoning for Indoor Fire Escape Assistance
- **Researchers:**
  - Tanishq Garg (`IIT2024038`)
  - Disha Gupta (`IIT2024039`)
  - Udaynoor Singh (`IIT2024006`)
- **Institution:** Indian Institute of Information Technology (IIIT)
- **GitHub Repository (Remote Code & Model Storage):**
  - **URL:** [https://github.com/Tanishq-garg-2007/FireDetect.git](https://github.com/Tanishq-garg-2007/FireDetect.git)
  - **Primary Branch:** `main`
  - **Commit Status:** All pipeline code, LoRA adapter weights, prompt engines, knowledge graphs, testing suites, evaluation benchmarks, research reports, and presentation slides are pushed and synced to GitHub.

---

## 2. Dataset Inventory, Provenance & Technical Facts

| Dataset Name | Local Archive Name | Raw Size | Total Samples | Format & Splits | Origin / Source | Key Role in Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Distilled Preference Dataset** | `merged_images.zip` | ~2.52 GB (Images) + 3.8 MB (JSON) | 1,981 images | 1,782 Train (90%) / 199 Test (10%), Seed 42 | Distilled from Gemma-4-31B-IT teacher annotations using 10 Gemini API keys in parallel | Fine-tuning student VLM (Florence-2-base / SmolVLM-256M) for joint scene description + tool detection. |
| **MIT Indoor Scenes (indoorCVPR_09)** | `archive.zip` | ~2.50 GB | 15,620 images | 67 Indoor Room Categories (`Images/`) | A. Quattoni & A. Torralba, *Recognizing Indoor Scenes*, CVPR 2009 | Baseline indoor scene diversity across rooms (kitchen, living room, office, corridor, etc.). |
| **Indoor Fire Detection (Roboflow)** | `Indoor Fire Detection.v1i.yolo26.zip` | 38.0 MB | 920 images | YOLOv8 / YOLO26 format (`train/`, `valid/`, `test/`) | Roboflow Universe (`https://universe.roboflow.com`) Exported Mar 19, 2026 | Bounding box ground truth for fire & flame detection in indoor settings. |
| **Indoor Fire 1000** | `indoor_fire_1000.zip` | 142.2 MB | 1,000 images | Standard JPEG/PNG images | Curated indoor fire scenes | Stress-testing perception under dense smoke, open flames, and low-light conditions. |
| **Disaster Benchmark Suite** | `images/` / `Testing results/fire/` | ~35 MB | 208 images | 100 disaster eval images (`image1.jpg`–`image107.jpeg`) + expanded set up to `image208.jpeg` | Curated evaluation set | Empirical end-to-end benchmarking against direct Gemma-4 and SmolVLM baselines. |

### Preference Dataset Split & Metadata Facts (`dataset_split_summary.json`)
- **Random Seed:** `42` (stratified train/test partition)
- **Total Samples:** 1,981 images
- **Training Partition:** 1,782 samples (90.0%) saved in `dataset_train.json`
- **Testing Partition:** 199 samples (10.0%) saved in `dataset_test.json`
- **Annotation Schema:**
  - `prompt`: *"Analyze this image comprehensively. Describe the atmosphere, setting, and activity in the scene, and list all objects present."*
  - `description`: Single cohesive paragraph (6–10 sentences) describing atmosphere, setting, state/condition of objects (open/closed, on/off, damaged/clean), with no speculative hallucination.
  - `objects`: Comma-separated list of distinct physical tools and furniture.
- **Teacher Distillation:** Generated via `gemma4_image_analyzer.py` executing over `google.genai` with automatic key rotation and interrupt-resilient checkpointing.

---

## 3. Fine-Tuned Model Weights & LoRA Checkpoints

All fine-tuned weights and configuration files are preserved in the GitHub repository under:
`Dataset/preference_dataset/preference_dataset/Fine Tune Model/`

- **Base Architecture:** `microsoft/Florence-2-base` (230M parameters)
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: `q_proj`, `v_proj`
- **Saved Files:**
  - `adapter_model.safetensors` (23.1 MB) — Trained LoRA adapter weights
  - `adapter_config.json` — HuggingFace PEFT configuration
  - `tokenizer.json` (3.5 MB) & `tokenizer_config.json` — Tokenizer vocab and configs
  - `processor_config.json` & `chat_template.jinja` — Multimodal vision-text processor definitions
- **Training Notebook:** `SmolVLM_256M_Colab_FineTuning.ipynb` & `create_notebook.py`

---

## 4. Pipeline Architecture Summary

The neuro-symbolic emergency assistance system executes a 6-stage cognitive loop:

1. **Step 1: Vision Perception (`Steps/Step_1_Vision.py`)**
   - Dual-turn prompting with CLAHE contrast enhancement and unsharp masking.
   - Extracts factual environment descriptions and candidate interactive tools.
2. **Step 2: Semantic & Ontology Mapping (`Steps/Step_2_Semantic.py`)**
   - Translates raw text into canonical ontology constants (`STATE_*` and `T_*`).
   - Uses zero-shot classification (`ModernBERT-large-zeroshot-v2.0`) and ChromaDB vector search (`all-MiniLM-L6-v2`).
3. **Step 3: Graph PathFinder (`Steps/Step_3_PathFinder.py`)**
   - Backward-chaining graph traversal over `Knowledge_Graph/knowledge_graph.graphml` (378 nodes, 865 edges).
   - Resolves: `Hazard State` $\leftarrow$ `Action` $\leftarrow$ `Affordance` $\leftarrow$ `Available Tool`.
   - Incorporates combustion-state gating (e.g., highly penalizing flammable items like `T_Sofa` for fire suppression).
4. **Step 4: Physical Feasibility Verification (`Steps/Step_4_Verification.py`)**
   - VLM spatial verification ensuring proposed tools are reachable and not engulfed in fire.
5. **Step 5: Feedback & Edge Penalty Loop (`Steps/Step_5_Feedback.py`)**
   - Dynamically penalizes infeasible paths and triggers re-routing until an optimal path is found.
6. **Step 6: SLM Instruction Generation (`Steps/Step_6_SLM.py`)**
   - Uses `Qwen` SLM to generate imperative, numbered life-safety egress actions.

---

## 5. Benchmark Results & LLM-as-a-Judge Evaluation

Benchmark outputs evaluated by Gemini Flash independent judge across 100 disaster scenes:
- **`Testing results/pipeline_batch_results.json`:** 100 end-to-end pipeline execution runs.
- **`Testing results/gamma_results.json`:** 100 raw Gemma-4 direct prompt runs.
- **`Testing results/evaluation_checkpoint.json`:** Multi-dimensional CoT judge evaluations.
- **Key Empirical Findings:**
  - End-to-end pipeline achieves **0% flammable shielding recommendations** (vs 12% in raw baseline).
  - Average latency on consumer GPU: **~2.8 seconds** end-to-end.
  - Zero hallucination of emergency equipment not physically present in the scene.

---

## 6. How to Re-clone and Run the Project

To clone and execute the project on any machine:
```bash
# 1. Clone repository
git clone https://github.com/Tanishq-garg-2007/FireDetect.git
cd FireDetect

# 2. Set up virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline on a test image
python main_pipeline.py --image images/image1.jpg

# 5. Reproduce benchmarks
python eval/evaluate_models.py
python eval/evaluate_pipeline.py
```
