# FitCoach 🏋️

A fine-tuned LLM fitness and nutrition coach that conducts a conversational intake — one question at a time — then generates a structured, personalized meal or workout plan.

**🔗 Live demo:** [huggingface.co/spaces/Harsh-k-007/fitcoach](https://huggingface.co/spaces/Harsh-k-007/fitcoach)

| Asset | Link |
|---|---|
| 8B LoRA adapter | [Harsh-k-007/fitcoach-8b-adapter](https://huggingface.co/Harsh-k-007/fitcoach-8b-adapter) |
| 3B merged model | [Harsh-k-007/fitcoach-3b](https://huggingface.co/Harsh-k-007/fitcoach-3b) |
| Dataset | [Harsh-k-007/fitcoach-conversations](https://huggingface.co/datasets/Harsh-k-007/fitcoach-conversations) |

## What it does

FitCoach behaves like a real coach: instead of dumping a generic plan, it asks intake questions (goals, experience, equipment, dietary preferences, schedule) one at a time, then produces a structured plan. Scope is deliberately narrow — meal plans (~60%) and workout plans (~40%). Injuries, health conditions, and medical advice are explicitly out of scope and the model declines them.

## How it was built

### 1. Synthetic data generation (`data/`)

- **1,407 unique multi-turn conversations** generated with **Llama 3.3 70B Instruct Turbo** via the DeepInfra API
- Diverse user profiles pre-generated across all axes (age, goal, experience, diet, equipment, cooperation style) — exactly one conversation per profile to prevent repetition bias
- 8 opener styles × 7 user cooperation styles, ~12% of conversations include scenario hints
- Near-duplicate filtering with `all-MiniLM-L6-v2` embeddings (0.92 cosine threshold)
- Resumable JSONL pipeline with per-record retry budgets and raw-failure logging for diagnosis
- Stratified train/val/test splits: **1,301 / 80 / 26** (~1M tokens total)

### 2. Fine-tuning (`training/`)

Two models trained with **Unsloth `FastLanguageModel` + QLoRA** on a free Google Colab T4:

| | FitCoach 3B | FitCoach 8B |
|---|---|---|
| Base | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| LoRA | r=16, α=16, dropout=0, 7 target modules | same |
| Seq length | 2048 | 1536 |
| Batch | 2 × 4 grad accum | 1 × 8 grad accum |
| Released as | merged 16-bit model | LoRA adapter |

Both runs use `adamw_8bit`, cosine schedule, and packing (`bfd`). Training is fully resumable with Google Drive checkpointing to survive free-tier Colab session limits.

### 3. Deployment (`app/`)

Gradio app on **Hugging Face Spaces with ZeroGPU**:

- Dual-model toggle (8B default, 3B for comparison), streaming responses via `TextIteratorStreamer`
- The 8B adapter loads on the **full-precision** `unsloth/Meta-Llama-3.1-8B-Instruct` base — QLoRA adapters are compatible with the full-precision version of the same architecture, which eliminates bitsandbytes entirely at inference
- ZeroGPU pattern: both models loaded fp16 on CPU at startup, moved to CUDA inside `@spaces.GPU`, moved back + cache cleared after generation
- `demo.queue()` turns hard GPU-quota rejections into graceful queue positions

## Known limitations (v1)

- Macros aren't generated unprompted — a dataset gap planned for the v2 training run
- Macro arithmetic can be inaccurate; v2 will route calculations through a tool/calculator layer instead of the LLM
- Loss is computed on full conversations (assistant-only masking pending TRL support for the Llama 3 chat template)

## Repo structure

```
fitcoach/
├── data/        # synthetic data generation notebook + sample records
├── training/    # fine-tuning notebooks (3B and 8B) + Hub push
├── app/         # Gradio app deployed to HF Spaces
└── docs/        # sample run logs
```

## Running the app locally

```bash
cd app
pip install -r requirements.txt
python app.py
```

Requires a CUDA GPU with ~20GB VRAM for the 8B model (or set `ENABLE_8B = False` to run the 3B only).

## License

MIT — see [LICENSE](LICENSE). Fine-tuned models inherit the Llama 3.1 / 3.2 community licenses of their respective base models.
