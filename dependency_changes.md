# Dependency Changes Log

## 2026-07-21 — Upgraded to support Qwen3.5-4B

**Reason:** `Qwen/Qwen3.5-4B` uses model type `qwen3_5` which is not recognized by transformers 4.57.x.

**Before:**
```
transformers==4.57.6
huggingface-hub==0.36.2
safetensors==0.7.0
```

**After:**
```
transformers==5.14.1
huggingface-hub==1.24.0
safetensors==0.8.0
```

**To roll back:**
```bash
pip install transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.7.0
```

## 2026-07-22 — Installed flash-linear-attention for Qwen3.5 speed

**Reason:** Qwen3.5-4B uses Gated DeltaNet (linear attention) layers that fall back to slow O(n²) PyTorch without the `fla` kernel, causing ~32s/step vs ~6s/step.

**Installed:**
```
flash-linear-attention==0.5.1
fla-core==0.5.1
cuda-nvcc-12-4  (via apt, needed to compile CUDA kernels)
```

**To uninstall:**
```bash
pip uninstall flash-linear-attention fla-core
```

---

**Known side effects:**
- The peft/transformers `EmbeddingParallel` ImportError that required a monkey-patch in `scripts/sft_pipeline.py` was caused by peft 0.19.x expecting a symbol introduced in a newer transformers. With transformers 5.14.1 the symbol should exist, so the monkey-patch becomes a no-op.
- vLLM compatibility: if vLLM was pinned against transformers 4.x, there may be import issues. Test with `python -c "from scripts.generators import VllmQwenGenerator"` before running inference.
