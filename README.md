---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-text
tags:
  - radiology
  - chest-xray
  - multimodal
  - vision-language
  - medical-imaging
  - biogpt
  - densenet
  - lora
  - peft
  - torchxrayvision
  - explainable-ai
datasets:
  - openi
metrics:
  - bleu
  - rouge
  - meteor
base_model:
  - microsoft/biogpt
---

# DenseNetBioGPT: Multimodal Chest Radiograph Report Generation

> **Data Governance Notice:** This model was rebuilt from the ground up using exclusively the
> publicly available [Indiana University Chest X-Ray (IU X-Ray / OpenI)](https://openi.nlm.nih.gov/)
> dataset released by the National Library of Medicine. **No MIMIC-CXR data, derivatives, or
> credentialed datasets are used anywhere in this codebase or in model weights.**  
> The repository slug (`mimic-vit-biogpt`) predates this rebuild; weights and training data are now
> 100% IU X-Ray (OpenI).

---

## Model Description

**DenseNetBioGPT** is a multimodal vision-language model for automated chest radiograph (CXR)
interpretation. It combines:

1. **Vision Encoder:** [TorchXRayVision](https://github.com/mlmed/torchxrayvision) DenseNet-121
   (`densenet121-res224-all`) pretrained on >100k clinical chest radiographs from six diverse
   sources. Early blocks (conv0, DenseBlocks 1–3) are frozen to preserve robust CXR-specific
   feature detectors; DenseBlock4 and the final BatchNorm layer are fine-tuned.
2. **Multimodal Projector:** A 2-layer MLP with GELU activation and LayerNorm mapping the 49
   spatial visual tokens (7×7 feature grid, 1024-dim) into the BioGPT embedding space (1024-dim).
3. **Language Decoder:** [BioGPT](https://huggingface.co/microsoft/biogpt) (`microsoft/biogpt`),
   adapted with [LoRA (PEFT)](https://github.com/huggingface/peft) on all four attention
   projection matrices (`q_proj`, `k_proj`, `v_proj`, `out_proj`), rank r=16, α=32.

Visual prefix tokens are prepended to the text embeddings in the decoder's attention sequence,
enabling every generated report token to attend to all 49 anatomical spatial patches. Spatial
attribution heatmaps are extracted from the activation energy of DenseNet features.

**Trainable Parameter Budget:**
- DenseBlock4 + Norm5 of DenseNet-121
- 2-layer MLP projector (≈2.1M parameters)
- LoRA adapter weights on BioGPT attention projections (≈3.1M parameters of 347M total)

---

## Intended Uses & Limitations

### Intended Uses
- Academic research in vision-language models for radiology.
- Benchmarking explainability and attention attribution methods on publicly available CXR datasets.
- Demonstration and educational purposes for multimodal AI system design.

### Out-of-Scope / Explicitly Prohibited
- Clinical decision-making, diagnostic support, or patient care of any kind.
- Medical diagnosis or triage under any regulatory or non-regulatory context.
- Deployment in any healthcare or clinical system without independent clinical validation by
  licensed medical professionals.

---

## Training Data

| Property | Details |
|---|---|
| **Dataset** | Indiana University Chest X-Ray (IU X-Ray / OpenI) |
| **Source** | [National Library of Medicine (NLM) Open-i Service](https://openi.nlm.nih.gov/) |
| **License** | Public Open Access (NLM Open-i terms; no credentialing required) |
| **Raw records** | 3,955 XML report files with associated radiograph images |
| **After filtering** | 3,337 usable studies (required: non-empty Findings section ≥ 10 chars and valid image identifier) |
| **Images** | 7,470 PNG chest radiographs (both frontal and lateral views provided) |
| **Avg findings length** | 31.4 words |

### Filtering Criteria (applied in `scripts/download_and_preprocess_openi.py`)
1. XML parsing of the official NLM OpenI release tarballs.
2. Mandatory non-empty FINDINGS section (≥ 10 characters).
3. Minimum one associated image identifier present.
4. Anonymization token normalization (`[ANON]` substitution for age/date values).
5. Patient-level split to prevent data contamination across train/val/test sets.

### Data Splits (seed=42, patient-level)
| Split | Studies |
|---|---|
| Train | 2,335 |
| Val | 333 |
| Test | 669 |

---

## Training Procedure

### Hardware
- **Training compute:** Google Colab free-tier (T4 GPU, 16 GB VRAM) / Kaggle Notebooks (T4/P100, 16 GB VRAM)
- **Checkpointing:** Every 100 optimizer steps to Google Drive / Kaggle output (fully resumable across session disconnects)

### Hyperparameters
| Parameter | Value |
|---|---|
| Micro-batch size | 4 |
| Gradient accumulation | 4 steps (effective batch size 16) |
| Optimizer | AdamW |
| LR – Projector & LoRA | 2e-4 |
| LR – DenseBlock4 (encoder fine-tune) | 2e-5 |
| Weight decay | 0.01 |
| LR schedule | Cosine with 5% linear warmup |
| Mixed precision | fp16 (torch.amp.autocast) |
| Gradient clipping | max_norm = 1.0 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, out_proj |
| Early stopping | Validation loss plateau monitoring |

### Training Objective
Standard causal language modeling cross-entropy loss. Visual prefix tokens (49 visual patches)
are masked with label=-100 so no loss is computed on the image prefix — loss is applied only
to the generated report tokens.

---

## Evaluation

> **Important:** The evaluation results table below will be populated **only after training and
> evaluation runs complete** on Colab/Kaggle hardware. Every number comes directly from a single
> run of `scripts/eval.py` outputting `results/metrics.json`. No numbers are hand-typed or estimated.

### Evaluation Protocol
- Script: `scripts/eval.py`
- Held-out test split: 669 IU X-Ray studies (patient-level, no overlap with training)
- BLEU-1/2/3/4: Sentence-level BLEU with Chen & Cherry smoothing (`nltk`)
- ROUGE-1/2/L: Rouge F-measure with stemming (`rouge_score`)
- METEOR: Unigram recall-precision harmonic mean with synonym matching (`nltk`)
- Corpus-level BLEU: `sacrebleu` for reporting consistency

### Results (post-training)

> *Metrics reported here will be injected from `results/metrics.json` after the training run completes. This section intentionally left blank to avoid any fabrication.*

---

## Usage (Verified Code Snippet)

After training completes and checkpoint is uploaded:

```python
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv
from transformers import BioGptTokenizer

# Load model from this repository (post-training)
# NOTE: The architecture below mirrors DenseNetBioGPT from src/model.py
# Replace 'path/to/checkpoint' with the saved checkpoint directory

from src.model import DenseNetBioGPT

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = DenseNetBioGPT(
    biogpt_model_name="microsoft/biogpt",
    xrv_weights="densenet121-res224-all",
    use_lora=True
)
model.load_checkpoint("path/to/checkpoint")
model.eval()

# Load and preprocess a CXR image (grayscale, xrv normalization)
img = Image.open("sample_cxr.png").convert("L").resize((224, 224))
np_arr = np.array(img, dtype=np.float32)
norm_arr = xrv.datasets.normalize(np_arr, maxval=255.0)
pixel_tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)

# Generate a clinical findings report
reports, _ = model.generate(pixel_values=pixel_tensor, tokenizer=tokenizer, max_new_tokens=128)
print(reports[0])
```

---

## Explainability

The system generates spatial attribution heatmaps by:
1. Extracting the 7×7 spatial feature map from DenseNet-121's final convolutional block.
2. Computing L2 activation energy across channels per spatial patch.
3. Bicubic upsampling from 7×7 to the original image resolution (224×224 or native).
4. Blending the normalized heatmap with the original CXR for anatomical localization.

The companion [Space demo](https://huggingface.co/spaces/Vikhram-S/explainable-vlm-rad-system)
shows this overlay interactively for uploaded chest radiographs.

---

## Bias, Risks & Limitations

- **Dataset Scope:** IU X-Ray is a single-institution dataset from Indiana University. It may not
  represent the full spectrum of CXR acquisition protocols, patient demographics, or disease
  prevalence seen globally.
- **Rare Pathologies:** With ~3,337 training studies, performance on rare or complex multi-system
  pathologies is expected to be limited.
- **Language Bias:** The model generates text in the reporting style of the IU X-Ray dataset.
  Radiology report style varies significantly across institutions.
- **Not Clinically Validated:** This system has not been evaluated by radiologists or compared
  against clinical standards of care.
- **Small Test Set:** With 669 test studies, confidence intervals on metric estimates are wide
  (~±2–4 BLEU points at 95% CI for BLEU-4). These limitations are stated explicitly here and
  must be conveyed in any presentation of results.

---

## Ethical Considerations

- The model is released for academic and non-clinical research purposes only.
- All training data (IU X-Ray / OpenI) is publicly available under NLM Open-i terms; patient
  information was de-identified by the original dataset creators.
- Attention overlays and saliency maps are visual approximations and do not constitute medically
  validated explanations.
- Deployment in any clinical or patient-facing context is explicitly out-of-scope and
  would require independent clinical validation, regulatory approval, and medical oversight.

---

## Repository Structure

```
Vikhram-S/mimic-vit-biogpt (model weights & card)
├── src/model.py              — DenseNetBioGPT architecture
├── src/dataset.py            — IUXRayDataset and DataLoader
├── scripts/
│   ├── download_and_preprocess_openi.py  — OpenI data download & preprocessing
│   ├── train.py              — Training script (LoRA, fp16, checkpointing)
│   ├── eval.py               — Reproducible evaluation (BLEU/ROUGE/METEOR)
│   └── generate_notebook.py  — Colab training notebook generator
├── notebooks/
│   └── Train_IU_XRay_Colab.ipynb  — Ready-to-run Colab notebook
├── tests/test_model_and_dataset.py — Unit + smoke tests
├── requirements.txt
└── results/metrics.json      — Evaluation output (populated post-training)
```

---

## Citation

If you use this model or the associated codebase in your research, please cite both this
repository and the original IU X-Ray dataset:

```
@misc{vikhrams2026densenetbiogpt,
  author    = {Vikhram S},
  title     = {DenseNetBioGPT: Multimodal Chest Radiograph Report Generation via TorchXRayVision + BioGPT with LoRA},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/Vikhram-S/mimic-vit-biogpt}
}

@article{demner2016preparing,
  title     = {Preparing a collection of radiology examinations for distribution and retrieval},
  author    = {Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal   = {Journal of the American Medical Informatics Association},
  volume    = {23},
  number    = {2},
  pages     = {304--310},
  year      = {2016},
  publisher = {Oxford University Press}
}
```
