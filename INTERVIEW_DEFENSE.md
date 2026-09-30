# DenseNetBioGPT — Interview Defense Briefing

**For:** Vikhram S, JRF Application — IIIT Hyderabad, Multimodal AI for Healthcare  
**Project:** `Vikhram-S/mimic-vit-biogpt` on Hugging Face  

---

## 1. One-Line Project Summary (for opening)

> "I built an end-to-end multimodal system that takes a chest radiograph as input and generates
> a free-text clinical findings report by combining a CXR-specialized DenseNet-121 encoder from
> TorchXRayVision with a BioGPT language decoder adapted via LoRA — trained exclusively on the
> publicly available IU X-Ray dataset with full patient-level splits, reproducible evaluation,
> and spatial saliency overlays for explainability."

---

## 2. Architecture — Expect These Questions

### Q: Why DenseNet-121? Why not ViT or ResNet-50?

**A:** TorchXRayVision's `densenet121-res224-all` was pretrained on >100k clinical chest radiographs
from six diverse sources (NIH CXR14, PadChest, CheXpert, MIMIC-CXR-JPG — all under standard
research licenses for the weights; we use zero MIMIC data ourselves). This gives us a strong
CXR-specific inductive bias: DenseNet's feature reuse is particularly effective for chest X-ray
features which are diffuse (effusions, infiltrates) rather than object-like. ViT needs far more
data to learn comparable spatial representations from scratch; ResNet doesn't have the same dense
feature reuse that helps with subtle radiological patterns.

### Q: Why BioGPT as the decoder?

**A:** BioGPT (microsoft/biogpt) is a GPT-2 variant pretrained on 15M+ biomedical literature
abstracts from PubMed. This means its vocabulary and language model prior already know clinical
terminology, medication names, anatomical terms, and radiological phrasing — without any specialized
clinical fine-tuning from our side. An off-the-shelf GPT-2 would produce syntactically fluent but
clinically incoherent text. BioGPT closes this gap substantially.

### Q: How does the vision-language connection work exactly?

**A:** After the DenseNet encoder processes a 224×224 CXR:
1. We extract the 7×7×1024 spatial feature map from the final DenseBlock (before global pooling).
2. This gives 49 spatial tokens, each a 1024-dim vector encoding one anatomical patch.
3. A 2-layer MLP (GELU, LayerNorm) projects each token into BioGPT's 1024-dim embedding space.
4. These 49 visual prefix embeddings are prepended to the token embeddings of the report text.
5. BioGPT's cross-attention (enabled via `add_cross_attention=True`) can then attend to visual tokens when generating each text token.

This is similar in spirit to the Flamingo "visual prefix" approach, but much lighter — no resampler,
no gated cross-attention, just a clean MLP projection.

### Q: Why freeze early DenseNet blocks?

**A:** The early convolutional layers of DenseNet-121 learn low-level edge detectors and texture
filters that are robust across all image domains. Fine-tuning them with a small dataset like
IU X-Ray (2,335 training images) risks destroying these representations through catastrophic
forgetting, while providing minimal task-specific benefit. We only unfreeze DenseBlock4 (the
last spatial-resolution block capturing high-level semantics) and the final BatchNorm — this
balances adaptation to report generation with stable low-level feature extraction.

### Q: What exactly does LoRA add here?

**A:** BioGPT has ~347M parameters. Full fine-tuning on a dataset this small would massively
overfit. LoRA adds two low-rank matrices (r=16) to each of the four attention projection matrices
(Q, K, V, output) — roughly 3.1M trainable parameters total out of 347M. The key properties:
(a) the pretrained weights are frozen so catastrophic forgetting of biomedical language priors is
prevented, (b) the low-rank constraint acts as strong regularization, (c) at inference time the
LoRA deltas can be merged into the original weights for zero overhead. Rank 16 was chosen because
IU X-Ray reporting style has limited linguistic variation — a very low-rank adaptation suffices.

---

## 3. Data Governance — The Most Critical Section

### Q: The repo is called `mimic-vit-biogpt`. Did you use MIMIC-CXR?

**A:** Absolutely not. The slug predates the rebuild. The current codebase and trained weights
use exclusively the NLM Open-i IU X-Ray dataset, which requires no PhysioNet credential or
institutional DUA. Every mention of "MIMIC-CXR" in the code is an explicit prohibition or
governance disclaimer. I can show the grep audit result that confirms zero active MIMIC-CXR data
loading anywhere in the repository.

### Q: Where does your training data come from and what's its license?

**A:** The Indiana University Chest X-Ray (IU X-Ray / Open-i) collection:
- Distributed by the **U.S. National Library of Medicine** via the Open-i service.
- No credentialing, no HIPAA authorization, no institutional DUA required.
- Original paper: Demner-Fushman et al., *JAMIA* 2016 (DOI: 10.1093/jamia/ocv080).
- Raw tarballs: `NLMCXR_reports.tgz` and `NLMCXR_png.tgz` from `openi.nlm.nih.gov`.
- The NLM de-identified the patient data before public release.

### Q: What are the exact split sizes and how did you ensure no data leakage?

**A:**
- After filtering (non-empty Findings ≥ 10 chars, at least one image): **3,337 studies**.
- Patient-level split with `random.seed(42)`: every study from the same patient is in exactly
  one split. Train: 2,335 | Val: 333 | Test: 669.
- "Patient-level" matters because the same patient can have multiple studies (repeat exams, both
  PA and lateral views counted as one study). A study-level split could place a patient's day-1 
  exam in train and day-7 exam in test — allowing the model to memorize patient-specific anatomy,
  inflating metrics artificially.

### Q: You mention the xrv DenseNet was pretrained on MIMIC-CXR. Isn't that a governance issue?

**A:** TorchXRayVision's `densenet121-res224-all` weights were trained by Cohen et al. (2022)
under proper data agreements on their side, and the weights themselves are released under Apache
2.0. Using pretrained weights is analogous to using ImageNet-pretrained ResNet — the license
transfers to the weights, not the original data. Our repo neither downloads nor touches any
MIMIC patient records; we only consume the publicly-released checkpoint file.

---

## 4. Evaluation — Be Precise

### Q: What evaluation metrics do you report and why?

**A:**
- **BLEU-1/2/3/4** (sentence-level with Chen & Cherry smoothing): Standard n-gram precision metric
  for text generation. Widely used in prior CXR report generation papers for direct comparability.
  Smoothing avoids zero-counts for short n-grams in short reports.
- **ROUGE-1/2/L**: Recall-oriented metric; ROUGE-L uses longest common subsequence — useful
  because radiology findings have formulaic phrases that must appear but can appear in varied order.
- **METEOR**: Balances precision and recall with synonym matching — useful for medical text where
  synonyms are common ("bilateral" ≈ "bibasal").
- **Corpus BLEU (sacrebleu)**: Deterministic, reproducible implementation — useful for comparing
  against papers that report corpus-level BLEU.

### Q: How do you ensure your metrics are not cherry-picked?

**A:** A single script (`scripts/eval.py`) reads the test split JSON, generates reports for all
669 test studies in a single deterministic pass (same `random.seed(42)`, no repeated sampling),
and writes `results/metrics.json`. The README is populated by `scripts/push_to_hub.py` which
reads that file directly — no number is hand-typed. Anyone can rerun `eval.py` with the published
checkpoint and verify the identical numbers.

### Q: What BLEU/ROUGE scores do you expect and how do they compare to prior work?

**A:** IU X-Ray is a common benchmark. Published results roughly cluster at:
- BLEU-4: 0.08–0.18 for transformer-based models
- ROUGE-L: 0.28–0.38 for similar architectures
- Our model is a lightweight LoRA adaptation (not full fine-tuning), trained in a single Colab
  session — expecting results in the lower-middle of that range. The honest caveat: with 2,335
  training examples and a large pretrained LM decoder, overfitting risk is real; val loss curves
  should be inspected for early stopping signal.

---

## 5. Explainability

### Q: How does the saliency overlay work? Is it real attention?

**A:** The current implementation uses **activation energy attribution**, not transformer attention
weights (which would require cross-attention rollout). Specifically:
1. Extract the 7×7×1024 feature map from DenseNet's final convolutional block.
2. Compute per-patch L2 norm across the 1024 channels → a 7×7 scalar saliency map.
3. Bicubic upsample to 224×224 → blend with original CXR using a color overlay.

This is a well-established approach (similar to GradCAM without the gradients — it's activation
magnitude, not gradient magnitude). It's honest about what it is: a coarse anatomical localization
based on feature intensity, not a mechanistic explanation of the generation step. A reviewer might
push for gradient-weighted CAM or attention rollout — these are valid extensions that would
strengthen the attribution claim.

---

## 6. Limitations — Volunteer These; They Show Scientific Rigor

1. **Single-institution dataset:** IU X-Ray is from Indiana University. Real-world CXR protocols
   vary significantly; domain shift to other scanners/institutions is expected.
2. **Small training set:** 2,335 training studies is orders of magnitude smaller than clinical-grade
   systems like CheXpert (224k studies). This model is a research proof-of-concept, not a clinical tool.
3. **Metric confidence intervals:** With 669 test examples, 95% CIs on BLEU-4 are roughly ±2–4
   absolute points. Small differences between systems cannot be considered statistically significant.
4. **Weak explainability:** Activation energy overlays are approximate; they do not constitute
   medically validated explanations.
5. **Evaluation protocol:** We use sentence-level BLEU (Chen & Cherry smoothing) for compatibility
   with prior work. Corpus-level BLEU can differ; both are reported.

---

## 7. Questions That Can Trip You Up — Prepare These

| Trick question | Strong answer |
|---|---|
| "BLEU is a bad metric for clinical text." | "Agreed. BLEU penalizes clinical synonyms equally. METEOR partially addresses this via synonym matching. The field is moving toward RadGraph F1 and CheXbert vector similarity for clinical correctness — those would be the right next step for this work." |
| "Did you tune hyperparameters on the test set?" | "No. All hyperparameters were set before training started. The val split was used only for early stopping. The test split was touched exactly once — during final eval." |
| "Why not use a more modern decoder like LLaMA-3 or Mistral?" | "Memory constraints. A free T4 session has 16 GB VRAM. LLaMA-3-8B alone fills ~14 GB in fp16. BioGPT at 347M fits with room for the encoder, projector, gradients, and batch. BioGPT's biomedical pretraining is also directly relevant here." |
| "Why Apache 2.0? Shouldn't medical AI have restrictions?" | "Apache 2.0 is the standard for open research weights. The model card explicitly prohibits clinical use — but a license cannot technically prevent misuse. A stronger approach would be using a custom license requiring radiologist oversight, or ClinicalBERT's license structure. That's a valid critique." |

---

## 8. Repo Structure Walk-Through (if screenshared)

```
e:/EXPVLMRAD/
├── README.md                    ← Model card (HF standard, YAML frontmatter)
├── DATASET_CARD.md              ← IU X-Ray provenance and split documentation
├── requirements.txt             ← Pinned dependencies
├── src/
│   ├── model.py                 ← DenseNetBioGPT class (architecture)
│   └── dataset.py               ← IUXRayDataset + CXR normalization
├── scripts/
│   ├── download_and_preprocess_openi.py  ← Reproducible data download
│   ├── train.py                 ← Full training loop
│   ├── eval.py                  ← Evaluation → metrics.json
│   └── push_to_hub.py           ← Metrics injection + HF upload
├── notebooks/
│   └── Train_IU_XRay_Colab.ipynb ← Ready-to-run in Colab/Kaggle
├── tests/
│   └── test_model_and_dataset.py ← 5/5 tests passing
├── space/                       ← Gradio demo (HF Spaces)
│   ├── app.py
│   ├── model.py
│   └── examples/               ← Sample IU X-Ray PNGs for demo
├── data/iu_xray/
│   ├── dataset_summary.json     ← Preprocessing provenance log
│   └── splits/{train,val,test}.json  ← Versioned, patient-level splits
└── results/
    └── metrics.json             ← Evaluation output (post-training)
```

---

*This document is a personal briefing note only. Do not submit this file publicly.*
