"""
scripts/generate_notebook.py

Generates notebooks/Train_IU_XRay_Colab.ipynb
"""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DenseNetBioGPT: End-to-End Training & Evaluation on IU X-Ray (OpenI)\n",
            "\n",
            "This notebook runs the complete, reproducible training and evaluation pipeline for **DenseNetBioGPT**:\n",
            "- **Vision Encoder:** TorchXRayVision DenseNet121 (`densenet121-res224-all`), with early layers frozen and DenseBlock4 fine-tuned.\n",
            "- **Projector:** 2-layer MLP with GELU + LayerNorm mapping 1024-d visual features to BioGPT embedding space.\n",
            "- **Language Decoder:** BioGPT (`microsoft/biogpt`) adapted using PEFT LoRA.\n",
            "- **Primary Dataset:** Indiana University Chest X-Ray (OpenI) - 100% public, open-access, zero credentialing required.\n",
            "- **Strict Data Governance:** Prohibits MIMIC-CXR and any non-credentialed derivatives.\n",
            "- **Compute Constraint:** Designed for free-tier Google Colab (T4 16GB) or Kaggle (P100/T4 16GB) with frequent Drive checkpointing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment & Hardware Verification\n",
            "!nvidia-smi\n",
            "import torch\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU Device:', torch.cuda.get_device_name(0))\n",
            "    print('Total VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Google Drive Persistent Checkpoint Mounting (Colab Only)\n",
            "import os\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    CHECKPOINT_DIR = '/content/drive/MyDrive/mimic_vit_biogpt_checkpoints'\n",
            "    os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
            "    print('Persistent Google Drive checkpoint directory mounted at:', CHECKPOINT_DIR)\n",
            "except Exception:\n",
            "    print('Running locally or on Kaggle. Using local checkpoint directory.')\n",
            "    CHECKPOINT_DIR = 'checkpoints/densenet_biogpt'\n",
            "    os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Install Dependencies\n",
            "!pip install -q torchxrayvision transformers peft sacrebleu rouge_score nltk\n",
            "import nltk\n",
            "nltk.download('wordnet', quiet=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Download and Preprocess IU X-Ray (OpenI) Dataset\n",
            "# Automatically downloads official NLM OpenI reports XML and images archive,\n",
            "# extracts findings/impressions, filters empty records, and creates 70/10/20 patient-level split.\n",
            "!python scripts/download_and_preprocess_openi.py --output_dir data/iu_xray --download_images\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Launch Training with LoRA, Mixed Precision, and Step Checkpointing\n",
            "# Micro-batch size: 4, Gradient accumulation: 4 -> Effective batch size: 16\n",
            "# Checkpoint saved every 100 steps to Google Drive so progress is never lost across session boundaries.\n",
            "# To resume after a disconnect, simply add: --resume_from {CHECKPOINT_DIR}/best_checkpoint\n",
            "!python scripts/train.py \\\n",
            "    --data_dir data/iu_xray \\\n",
            "    --output_dir \"$CHECKPOINT_DIR\" \\\n",
            "    --batch_size 4 \\\n",
            "    --grad_accum_steps 4 \\\n",
            "    --epochs 10 \\\n",
            "    --lr_projector 2e-4 \\\n",
            "    --lr_encoder 2e-5 \\\n",
            "    --lora_r 16 \\\n",
            "    --lora_alpha 32 \\\n",
            "    --fp16 \\\n",
            "    --save_steps 100 \\\n",
            "    --eval_steps 100\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Evaluate Best Checkpoint on Held-Out IU X-Ray Test Split\n",
            "# Computes BLEU-1..4, ROUGE-1/2/L, METEOR using standard benchmark libraries\n",
            "!python scripts/eval.py \\\n",
            "    --checkpoint_dir \"$CHECKPOINT_DIR/best_checkpoint\" \\\n",
            "    --split_file data/iu_xray/splits/test.json \\\n",
            "    --images_dir data/iu_xray/images \\\n",
            "    --output_file results/metrics.json \\\n",
            "    --batch_size 8\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Display Generated Evaluation Results\n",
            "import json\n",
            "if os.path.exists('results/metrics.json'):\n",
            "    with open('results/metrics.json', 'r') as f:\n",
            "        results = json.load(f)\n",
            "    print('Evaluation Results for Held-Out Test Split:')\n",
            "    for k, v in results.get('metrics', {}).items():\n",
            "        print(f'  {k}: {v}')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/Train_IU_XRay_Colab.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Colab training notebook generated at {out_path}")
