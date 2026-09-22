---
title: Explainable Vlm Rad System
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-modal AI system for explainable radiology on IU X-Ray
---

# ExplainableVLM-Rad: Interactive Multimodal Radiology System

An interactive demonstration of **DenseNetBioGPT**, a multimodal vision-language architecture for automated chest radiograph interpretation and spatial visual attribution.

## Architecture
- **Vision Encoder:** TorchXRayVision DenseNet-121 (`densenet121-res224-all`) pretrained on clinical chest radiographs.
- **Multimodal Projector:** 2-layer MLP with GELU and LayerNorm mapping 1024-d visual features to the language decoder.
- **Language Decoder:** BioGPT (`microsoft/biogpt`) adapted with LoRA (PEFT) on attention projection matrices.
- **Explainability:** Spatial activation and cross-modal attention overlays highlighting anatomical evidence (lungs, cardiac silhouette, costophrenic angles).

## Data Governance & Compliance
This space is trained and evaluated strictly on the **Indiana University Chest X-Ray (IU X-Ray / OpenI)** public dataset released by the National Library of Medicine (NLM). **Zero credentialed data (e.g., MIMIC-CXR) or derived non-public weights are used.**

## Clinical Disclaimer
This system is an academic research demonstration developed under a Multimodal AI for Healthcare initiative. It is strictly intended for scientific exploration, explainability benchmarking, and algorithmic research. It is **not** certified for diagnostic, prognostic, or clinical decision-making.
