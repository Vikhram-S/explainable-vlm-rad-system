# ExplainableVLM-Rad: Multi-Modal Scientific Reasoning System

## Overview

This repository presents **ExplainableVLM-Rad**, a multi-modal AI system for radiology report generation, designed with a **system-level perspective toward scientific reasoning and interpretability**.

The project integrates a vision encoder and biomedical language model into a **modular pipeline** capable of transforming medical images into structured clinical reports, while maintaining traceability and extensibility toward broader scientific AI systems.

---

## Live Model

Hugging Face Model:  
https://huggingface.co/Vikhram-S/mimic-vit-biogpt  

Deployed using the Hugging Face Transformers ecosystem with reproducible inference pipeline.

---

## System Architecture

The system is designed as a **multi-stage reasoning pipeline**:

1. **Perception Layer**
   - Vision Transformer (ViT) extracts visual features from radiological images

2. **Semantic Reasoning Layer**
   - BioGPT generates domain-specific clinical text from visual embeddings

3. **Structured Output Layer**
   - Produces organized outputs (findings, impressions)

4. **Explainability Layer**
   - Attention and gradient-based methods for interpretability

---

## Key Features

- Multi-modal vision-language integration  
- Structured clinical report generation  
- Explainability via attention and saliency  
- Deployment-ready inference via Hugging Face  
- Modular architecture for extensibility  

---

## Extension to Scientific AI Systems

The architecture is designed for **generalization beyond radiology**, and can be extended to scientific instrumentation workflows by:

- Integrating **Retrieval-Augmented Generation (RAG)** with domain knowledge bases  
- Adding **rule-based validation layers** for parameter constraints  
- Enabling **hybrid neuro-symbolic reasoning**  
- Supporting **offline deployment via model optimization and quantization**  

This aligns with the design of **AI-driven research intelligence systems**.

---

## Installation

```bash
pip install transformers torch pillow
```
## Minimal Inference Example

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("Vikhram-S/mimic-vit-biogpt")
processor = ViTImageProcessor.from_pretrained("Vikhram-S/mimic-vit-biogpt")
tokenizer = AutoTokenizer.from_pretrained("Vikhram-S/mimic-vit-biogpt")

image = Image.open("sample_xray.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

output_ids = model.generate(pixel_values, max_length=128)
report = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(report)
```
---
## Tech Stack

- **Python**  
- **Hugging Face Transformers**  
- **Vision Transformer (ViT)**  
- **BioGPT**  
- **PyTorch**  

---

## Limitations

- Trained on a **limited subset** of MIMIC-CXR  
- Focused on **single-view radiographs**  
- **Not clinically validated**  
- Prototype-scale system  

These limitations are explicitly acknowledged as part of a **responsible AI systems approach**, where understanding system boundaries is critical for high-stakes deployment.

---

## Ethical Considerations

- Potential **dataset bias** may influence outputs  
- Not intended for **clinical decision-making or diagnostic use**  
- Designed strictly for **research, experimentation, and system development purposes**  

The system emphasizes **interpretability, transparency, and controlled usage**, aligning with best practices for deploying AI in sensitive scientific domains.

---

## Author

**Vikhram S**  
AI Systems | Vision-Language Models | Scientific AI Infrastructure  

---

## Citation

**Vikhram S. (2026)**  
*ExplainableVLM-Rad: Multi-Modal Scientific Reasoning System for Radiology*
