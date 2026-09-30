"""
space/app.py

ExplainableVLM-Rad: Multi-Modal Chest Radiograph Interpretation System.
Architecture: TorchXRayVision DenseNet-121 (CXR-pretrained) + BioGPT Decoder with LoRA (PEFT).
Dataset Provenance: Trained and evaluated exclusively on public Indiana University Chest X-Ray (OpenI).
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gradio as gr
from transformers import BioGptTokenizer, BioGptConfig

import torchxrayvision as xrv
import importlib.util

# Explicitly load local model.py to avoid name conflict with torchxrayvision's internal model package
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py")
_spec = importlib.util.spec_from_file_location("local_model", _model_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DenseNetBioGPT = _mod.DenseNetBioGPT


# Global Model & Tokenizer Initializer (CPU-compatible)
print("[Space Init] Loading Tokenizer and Model on CPU...")
DEVICE = torch.device("cpu")
MODEL_NAME = "microsoft/biogpt"

USE_MOCK = os.environ.get("MOCK_TEST") == "1"

if USE_MOCK:
    print("[Space Init] MOCK_TEST=1 enabled: using lightweight architecture for verification.")
    tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAME) if os.path.exists(MODEL_NAME) else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        # Fallback to standard fast tokenizer if offline
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    decoder_cfg = BioGptConfig(vocab_size=len(tokenizer) if tokenizer else 1000, hidden_size=256, num_attention_heads=4, num_hidden_layers=2, intermediate_size=512)
    model = DenseNetBioGPT(
        biogpt_model_name=MODEL_NAME,
        xrv_weights="densenet121-res224-all",
        freeze_encoder_blocks=3,
        use_lora=True,
        load_pretrained_weights=False,
        decoder_config=decoder_cfg
    )
else:
    tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAME)
    # Initialize model
    model = DenseNetBioGPT(
        biogpt_model_name=MODEL_NAME,
        xrv_weights="densenet121-res224-all",
        freeze_encoder_blocks=3,
        use_lora=True,
        load_pretrained_weights=True
    )

# Load checkpoint if available in space
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
if os.path.exists(checkpoint_dir):
    try:
        model.load_checkpoint(checkpoint_dir, device="cpu")
        print(f"[Space Init] Loaded fine-tuned weights from {checkpoint_dir}")
    except Exception as e:
        print(f"[Space Init] Warning: could not load checkpoint ({e}). Using base initialization.")

model.to(DEVICE)
model.eval()
print("[Space Init] Ready for inference!")


def compute_saliency_overlay(image_pil: Image.Image, pixel_tensor: torch.Tensor, model: DenseNetBioGPT) -> Image.Image:
    """
    Computes spatial activation heatmap on DenseNet121 features and overlays it on the CXR.
    """
    with torch.no_grad():
        # Extract 7x7 spatial feature map: (1, 1024, 7, 7)
        features = model.vision_encoder.features(pixel_tensor)
        # Activation energy per spatial patch (L2 norm across channels)
        act_map = torch.norm(features, dim=1, keepdim=True)  # (1, 1, 7, 7)
        # Upsample to original image resolution
        orig_w, orig_h = image_pil.size
        upsampled = F.interpolate(act_map, size=(orig_h, orig_w), mode="bicubic", align_corners=False)
        heatmap = upsampled.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert grayscale original to RGB numpy
    orig_np = np.array(image_pil.convert("RGB")) / 255.0

    # Apply colormap (turbo / jet)
    colormap = cm.get_cmap("turbo")
    colored_heatmap = colormap(heatmap)[:, :, :3]  # Drop alpha

    # Blend original CXR with heatmap
    alpha = 0.42
    blended = (1 - alpha) * orig_np + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def analyze_xray(input_image: Image.Image, min_length: int = 15, max_length: int = 128, num_beams: int = 3):
    """
    Main inference pipeline for chest radiograph interpretation and attribution.
    """
    if input_image is None:
        return None, "Please upload a chest radiograph image to analyze.", ""

    # Preprocess image
    gray_img = input_image.convert("L")
    resized_gray = gray_img.resize((224, 224), Image.Resampling.BILINEAR)
    np_arr = np.array(resized_gray, dtype=np.float32)
    norm_arr = xrv.datasets.normalize(np_arr, maxval=255.0)
    pixel_tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 224, 224)

    # Generate clinical report
    with torch.no_grad():
        reports, attentions = model.generate(
            pixel_values=pixel_tensor,
            tokenizer=tokenizer,
            min_length=int(min_length),
            max_new_tokens=int(max_length),
            num_beams=int(num_beams),
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    generated_report = reports[0] if reports else "No findings generated."

    # Compute spatial attribution heatmap overlay
    overlay_image = compute_saliency_overlay(input_image, pixel_tensor, model)

    # Build technical audit metadata
    metadata_text = f"""### Technical Audit Information
- **Vision Backbone:** TorchXRayVision DenseNet-121 (`densenet121-res224-all`)
- **Visual Spatial Grid:** 7x7 spatial tokens (49 visual tokens, 1024-dim)
- **Language Decoder:** BioGPT (`microsoft/biogpt`) via LoRA (PEFT, rank=16)
- **Primary Training Benchmark:** Indiana University Chest X-Ray (OpenI)
- **Data Governance Policy:** Strictly public NLM Open-i data only; 0% MIMIC-CXR.
- **Compute:** CPU Inference (~1-3s execution time).

> **Clinical Disclaimer:** This system is an academic research prototype developed under a Multimodal AI for Healthcare research initiative. It is strictly intended for scientific exploration, explainability benchmarking, and research reproduction. It is not approved for clinical diagnostics or patient care.
"""

    return overlay_image, generated_report, metadata_text


# Build Gradio Interface
example_images = []
examples_dir = os.path.join(os.path.dirname(__file__), "examples")
if os.path.exists(examples_dir):
    for f in ["sample_normal_cxr.png", "sample_lateral_cxr.png"]:
        p = os.path.join(examples_dir, f)
        if os.path.exists(p):
            example_images.append([p, 15, 100, 3])

custom_css = """
.gradio-container { font-family: 'Inter', -apple-system, sans-serif; }
.output-textbox { font-size: 1.05rem; line-height: 1.6; }
"""

with gr.Blocks(title="ExplainableVLM-Rad", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🫁 ExplainableVLM-Rad: Multi-Modal Chest Radiograph Interpretation System
        **DenseNet-121 (TorchXRayVision) + BioGPT Decoder with LoRA & Spatial Saliency Attribution**  
        *Trained & Evaluated on NLM Open-i Indiana University Chest X-Ray (IU X-Ray)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Chest Radiograph (CXR)")
            with gr.Accordion("Generation Parameters", open=False):
                min_len_slider = gr.Slider(minimum=5, maximum=50, value=15, step=5, label="Minimum Report Length")
                max_len_slider = gr.Slider(minimum=30, maximum=160, value=100, step=10, label="Maximum Report Length")
                beams_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Beam Search Width")
            submit_btn = gr.Button("🔍 Analyze Radiograph & Generate Report", variant="primary")

        with gr.Column(scale=1):
            saliency_img = gr.Image(type="pil", label="Spatial Attention / Saliency Overlay")
            report_out = gr.Textbox(label="Generated Clinical Findings", lines=5, elem_classes=["output-textbox"])

    audit_box = gr.Markdown()

    if example_images:
        gr.Examples(
            examples=example_images,
            inputs=[input_img, min_len_slider, max_len_slider, beams_slider],
            outputs=[saliency_img, report_out, audit_box],
            fn=analyze_xray,
            cache_examples=False,
            label="Sample IU X-Ray Radiographs (NLM Open-i)"
        )

    submit_btn.click(
        fn=analyze_xray,
        inputs=[input_img, min_len_slider, max_len_slider, beams_slider],
        outputs=[saliency_img, report_out, audit_box]
    )

if __name__ == "__main__":
    demo.launch()
