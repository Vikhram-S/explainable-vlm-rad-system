"""
src/model.py

DenseNetBioGPT: Multimodal Chest Radiograph Interpretation Model.
Combines a TorchXRayVision DenseNet121 vision encoder (pretrained on clinical CXRs)
with a BioGPT language decoder adapted via Low-Rank Adaptation (LoRA / PEFT).

Architecture Overview:
1. Vision Encoder:
   - TorchXRayVision DenseNet-121 (`densenet121-res224-all`).
   - Early feature extraction layers (conv0 through transition3) frozen to preserve robust
     radiological feature detectors.
   - DenseBlock4 and final Norm5 unfrozen for domain-specific fine-tuning.
   - Extracts 7x7 spatial feature map -> 49 visual tokens of dimension 1024.
2. Multimodal Projector:
   - 2-layer MLP with GELU activation and LayerNorm: maps 1024-d visual features
     into BioGPT's 1024-d embedding space.
3. Language Decoder:
   - BioGPT (microsoft/biogpt) adapted via PEFT LoRA on projection matrices
     (q_proj, k_proj, v_proj, out_proj).
   - Visual tokens are prepended as multimodal prefix tokens to text embeddings.
   - Full causal attention allows every generated report token to attend directly
     to all 49 anatomical visual patches.
   - True spatial attribution heatmaps extracted from cross-modal attention weights.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import BioGptConfig, BioGptForCausalLM, BioGptTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torchxrayvision as xrv


class MultimodalVisionProjector(nn.Module):
    """
    Projects 2D spatial feature tokens from DenseNet121 (1024-dim)
    into BioGPT's token embedding space (1024-dim).
    """
    def __init__(self, in_features: int = 1024, out_features: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_patches, in_features)
        return self.proj(x)


class DenseNetBioGPT(nn.Module):
    """
    End-to-end Multimodal Vision-Language Model for Chest X-Ray report generation.
    """
    def __init__(
        self,
        biogpt_model_name: str = "microsoft/biogpt",
        xrv_weights: str = "densenet121-res224-all",
        freeze_encoder_blocks: int = 3,  # freeze denseblocks 1, 2, 3; fine-tune 4
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        load_pretrained_weights: bool = True,
        decoder_config: Optional[BioGptConfig] = None
    ):
        super().__init__()
        self.biogpt_model_name = biogpt_model_name
        self.num_visual_tokens = 49  # 7x7 spatial grid for 224x224 input
        self.visual_dim = 1024

        # 1. Vision Encoder (TorchXRayVision DenseNet121)
        if load_pretrained_weights:
            self.vision_encoder = xrv.models.DenseNet(weights=xrv_weights)
        else:
            self.vision_encoder = xrv.models.DenseNet(weights=None)

        # Freeze early layers as configured
        self._setup_encoder_freezing(freeze_encoder_blocks)

        # 2. Vision Projector
        decoder_hidden_size = 1024 if decoder_config is None else decoder_config.hidden_size
        self.vision_projector = MultimodalVisionProjector(
            in_features=self.visual_dim,
            out_features=decoder_hidden_size
        )

        # 3. BioGPT Decoder
        if decoder_config is not None:
            base_decoder = BioGptForCausalLM(decoder_config)
        elif load_pretrained_weights:
            base_decoder = BioGptForCausalLM.from_pretrained(biogpt_model_name)
        else:
            cfg = BioGptConfig.from_pretrained(biogpt_model_name)
            base_decoder = BioGptForCausalLM(cfg)

        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.decoder = get_peft_model(base_decoder, peft_config)
        else:
            self.decoder = base_decoder

    def _setup_encoder_freezing(self, freeze_blocks: int):
        """Freeze conv0, transition layers, and initial dense blocks."""
        features = self.vision_encoder.features
        # Always freeze initial stem
        for name, param in features.named_parameters():
            param.requires_grad = True

        freeze_prefixes = ["conv0", "norm0"]
        for b in range(1, freeze_blocks + 1):
            freeze_prefixes.extend([f"denseblock{b}", f"transition{b}"])

        for name, param in features.named_parameters():
            if any(name.startswith(p) for p in freeze_prefixes):
                param.requires_grad = False

    def extract_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract and project 7x7 spatial feature map into visual tokens.
        pixel_values: (B, 1, 224, 224) in range [-1024, 1024]
        Returns: (B, 49, decoder_hidden_size)
        """
        # Ensure 1-channel grayscale
        if pixel_values.dim() == 4 and pixel_values.shape[1] == 3:
            pixel_values = pixel_values.mean(dim=1, keepdim=True)

        features = self.vision_encoder.features(pixel_values)  # (B, 1024, 7, 7)
        B, C, H, W = features.shape
        # Flatten spatial grid: (B, 49, 1024)
        flat_features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Project to decoder space
        visual_tokens = self.vision_projector(flat_features)  # (B, 49, 1024)
        return visual_tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass with multimodal prefix concatenation.
        """
        device = pixel_values.device
        visual_embeds = self.extract_visual_features(pixel_values)  # (B, 49, D)
        B, num_vis, D = visual_embeds.shape

        if input_ids is not None:
            # Embed text tokens using BioGPT's embedding layer
            text_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T, D)
            # Concatenate visual prefix + text tokens
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 49 + T, D)

            # Combined attention mask
            if attention_mask is not None:
                vis_mask = torch.ones((B, num_vis), dtype=attention_mask.dtype, device=device)
                combined_mask = torch.cat([vis_mask, attention_mask], dim=1)
            else:
                combined_mask = torch.ones((B, num_vis + input_ids.shape[1]), dtype=torch.long, device=device)

            # Combined labels: mask visual tokens (-100 so no loss computed on image prefix)
            combined_labels = None
            if labels is not None:
                vis_labels = torch.full((B, num_vis), -100, dtype=labels.dtype, device=device)
                combined_labels = torch.cat([vis_labels, labels], dim=1)

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            return outputs
        else:
            # Inference without input_ids: just return visual embeddings
            return {"visual_embeds": visual_embeds}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer: BioGptTokenizer,
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 128,
        min_length: int = 15,
        num_beams: int = 3,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        Autoregressively generate clinical report findings from a chest radiograph.
        """
        self.eval()
        device = pixel_values.device
        visual_embeds = self.extract_visual_features(pixel_values)  # (B, 49, D)
        B, num_vis, D = visual_embeds.shape

        if prompt_text:
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
            prompt_embeds = self.decoder.get_input_embeddings()(prompt_tokens.input_ids)
            inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
            attention_mask = torch.ones((B, inputs_embeds.shape[1]), dtype=torch.long, device=device)
        else:
            inputs_embeds = visual_embeds
            attention_mask = torch.ones((B, num_vis), dtype=torch.long, device=device)

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_attentions": output_attentions,
            "return_dict_in_generate": True,
            **kwargs
        }

        outputs = self.decoder.generate(**gen_kwargs)
        sequences = outputs.sequences

        # Decode reports
        reports = []
        for seq in sequences:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            reports.append(text)

        attentions = getattr(outputs, "attentions", None)
        return reports, attentions

    def get_token_saliency_map(
        self,
        pixel_values: torch.Tensor,
        token_attentions: Tuple[Tuple[torch.Tensor]],
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract spatial attention heatmap (224x224) from cross-modal attention.
        Averages across attention heads for the target layer, extracting the weights
        from generated report tokens back to the 49 image patch tokens.
        """
        # layer_attentions: tuple of length num_generated_steps
        # each step: tuple of layers; each layer: (batch, heads, 1, total_seq_len)
        if not token_attentions:
            # Fallback uniform heatmap if attention extraction unavailable
            return torch.ones((224, 224), dtype=torch.float32)

        step_heatmaps = []
        for step_layers in token_attentions:
            layer_attn = step_layers[layer_idx]  # (B, H, 1, seq_len)
            # Attention to the 49 visual tokens (positions 0..48)
            vis_attn = layer_attn[:, :, 0, :self.num_visual_tokens]  # (B, H, 49)
            mean_attn = vis_attn.mean(dim=1).squeeze(0)  # (49,)
            step_heatmaps.append(mean_attn)

        # Average across all generation steps
        all_steps = torch.stack(step_heatmaps, dim=0).mean(dim=0)  # (49,)
        grid = all_steps.view(1, 1, 7, 7)
        # Upsample bicubic to 224x224
        upsampled = F.interpolate(grid, size=(224, 224), mode="bicubic", align_corners=False)
        heatmap = upsampled.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def save_checkpoint(self, save_dir: str):
        """
        Save LoRA adapter weights, vision projector, and fine-tuned DenseNet parameters.
        """
        os.makedirs(save_dir, exist_ok=True)
        # 1. Save LoRA weights
        self.decoder.save_pretrained(os.path.join(save_dir, "biogpt_lora"))

        # 2. Save Vision Projector
        torch.save(self.vision_projector.state_dict(), os.path.join(save_dir, "vision_projector.pt"))

        # 3. Save unfrozen DenseNet parameters (denseblock4 & norm5)
        encoder_trainable = {
            k: v.cpu() for k, v in self.vision_encoder.features.named_parameters() if v.requires_grad
        }
        torch.save(encoder_trainable, os.path.join(save_dir, "densenet_finetuned.pt"))

        # 4. Save metadata config
        meta = {
            "model_architecture": "DenseNetBioGPT",
            "vision_encoder": "torchxrayvision/densenet121-res224-all",
            "language_decoder": self.biogpt_model_name,
            "num_visual_tokens": self.num_visual_tokens,
            "visual_dim": self.visual_dim,
            "trainable_components": ["denseblock4", "norm5", "vision_projector", "biogpt_lora"]
        }
        with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, save_dir: str, device: str = "cpu"):
        """
        Load LoRA adapter weights, vision projector, and fine-tuned DenseNet parameters.
        """
        # 1. Load LoRA
        lora_path = os.path.join(save_dir, "biogpt_lora")
        if os.path.exists(lora_path):
            self.decoder = PeftModel.from_pretrained(self.decoder.get_base_model(), lora_path)

        # 2. Load Vision Projector
        proj_path = os.path.join(save_dir, "vision_projector.pt")
        if os.path.exists(proj_path):
            self.vision_projector.load_state_dict(torch.load(proj_path, map_location=device))

        # 3. Load fine-tuned DenseNet weights
        densenet_path = os.path.join(save_dir, "densenet_finetuned.pt")
        if os.path.exists(densenet_path):
            densenet_state = torch.load(densenet_path, map_location=device)
            current_state = self.vision_encoder.features.state_dict()
            current_state.update(densenet_state)
            self.vision_encoder.features.load_state_dict(current_state)

        self.to(device)
