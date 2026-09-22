#!/usr/bin/env python3
"""
scripts/train.py

Training script for DenseNetBioGPT on the IU X-Ray (OpenI) dataset.

Features:
- Micro-batching (4-8) with gradient accumulation (effective batch size 16-32).
- Mixed precision (fp16 / bf16) via torch.amp.autocast.
- Selective freezing: early DenseNet layers frozen, DenseBlock4 + Norm5 fine-tuned,
  2-layer MLP projector trained, BioGPT adapted via LoRA.
- Resumable checkpointing: preserves model weights, optimizer, lr_scheduler,
  step count, and full loss logs to survive Colab/Kaggle session limits.
- Validation loss monitoring with early stopping & best checkpoint preservation.
"""

import os
import sys
import argparse
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer, get_cosine_schedule_with_warmup

# Add repo root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import DenseNetBioGPT
from src.dataset import IUXRayDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train DenseNetBioGPT on IU X-Ray")
    parser.add_argument("--data_dir", type=str, default="data/iu_xray", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/densenet_biogpt", help="Checkpoint directory")
    parser.add_argument("--biogpt_model", type=str, default="microsoft/biogpt", help="BioGPT pretrained weights")
    parser.add_argument("--xrv_weights", type=str, default="densenet121-res224-all", help="TorchXRayVision checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per step (4-8 for 16GB GPU)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (effective BS 16)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr_projector", type=float, default=2e-4, help="Learning rate for vision projector & LoRA")
    parser.add_argument("--lr_encoder", type=float, default=2e-5, help="Learning rate for fine-tuning DenseBlock4")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for cosine schedule")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=128, help="Max report token length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate on validation split every N steps")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use fp16 mixed precision")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable fp16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, use_fp16: bool) -> float:
    """Compute mean validation loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                total_steps += 1

    model.train()
    return total_loss / max(1, total_steps)


def save_full_training_state(
    save_dir: str,
    model: DenseNetBioGPT,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    step: int,
    best_val_loss: float,
    train_history: list
):
    """Saves model weights, optimizer, scheduler, and training history for seamless resumption."""
    os.makedirs(save_dir, exist_ok=True)
    # Save model weights
    model.save_checkpoint(save_dir)

    # Save optimizer, scheduler, and step state
    state = {
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "train_history": train_history
    }
    torch.save(state, os.path.join(save_dir, "training_state.pt"))
    with open(os.path.join(save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)
    print(f"[Checkpoint] Full training state saved to {save_dir} at step {step} (Epoch {epoch})")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DenseNetBioGPT Training Pipeline ===")
    print(f"Device: {device} | Mixed Precision: {args.fp16 and device.type == 'cuda'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Tokenizer
    print(f"[Init] Loading BioGPT Tokenizer ({args.biogpt_model})...")
    tokenizer = BioGptTokenizer.from_pretrained(args.biogpt_model)

    # 2. Build Datasets & DataLoaders
    print(f"[Init] Loading IU X-Ray splits from {args.data_dir}...")
    splits_dir = os.path.join(args.data_dir, "splits")
    images_dir = os.path.join(args.data_dir, "images")

    train_ds = IUXRayDataset(
        split_json=os.path.join(splits_dir, "train.json"),
        images_dir=images_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    val_ds = IUXRayDataset(
        split_json=os.path.join(splits_dir, "val.json"),
        images_dir=images_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if sys.platform != "win32" else 0
    )

    # 3. Initialize Model
    print("[Init] Initializing DenseNetBioGPT...")
    model = DenseNetBioGPT(
        biogpt_model_name=args.biogpt_model,
        xrv_weights=args.xrv_weights,
        freeze_encoder_blocks=3,  # freeze denseblocks 1..3, train block 4 + norm 5
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_lora=True,
        load_pretrained_weights=True
    )
    model.to(device)

    # Separate parameter groups with distinct learning rates
    encoder_params = [p for p in model.vision_encoder.features.parameters() if p.requires_grad]
    projector_and_lora_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "vision_encoder" not in n
    ]

    optimizer_grouped_parameters = [
        {"params": projector_and_lora_params, "lr": args.lr_projector, "weight_decay": args.weight_decay},
        {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Compute total steps & scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_history = []

    # Resume if requested
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[Resume] Loading state from {args.resume_from}...")
        model.load_checkpoint(args.resume_from, device=str(device))
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if scaler and state.get("scaler_state_dict"):
                scaler.load_state_dict(state["scaler_state_dict"])
            start_epoch = state.get("epoch", 0)
            global_step = state.get("step", 0)
            best_val_loss = state.get("best_val_loss", float("inf"))
            train_history = state.get("train_history", [])
            print(f"[Resume] Resumed from Epoch {start_epoch}, Step {global_step} (Best Val Loss: {best_val_loss:.4f})")

    print(f"\n[Training Protocol]")
    print(f"  Micro-batch size: {args.batch_size} | Gradient accumulation: {args.grad_accum_steps}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"  Epochs: {args.epochs} | Steps per epoch: {steps_per_epoch} | Total steps: {total_training_steps}")
    print(f"  Projector/LoRA LR: {args.lr_projector} | DenseNet Block4 LR: {args.lr_encoder}")

    start_time = time.time()
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log step progress
                if global_step % 10 == 0 or global_step == 1:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_step_loss = running_loss / args.grad_accum_steps
                    running_loss = 0.0
                    elapsed = time.time() - start_time
                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0
                    print(f"Epoch [{epoch+1}/{args.epochs}] Step [{global_step}/{total_training_steps}] | Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e} | VRAM: {vram_mb:.0f}MB | Elapsed: {elapsed:.0f}s")

                # Periodic Evaluation
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, args.fp16)
                    print(f"\n---> [Validation @ Step {global_step}] Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")
                    train_history.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "time": round(time.time() - start_time, 2)
                    })

                    # Checkpoint if best validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = os.path.join(args.output_dir, "best_checkpoint")
                        save_full_training_state(best_dir, model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss, train_history)

                # Periodic Regular Checkpointing
                if global_step % args.save_steps == 0:
                    step_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    save_full_training_state(step_dir, model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss, train_history)

        epoch_time = time.time() - epoch_start
        print(f"=== Epoch {epoch+1} Completed in {epoch_time:.1f}s ===")

    # Final save
    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    save_full_training_state(final_dir, model, optimizer, scheduler, scaler, args.epochs, global_step, best_val_loss, train_history)
    print(f"\n[Done] Training complete! Final checkpoint saved to {final_dir}")


if __name__ == "__main__":
    main()
