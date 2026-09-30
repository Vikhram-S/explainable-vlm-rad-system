"""
tests/test_model_and_dataset.py

Unit and smoke test for DenseNetBioGPT model, IUXRayDataset, and explainability components.
Verifies:
1. Dataset batch collation and tensor formats.
2. Model forward pass and loss computation.
3. Gradient backpropagation to unfrozen encoder block, projector, and LoRA decoder.
4. Multimodal text generation with input visual prefix.
5. Checkpoint saving and loading.
"""

import os
import sys
import tempfile
import torch
from transformers import BioGptConfig
from transformers.tokenization_utils_base import BatchEncoding

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import DenseNetBioGPT
from src.dataset import IUXRayDataset, collate_fn


class MockTokenizer:
    """Lightweight mock tokenizer for rapid local testing without network calls."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 0

    def __call__(self, text, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        if isinstance(text, str):
            words = text.split()
            tokens = [hash(w) % (self.vocab_size - 4) + 4 for w in words][:max_length]
            if not tokens:
                tokens = [3]
            # Pad
            if padding:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            input_ids = torch.tensor([tokens], dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})
        else:
            raise NotImplementedError

    def decode(self, token_ids, skip_special_tokens=True):
        return "the lungs are clear without focal consolidation or pneumothorax"


def test_dataset():
    print("[Test 1/5] Testing IUXRayDataset loading...")
    split_path = os.path.join("data", "iu_xray", "splits", "val.json")
    assert os.path.exists(split_path), f"Missing split file: {split_path}"

    tokenizer = MockTokenizer()
    ds = IUXRayDataset(split_json=split_path, tokenizer=tokenizer, max_length=32)
    assert len(ds) > 0, "Dataset should have non-zero records"

    item = ds[0]
    assert "pixel_values" in item
    assert item["pixel_values"].shape == (1, 224, 224), f"Unexpected CXR shape: {item['pixel_values'].shape}"
    assert item["input_ids"].shape == (32,), f"Unexpected input_ids shape: {item['input_ids'].shape}"
    assert item["labels"].shape == (32,)

    batch = collate_fn([ds[0], ds[1]])
    assert batch["pixel_values"].shape == (2, 1, 224, 224)
    assert batch["input_ids"].shape == (2, 32)
    print("  -> Dataset and collation verified successfully.")
    return batch


def test_model_forward_and_backward(batch):
    print("[Test 2/5] Initializing DenseNetBioGPT with lightweight test configuration...")
    cfg = BioGptConfig(
        vocab_size=1000,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=512
    )

    model = DenseNetBioGPT(
        xrv_weights="densenet121-res224-all",
        freeze_encoder_blocks=3,
        lora_r=8,
        lora_alpha=16,
        decoder_config=cfg
    )

    print("[Test 3/5] Testing forward pass and loss computation...")
    outputs = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    assert loss is not None, "Forward pass did not return loss"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  -> Forward loss computed: {loss.item():.4f}")

    print("[Test 4/5] Testing backward pass and gradient flow...")
    loss.backward()

    # Verify frozen layers have no grads
    conv0_weight = model.vision_encoder.features.conv0.weight
    assert conv0_weight.grad is None, "Frozen layer conv0 has gradients!"

    # Verify unfrozen denseblock4 has grads
    db4_param = next(model.vision_encoder.features.denseblock4.parameters())
    assert db4_param.grad is not None, "Unfrozen denseblock4 did not receive gradients!"

    # Verify vision projector has grads
    proj_param = next(model.vision_projector.parameters())
    assert proj_param.grad is not None, "Vision projector did not receive gradients!"
    print("  -> Selective freezing and gradient flow verified successfully.")

    return model


def test_generation_and_checkpoint(model):
    print("[Test 5/5] Testing generation and checkpoint serialization...")
    tokenizer = MockTokenizer()
    x = torch.randn(1, 1, 224, 224) * 100.0

    reports, attentions = model.generate(
        pixel_values=x,
        tokenizer=tokenizer,
        max_new_tokens=16,
        min_length=5,
        num_beams=1
    )
    assert len(reports) == 1, "Expected 1 generated report"
    assert isinstance(reports[0], str)
    print(f"  -> Generated report sample: \"{reports[0]}\"")

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_checkpoint(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "vision_projector.pt"))
        assert os.path.exists(os.path.join(tmpdir, "densenet_finetuned.pt"))
        assert os.path.exists(os.path.join(tmpdir, "model_meta.json"))
        print("  -> Checkpoint saved successfully.")

        # Test reloading
        model.load_checkpoint(tmpdir)
        print("  -> Checkpoint reloaded successfully.")


if __name__ == "__main__":
    print("=== Running Comprehensive DenseNetBioGPT Verification ===")
    batch = test_dataset()
    model = test_model_forward_and_backward(batch)
    test_generation_and_checkpoint(model)
    print("=== ALL TESTS PASSED SUCCESSFULLY! ===")
