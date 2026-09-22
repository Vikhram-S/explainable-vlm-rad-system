#!/usr/bin/env python3
"""
scripts/download_and_preprocess_openi.py

Downloads and preprocesses the Indiana University Chest X-Ray (IU X-Ray / OpenI)
dataset directly from the official National Library of Medicine (NLM) Open-i repository.

Compliance & Data Governance:
- NLM OpenI IU X-Ray is an openly distributed public radiology benchmark.
- No credentialed data (e.g. MIMIC-CXR) is downloaded or processed.
- Filters reports to require non-empty Findings.
- Splits data at the patient/study level (Train: 70%, Val: 10%, Test: 20%) with fixed seed 42.

Usage:
    python scripts/download_and_preprocess_openi.py --output_dir data/iu_xray
    # To also download the 1.3GB image archive:
    python scripts/download_and_preprocess_openi.py --output_dir data/iu_xray --download_images
"""

import os
import sys
import argparse
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[Info] File already exists: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return dest_path

    print(f"[Download] Fetching: {url} -> {dest_path}")
    
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print("\n[Download] Complete!")
    return dest_path


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize report text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize anonymization tags like 'xxxx' or 'XXXX'
    text = re.sub(r"\b[Xx]{2,}\b", "[ANON]", text)
    # Remove leading numbering or bullet artifacts if any
    text = re.sub(r"^(\d+\.|\-|\*)\s*", "", text)
    return text.strip()


def parse_iu_xray_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse an IU X-Ray report XML file.
    Returns parsed dictionary or None if malformed.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return None

    # Extract study UID
    uid_elem = root.find(".//uId")
    study_id = uid_elem.attrib.get("id") if uid_elem is not None else xml_path.stem

    findings = ""
    impression = ""
    indication = ""
    comparison = ""

    # Parse sections from AbstractText
    for elem in root.findall(".//AbstractText"):
        label = elem.attrib.get("Label", "").upper()
        content = elem.text or ""
        if label == "FINDINGS":
            findings = clean_text(content)
        elif label == "IMPRESSION":
            impression = clean_text(content)
        elif label == "INDICATION":
            indication = clean_text(content)
        elif label == "COMPARISON":
            comparison = clean_text(content)

    # Parse associated image files
    image_entries = []
    for p_img in root.findall(".//parentImage"):
        img_id = p_img.attrib.get("id", "")
        # Look for figure caption or view info
        caption_elem = p_img.find(".//caption")
        caption = (caption_elem.text or "") if caption_elem is not None else ""
        
        # Check projection if indicated in figure or filename
        img_filename = f"{img_id}.png" if not img_id.endswith(".png") else img_id
        is_lateral = "lateral" in caption.lower() or "lat" in caption.lower()
        
        image_entries.append({
            "image_id": img_id,
            "filename": img_filename,
            "caption": caption,
            "is_lateral": is_lateral
        })

    # Basic validity filtering:
    # Must have non-trivial findings
    if not findings or len(findings) < 10:
        return None
    # Filter out empty or trivial records
    if findings.lower() in {"none", "none.", "normal", "no acute cardiopulmonary abnormality."} and not impression:
        # Keep if impression has diagnostic text, otherwise record may have insufficient supervision
        pass

    if not image_entries:
        return None

    # Identify frontal vs lateral images
    frontal_img = None
    lateral_img = None
    
    if len(image_entries) == 1:
        frontal_img = image_entries[0]["filename"]
    elif len(image_entries) >= 2:
        # Check if captions specifically distinguish one as purely lateral
        has_pure_lateral = [bool(re.search(r"\blateral\b", e["caption"], re.I) and not re.search(r"\b(pa|ap|frontal)\b", e["caption"], re.I)) for e in image_entries]
        has_pure_frontal = [bool(re.search(r"\b(pa|ap|frontal)\b", e["caption"], re.I) and not re.search(r"\blateral\b", e["caption"], re.I)) for e in image_entries]
        
        if any(has_pure_frontal) and any(has_pure_lateral):
            f_idx = has_pure_frontal.index(True)
            l_idx = has_pure_lateral.index(True)
            frontal_img = image_entries[f_idx]["filename"]
            lateral_img = image_entries[l_idx]["filename"]
        else:
            # Standard OpenI pairing convention: first is Frontal (PA), second is Lateral
            frontal_img = image_entries[0]["filename"]
            lateral_img = image_entries[1]["filename"]

    full_report = findings
    if impression:
        full_report = f"{findings} IMPRESSION: {impression}"

    return {
        "study_id": study_id,
        "indication": indication,
        "comparison": comparison,
        "findings": findings,
        "impression": impression,
        "report": full_report,
        "images": [e["filename"] for e in image_entries],
        "frontal_image": frontal_img,
        "lateral_image": lateral_img,
        "num_images": len(image_entries)
    }


def split_dataset(records: List[Dict[str, Any]], train_ratio=0.70, val_ratio=0.10, seed=42):
    """
    Split records deterministically by patient ID to prevent data leakage.
    In IU X-Ray, study_id / filename prefixes (e.g. CXR1234_...) identify the patient/case.
    """
    import random
    rng = random.Random(seed)

    # Group by patient ID prefix (e.g., 'CXR1234' from 'CXR1234_IM-...')
    patient_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        sid = r["study_id"]
        # Extract patient token
        match = re.match(r"(CXR\d+)", sid, re.IGNORECASE)
        pid = match.group(1).upper() if match else sid
        patient_groups.setdefault(pid, []).append(r)

    pids = sorted(list(patient_groups.keys()))
    rng.shuffle(pids)

    n_total = len(pids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train:n_train + n_val])
    test_pids = set(pids[n_train + n_val:])

    train_set = [r for pid in train_pids for r in patient_groups[pid]]
    val_set = [r for pid in val_pids for r in patient_groups[pid]]
    test_set = [r for pid in test_pids for r in patient_groups[pid]]

    return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess IU X-Ray (OpenI) dataset.")
    parser.add_argument("--output_dir", type=str, default="data/iu_xray", help="Target output directory")
    parser.add_argument("--download_images", action="store_true", help="Download full 1.36GB image tarball")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download reports
    reports_tar = raw_dir / "NLMCXR_reports.tgz"
    download_file(REPORTS_URL, reports_tar)

    # 2. Extract reports
    reports_extract_dir = raw_dir / "reports_xml"
    if not reports_extract_dir.exists() or not any(reports_extract_dir.iterdir()):
        print(f"[Extract] Extracting XML reports to {reports_extract_dir}...")
        reports_extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(reports_tar, "r:gz") as tar:
            tar.extractall(path=reports_extract_dir)
        print("[Extract] Reports extracted successfully.")
    else:
        print(f"[Info] XML reports already extracted at {reports_extract_dir}")

    # Find all xml files
    xml_files = list(reports_extract_dir.rglob("*.xml"))
    print(f"[Process] Found {len(xml_files)} total raw XML files.")

    # 3. Parse and filter
    valid_records = []
    skipped_no_findings = 0
    skipped_malformed = 0

    for xf in xml_files:
        rec = parse_iu_xray_xml(xf)
        if rec is None:
            skipped_no_findings += 1
        else:
            valid_records.append(rec)

    print(f"[Process] Usable reports with valid Findings: {len(valid_records)}")
    print(f"[Process] Filtered out (empty/insufficient findings): {skipped_no_findings}")

    # 4. Deterministic split
    train_set, val_set, test_set = split_dataset(valid_records, train_ratio=0.70, val_ratio=0.10, seed=args.seed)
    print(f"[Split] Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # 5. Save splits to JSON
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)
    with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    # 6. Compute statistics summary
    total_words = [len(r["findings"].split()) for r in valid_records]
    summary = {
        "dataset_name": "Indiana University Chest X-Ray (IU X-Ray / OpenI)",
        "source": "National Library of Medicine (NLM) Open-i",
        "provenance_url": "https://openi.nlm.nih.gov/",
        "license": "Public Open Access (NLM Open-i terms)",
        "total_raw_reports": len(xml_files),
        "total_filtered_reports": len(valid_records),
        "train_reports": len(train_set),
        "val_reports": len(val_set),
        "test_reports": len(test_set),
        "avg_findings_words": round(sum(total_words) / len(total_words), 2) if total_words else 0,
        "max_findings_words": max(total_words) if total_words else 0,
        "min_findings_words": min(total_words) if total_words else 0,
        "filtering_criteria": [
            "XML parsing of official NLM OpenI release",
            "Mandatory non-empty FINDINGS section (length >= 10 chars)",
            "Associated chest radiograph image identifier present",
            "Standard anonymization token normalization ([ANON])",
            "Patient-level split to prevent data contamination across train/val/test"
        ]
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Done] Dataset metadata and splits saved to {splits_dir}")
    print(f"[Done] Dataset summary saved to {out_dir / 'dataset_summary.json'}")

    # 7. Optional image download
    if args.download_images:
        images_tar = raw_dir / "NLMCXR_png.tgz"
        download_file(IMAGES_URL, images_tar)
        img_extract_dir = out_dir / "images"
        if not img_extract_dir.exists() or not any(img_extract_dir.iterdir()):
            print(f"[Extract] Extracting images to {img_extract_dir}...")
            img_extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(images_tar, "r:gz") as tar:
                tar.extractall(path=img_extract_dir)
            print("[Extract] Images extracted.")
        else:
            print(f"[Info] Images already extracted at {img_extract_dir}")


if __name__ == "__main__":
    main()
