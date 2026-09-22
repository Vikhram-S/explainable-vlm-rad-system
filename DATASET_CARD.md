---
license: apache-2.0
language:
- en
pretty_name: IU X-Ray (OpenI) - Filtered & Split for Report Generation
tags:
  - radiology
  - chest-xray
  - medical-imaging
  - report-generation
  - nlm-openi
  - indiana-university
---

# IU X-Ray (OpenI) — Filtered & Split Dataset for CXR Report Generation

## Dataset Description

This dataset card documents the filtered and split version of the **Indiana University Chest X-Ray
(IU X-Ray)** collection, publicly released by the **National Library of Medicine (NLM)** via the
[Open-i biomedical image search service](https://openi.nlm.nih.gov/).

The original IU X-Ray dataset contains chest radiographs and associated radiology reports from
Indiana University's Health hospital network. It is widely used as a public benchmark for
automated radiology report generation research.

This processed version is used to train and evaluate [DenseNetBioGPT](https://huggingface.co/Vikhram-S/mimic-vit-biogpt).

---

## Source & License

| Property | Details |
|---|---|
| **Primary Source** | [NLM Open-i Service](https://openi.nlm.nih.gov/) |
| **Download URLs** | Reports: `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz` <br> Images: `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz` |
| **License** | Public Open Access — NLM Open-i Service Terms of Use (no credentialing required) |
| **Original Paper** | Demner-Fushman et al., *JAMIA* 2016 ([DOI: 10.1093/jamia/ocv080](https://doi.org/10.1093/jamia/ocv080)) |

---

## Filtering Steps

All preprocessing is performed by [`scripts/download_and_preprocess_openi.py`](https://huggingface.co/Vikhram-S/mimic-vit-biogpt/blob/main/scripts/download_and_preprocess_openi.py).

1. **XML Parsing:** Official NLM OpenI release tarballs parsed using Python's `xml.etree.ElementTree`.
2. **Mandatory Findings:** Records filtered to those with a non-empty `FINDINGS` section of ≥ 10 characters.
3. **Image Identifier:** At least one valid `parentImage` element must be present in the XML.
4. **Anonymization Normalization:** Age/date tokens (`XXXX`, `xxxx`) replaced with `[ANON]` for consistency.
5. **View Assignment:** `frontal_image` and `lateral_image` assigned using caption parsing (PA/AP vs. Lateral keywords), with IU X-Ray's standard first-is-PA, second-is-lateral convention as fallback.

---

## Dataset Statistics

| Property | Value |
|---|---|
| Raw XML report files | 3,955 |
| Usable reports after filtering | 3,337 |
| Reports filtered out | 618 |
| Average findings length | 31.4 words |
| Max findings length | 169 words |
| Min findings length | 7 words |

### Splits (seed=42, patient-level stratification)

Patient-level splitting ensures that both frontal and lateral views from the same patient appear
in exactly one split (no data contamination):

| Split | Studies |
|---|---|
| Train | 2,335 |
| Val | 333 |
| Test | 669 |
| **Total** | **3,337** |

---

## Data Fields

Each record in the split JSON files contains:

| Field | Type | Description |
|---|---|---|
| `study_id` | string | Study identifier (e.g., `CXR1234`) |
| `indication` | string | Clinical indication for the exam (may contain `[ANON]`) |
| `comparison` | string | Prior exam comparison text |
| `findings` | string | Normalized chest X-ray findings text |
| `impression` | string | Radiology impression summary |
| `report` | string | Combined `findings + IMPRESSION: {impression}` string (training target) |
| `images` | list[str] | All associated image filenames |
| `frontal_image` | string | Filename of the frontal (PA/AP) radiograph |
| `lateral_image` | string | Filename of the lateral radiograph (or `null` if single-view) |
| `num_images` | int | Total number of associated radiographs |

---

## Citation

If you use this processed dataset split, please cite both the original IU X-Ray dataset and this
preprocessing pipeline:

```
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

@misc{vikhrams2026densenetbiogpt,
  author    = {Vikhram S},
  title     = {DenseNetBioGPT: Multimodal Chest Radiograph Report Generation},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/Vikhram-S/mimic-vit-biogpt}
}
```
