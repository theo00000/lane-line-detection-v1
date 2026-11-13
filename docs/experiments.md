# Experiment log

Use this record to preserve decisions and avoid claiming results that were never obtained.

| Field | YOLO V1 | YOLO V2 |
|---|---|---|
| Purpose | Baseline bounding-box detector | Improve sensitivity with fine-tuning |
| Initialization | `yolo11n.pt` pretrained | `models/yolo_v1/best.pt` (proposed) |
| Images | Train 40 / val 10 | Intended same split |
| Input size during train | 416 | Planned 512 |
| Batch size | 1 | Planned 1 |
| Epochs | 30 completed | Up to 40 (not confirmed) |
| Mixed precision | Auto-disabled during V1 run | Planned `amp=False` |
| Optimizer | AdamW auto (LR 0.002) | AdamW LR 0.0005 (planned) |
| Precision | **0.741** | Not available |
| Recall | **0.321** | Not available |
| mAP@50 | **0.345** | Not available |
| mAP@50:95 | **0.219** | Not available |
| Notes | Missed markings; sky-region FP | Await training and matched validation |

**Important:** The published V1 scores above come from the earlier `imgsz=416` validation. Before ranking V1 and V2, validate **both models at the same resolution** on the same dataset. Record those matched results below when ready.

## Matched validation comparison — pending

After running `scripts/evaluate_models.py --data ... --imgsz 512 --v2 ...`, open `reviews/comparison/summary.md` and copy the verified values into this section. Do not replace the original V1 metrics above without explaining the changed evaluation settings.

## Follow-up research questions

- Does higher resolution improve recall of narrow line fragments at 2GB VRAM limits?
- Does correcting annotation inconsistencies remove sky-region false positives?
- Are observed changes stable across scene-level data splits and unseen test images?
- Would polyline or pixel-mask labels support the stated lane-line objective better than boxes?
