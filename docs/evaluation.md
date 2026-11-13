# Evaluation and error analysis

## Definitions

- **Precision** = TP / (TP + FP): fraction of accepted detections that match labeled objects under a chosen IoU/confidence policy.
- **Recall** = TP / (TP + FN): fraction of labeled objects that were found under the evaluation policy.
- **mAP@50**: area under precision-recall curve, computed at IoU threshold 0.50.
- **mAP@50:95**: mAP averaged over IoU thresholds 0.50 to 0.95 in 0.05 increments.

**These are detection box metrics**, not pixel-wise mask IoU, left/right lane accuracy, road-safety reliability, or video-frame success rate. Displayed YOLO `road_line 0.8` values are **model confidence**, not measured accuracy.

## Completed V1 evaluation

| Metric | V1 validation score |
|---|---:|
| Precision | 0.741 |
| Recall | 0.321 |
| mAP@50 | 0.345 |
| mAP@50:95 | 0.219 |

Validation: 10 images, 53 labels. In a separate ten-image prediction run at `conf=0.25`, 26 boxes were output; **26 detections are not necessarily 26 true positives**. Reported speed: preprocessing 8.1 ms, model inference 21.4 ms, postprocessing 5.2 ms per image in that run, without full video capture/render/decode throughput.

## Visual failure cases in supplied outputs

| Case | Observation | Hypothesis / next check |
|---|---|---|
| Large boxes in sky | Predictions occasionally cover non-road regions | Confirm these are predictions, not XML label errors; compare `val_batch*_labels` to matching `*_pred` |
| Straight/continuous edge missed | Visible long white/yellow lines not detected in some views | Ground-truth coverage and model's `road_line` annotation policy may omit or fragment long markings |
| Dashed center-line fragments | Some correct boxes around individual white dashes; many missing | Low recall / tiny-object sensitivity / low-resolution downsampling |
| Large boxes extending to shoulder | Model localizes broad areas instead of marking shape | Examine box shapes and annotation consistency |
| No predictions on a road scene | Image with visible markings has no boxes at `conf=0.25` | Compare ground truth and repeat with alternate confidence (e.g. 0.10) while tracking FP cost |

We cannot determine from screenshots alone whether every missed visible marking is labeled. Likewise, a smaller `val_batch*_pred` plot does not prove the source of sky-region predictions in full-image inference: settings, image resizing, or metric visualization could differ.

## Evaluation protocol for V2

1. Fix a validation image list and label version; never change it between V1 and V2 comparisons.
2. Evaluate both checkpoints at the **same input size** and environment with `scripts/evaluate_models.py`.
3. Review a small set of labeled failure cases with box overlays; categorize FP, FN and localization errors.
4. Record qualitative outcomes separately from dataset-level scores.
5. Test on an independent annotated set. A held-out set is necessary for stronger generalization claims.

## Recommendation

Continue with YOLO as a **road marking object detection** baseline. Do not claim it detects continuous drivable lane geometry until developing a model trained with appropriate polyline/keypoint/segmentation annotations.
