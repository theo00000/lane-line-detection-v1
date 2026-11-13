# Dataset and quality checks

## Source snapshot (provided by project author)

- 50 annotated road images paired with XML files, in an `archive.zip` (approximately 193 MiB ZIP in the inspected working environment; may differ in other copies).
- Single object label `road_line`; 172 annotated bounding boxes across the archive.
- 49 images contain at least one annotated box and one image has no labeled objects.
- Original annotations are Pascal VOC-style XML boxes (`xmin`, `ymin`, `xmax`, `ymax`).
- Previous inspection identified **15 images** requiring EXIF orientation correction; `prepare_yolo.py` applies orientation normalization before writing converted images.
- Initial deterministic 80/20 split: 40 train / 10 validation (53 boxes in validation), seed 42.

## Conversion

```powershell
python prepare_yolo.py --zip "$env:USERPROFILE\Downloads\archive.zip" --out yolo_road_line --seed 42 --train-ratio 0.8
```

The generated `yolo_road_line/data.yaml` uses an **absolute path on the machine where conversion ran**. This is intentionally ignored by Git; re-run conversion on a new machine rather than committing a path that only works on one laptop.

YOLO box lines use: `class_id x_center y_center width height`, with coordinates normalized to image width and height.

## Quality risks

1. **Annotation semantics:** some boxes tightly cover small dashed segments while others include broad road areas; audit examples before claiming box-level quality.
2. **Incomplete labels:** unmarked road lines in an image are not necessarily model false negatives under benchmark rules if they were never labeled.
3. **EXIF mismatch:** boxes must be drawn on EXIF-transposed images with dimensions matching the XML record.
4. **Random split leakage:** similar views or adjacent frames may appear in both train and validation; if grouping metadata exists, split by scene/clip/location instead.
5. **Limited diversity:** 40 training images and 10 validation images are inadequate for reliable generalization claims; expand to multiple locations, lighting, perspective, wet roads and shadows.

## Dataset license and provenance

Do **not** push original images/XML, `archive.zip`, or normalized image dataset without verifying licensing and privacy. Record source, creator, download date, annotation license, and redistribution terms in this file when confirmed. These details were not provided in the project log and must not be fabricated.

## Stronger future evaluation

Create an independent, annotated test set, never reused in model selection. Document its capture conditions and apply the same annotation policy consistently. Report results separately from training/validation statistics.
