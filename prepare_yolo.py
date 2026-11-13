"""Convert road_line XML annotations from archive.zip into a YOLO detection dataset.

Usage:
    python prepare_yolo.py --zip archive.zip --out yolo_road_line

Dependencies: Pillow; standard Python libraries only otherwise.
Note: This is bounding-box detection, NOT lane segmentation or lane curves.
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, type=Path, help='Path to original archive.zip')
    parser.add_argument('--out', default=Path('yolo_road_line'), type=Path)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train-ratio', default=0.8, type=float)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error('train-ratio must be between zero and one')

    with zipfile.ZipFile(args.zip) as archive:
        entries = archive.namelist()
        images = {Path(e).name: e for e in entries if e.lower().endswith(('.jpg', '.jpeg', '.png'))}
        xmls = sorted(e for e in entries if e.lower().endswith('.xml'))
        samples = []
        for xml_path in xmls:
            root = ET.fromstring(archive.read(xml_path))
            img_name = Path((root.findtext('filename') or '').replace('\\', '/')).name
            if img_name not in images:
                raise FileNotFoundError(f'Image missing for {xml_path}: {img_name}')
            samples.append((images[img_name], xml_path, root))
        if not samples:
            raise RuntimeError('No image + XML pairs found')
        random.Random(args.seed).shuffle(samples)
        split = round(len(samples) * args.train_ratio)
        for subset in ('train', 'val'):
            (args.out / 'images' / subset).mkdir(parents=True, exist_ok=True)
            (args.out / 'labels' / subset).mkdir(parents=True, exist_ok=True)

        total_boxes = 0
        empty_images = 0
        rotated = 0
        for idx, (image_path, xml_path, root) in enumerate(samples):
            subset = 'train' if idx < split else 'val'
            with Image.open(BytesIO(archive.read(image_path))) as raw:
                if raw.getexif().get(274) in (3, 5, 6, 7, 8):
                    rotated += 1
                im = ImageOps.exif_transpose(raw).convert('RGB')
            width, height = im.size
            expected = (int(root.findtext('size/width')), int(root.findtext('size/height')))
            if (width, height) != expected:
                raise ValueError(f'Dimension mismatch after EXIF transpose: {xml_path}, actual={(width,height)}, xml={expected}')

            labels = []
            for obj in root.findall('object'):
                cls = (obj.findtext('name') or '').strip()
                if cls != 'road_line':
                    raise ValueError(f'Unexpected class: {cls}')
                box = obj.find('bndbox')
                if box is None:
                    continue
                x1, y1, x2, y2 = [float(box.findtext(tag)) for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                x1, x2 = max(0., x1), min(float(width), x2)
                y1, y2 = max(0., y1), min(float(height), y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f'Invalid box in {xml_path}: {(x1,y1,x2,y2)}')
                xc = (x1+x2) / (2*width)
                yc = (y1+y2) / (2*height)
                bw = (x2-x1) / width
                bh = (y2-y1) / height
                labels.append(f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
            total_boxes += len(labels)
            if not labels:
                empty_images += 1
            stem = Path(image_path).stem
            # Normalizing EXIF orientation before saving is essential to keep boxes aligned.
            im.save(args.out / 'images' / subset / (stem + '.jpg'), quality=90)
            (args.out / 'labels' / subset / (stem + '.txt')).write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')

    config = (f'path: "{args.out.resolve().as_posix()}"\n'
              'train: images/train\n'
              'val: images/val\n'
              'names:\n  0: road_line\n')
    (args.out / 'data.yaml').write_text(config, encoding='utf-8')
    print(f'Done: {len(samples)} images, {split} train, {len(samples)-split} val')
    print(f'{total_boxes} road_line boxes; {empty_images} images with zero boxes; {rotated} images corrected from EXIF rotation')
    print(f'Dataset YAML: {(args.out / "data.yaml").resolve()}')


if __name__ == '__main__':
    main()
