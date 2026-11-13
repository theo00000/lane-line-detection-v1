# Publishing to GitHub safely

## Include

- Python source code (`src`, `prepare_yolo.py`, `generate_sample.py`, `scripts`)
- Tests and non-sensitive configuration/examples
- `README.md`, `docs`, `.gitignore`, `requirements.txt`
- A few **publicly shareable** demonstration screenshots (synthetic example already included)
- Selected `models/yolo_v1/best.pt` and optionally `models/yolo_v2/best.pt` **only if the original model/data license allows redistribution**

## Exclude

- `.venv/` (large local binary dependencies)
- `data/`, `yolo_road_line/`, the source dataset `archive.zip`, and other images if rights are unknown
- `runs/`, `reviews/`, and `outputs/` (training caches, raw checkpoints, bulky image/video output)
- `.env` secrets, machine-specific paths and private recordings

## Suggested commands in PowerShell

```powershell
cd D:\Projects\lane-line-detection-v1
# Confirm what will be tracked BEFORE commit.
git status --short
git add README.md docs scripts tests assets .gitignore
git status --short
git diff --cached --stat
# If intended, include source and configuration code already reviewed.
git add src prepare_yolo.py generate_sample.py requirements.txt
# Include weights only after confirming redistribution rights:
# git add models/yolo_v1/best.pt
# git add models/yolo_v2/best.pt
# Then commit and push to an existing configured remote:
git commit -m "docs: document YOLO road-marking experiments and evaluation"
git remote -v
git push -u origin main
```

If Git has not been initialized, run `git init`, `git branch -M main`, and add the remote URL for a repository you actually created first. Do not replace an existing remote blindly.

## Check tracked-file sizes

```powershell
git ls-files | ForEach-Object { if (Test-Path $_) { Get-Item $_ } } |
    Sort-Object Length -Descending |
    Select-Object -First 15 Name, Length
```

The package itself is small, but the total repository size depends on *what is already tracked*, including historic commits. A `.gitignore` change does not automatically untrack previously committed files. If a large or sensitive file was committed earlier, review its history and remove it appropriately before publishing.
