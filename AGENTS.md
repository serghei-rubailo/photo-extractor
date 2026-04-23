# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Photo extractor tool for the Krugosvet project. Detects and extracts individual sub-photos from scanned images containing multiple photos on a single scan. Uses OpenCV contour detection to find photo boundaries, deskews rotated photos, and saves each as a separate file.

## Running

```bash
python splitPhoto.py
```

Reads `.jpg` files from `./scan/`, writes extracted sub-photos to `./output/`, moves processed originals to `./processed/`. All three directories are auto-created on startup.

## Dependencies

Install: `pip install -r requirements-dev.txt`

Runtime: opencv-python, numpy. Dev: pytest, ruff.

## Lint and Test

```bash
ruff check .
pytest tests/ -v
```

## Architecture

Single-file tool (`splitPhoto.py`):

- `find_subphotos_and_save()` — core pipeline: read image → grayscale → threshold → find contours → filter by area → deskew via minAreaRect → crop → save. Moves the original to `processed/` after extraction.
- `check_and_create_folder()` — ensures a directory exists.
- `__main__` block — iterates over `./scan/*.jpg` with a hardcoded `min_contour_area=500_000`.

Key OpenCV flow: binary threshold at 200 → `RETR_EXTERNAL` contours → `minAreaRect` for rotation angle → `warpAffine` to deskew → crop bounding box.
