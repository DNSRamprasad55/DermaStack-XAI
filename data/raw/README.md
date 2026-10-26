# Raw Data Directory (`data/raw/`)

This directory is reserved for raw, unmodified HAM10000 dataset files.

## Expected Files:
1. `HAM10000_metadata.csv` (The official metadata table containing `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`).
2. `HAM10000_images_part_1/` (Folder containing initial batch of `.jpg` dermatoscopic images).
3. `HAM10000_images_part_2/` (Folder containing second batch of `.jpg` dermatoscopic images).

*Note: Alternatively, images from both parts may be placed into a single `images/` directory.*

> [!WARNING]
> Do NOT commit image files to Git. Large binary files are excluded in `.gitignore`.
