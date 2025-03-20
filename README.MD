# Texture Atlas Generator

A simple Python script that automatically generates texture atlas data from sprite sheets by detecting non-transparent regions.

## Overview

This tool analyzes a sprite sheet image with transparency and automatically generates a JSON file containing the coordinates and dimensions of each sprite. It works by:

1. Detecting non-transparent regions in the image
2. Creating bounding boxes around these regions
3. Filtering out overlapping or contained boxes
4. Outputting the coordinates as a JSON atlas file

## Requirements

- Python 3.x
- OpenCV (`cv2`)

## Installation

```bash
# Install required dependencies
pip install opencv-python
```

## Usage

```bash
python generate-texture-atlas.py <sprite_sheet.png>
```

### Example

```bash
python generate-texture-atlas.py assets/characters.png
```

This will generate a file called `characters_atlas.json` in the same directory as the input image.

### Output Format

The generated JSON file has the following structure:

```json
{
    "sprite_0": {
        "x": 10,
        "y": 20,
        "width": 64,
        "height": 64
    },
    "sprite_1": {
        "x": 84,
        "y": 20,
        "width": 32,
        "height": 32
    }
}
```

## Customization

You can modify the script to adjust:

- `alpha_threshold` (default: 10): Pixels with alpha values below this threshold are considered transparent
- `min_width` and `min_height` (default: 10): Minimum dimensions for a valid sprite

## How It Works

1. The script loads the image and extracts its alpha channel
2. It creates a binary mask of non-transparent pixels
3. Contours are detected around these regions
4. Bounding boxes are created for each contour
5. Small boxes are filtered out
6. Boxes contained within larger boxes are removed
7. The remaining boxes are saved as sprite coordinates

## License

MIT License