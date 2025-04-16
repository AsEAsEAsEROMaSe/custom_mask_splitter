# Custom Mask Splitter for ComfyUI

[Read in Ukrainian | Читати українською](README_uk.md)

A collection of nodes for ComfyUI that provide advanced mask manipulation operations, specifically designed for splitting and processing intersecting masks.

## Features

- **Mask Splitter**: Splits two intersecting masks by dividing the intersection area based on the distance from each pixel to the center of mass of each mask.
- **Push Bubbles To Zones**: Separates intersecting masks by simulating "bubbles" pushing towards their respective zones.
- **Gravity Mask Splitter**: Uses a gravity-like force field to separate overlapping masks, with customizable strength parameters.
- **Watershed Mask Splitter**: Implements the watershed algorithm to split intersecting masks, useful for complex shapes.
- **Image Mask Inserter**: Inserts an image into the white area of a mask, scaling it to fit exactly.

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/custom_mask_splitter.git
```

2. Restart ComfyUI or reload the web interface.

## Usage

After installation, the nodes will appear in the "image/masking" category in the ComfyUI node browser.

### Mask Splitter
Takes two masks as input and splits their intersection area based on the distance to the center of mass of each mask.

### Push Bubbles To Zones
Separates intersecting masks by simulating bubbles being pushed to their respective zones, with adjustable iterations and filter size.

### Gravity Mask Splitter
Provides a physics-based approach to mask separation with the following parameters:
- **gravity_strength**: Force of attraction to respective mask pixels
- **center_strength**: Force of attraction to the center of mass
- **iterations**: Number of algorithm iterations
- **protect_core**: Whether to protect the central part of the mask
- **core_radius**: Radius of the protected central area

### Watershed Mask Splitter
Uses the watershed algorithm to divide overlapping masks, with a filter size parameter to remove isolated areas.

### Image Mask Inserter
Inserts an image into the white area of a mask, with options for different interpolation methods.

## Examples

[Add example images/workflows here]

## License

[Add license information]

## Credits

[Add credits/acknowledgments if needed] 