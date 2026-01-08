# Automated Dermoscopic Lesion Detection: A Classical Image Processing Pipeline

## Introduction

Dermoscopy is a non-invasive imaging technique that magnifies and illuminates skin lesions, aiding dermatologists in diagnosing melanoma and other skin cancers. Early detection of melanoma significantly improves patient survival.

Automated lesion detection can assist clinicians by quickly and objectively segmenting the lesion area from surrounding skin, which is crucial for computer-aided diagnosis (many diagnostic features rely on lesion shape and color relative to skin).

**Challenges:** Dermoscopic images often contain artifacts (hair, ruler marks, bubbles) and low contrast between lesion and skin, making accurate segmentation difficult. An effective solution requires robust preprocessing and segmentation techniques to handle these issues.

## Method Overview

We implement a classic image processing pipeline (no machine learning) to segment lesions from dermoscopic images. The steps include:

1. **Hair artifact removal:** Detect hairs via morphological filtering and remove them by inpainting.
2. **Lesion likelihood mapping:** Convert image to Lab color space and compute a "likelihood" map based on color differences (lesion vs. skin).
3. **Thresholding:** Apply Otsu's threshold on the likelihood map to get a binary mask of the lesion. Keep only the largest connected component to discard noise.
4. **Morphological cleanup:** Fill any holes in the lesion mask (e.g., gaps caused by hairs), then apply closing (and a light opening) to smooth the lesion boundary.
5. **Output generation:** Produce visual outputs for each stage – hair-removed image, likelihood map, initial mask, final mask, and overlays – to evaluate the segmentation results.

## Hair Artifact Removal (Preprocessing)

The original dermoscopic image often has dark hairs obscuring the lesion. We perform multi-scale black-hat filtering to detect hair pixels at various scales, generating a hair mask (hair locations are bright). The mask is then morphologically cleaned to remove noise and small components, since hair artifacts greatly hinder lesion segmentation. 

Finally, the hair pixels are removed via inpainting, replacing them with neighboring skin-colored pixels. This DullRazor-style approach ensures that subsequent steps focus on actual lesion pixels without interference from hairs.

### Key Parameters:
- Multi-scale kernel sizes: [9, 13, 17, 21, 27, 31]
- CLAHE contrast boost: clipLimit=3.0
- Morphological cleanup: closing + opening operations
- Minimum hair component size: 30 pixels

## Color-Based Lesion Segmentation (Lab & Otsu)

After hair removal, the algorithm focuses on the lesion itself. We convert the image to Lab color space, where the A and B channels represent color-opponent dimensions. The lesion's pigmentation typically differs in chromaticity from surrounding skin.

We calculate a lesion likelihood map by measuring how far each pixel's A/B values deviate from the median skin color (assuming the bulk of the image is normal skin). This yields a grayscale map highlighting the lesion region (brighter = more likely lesion). Gaussian smoothing is applied to this map to reduce noise and small texture details.

Next, we use **Otsu's thresholding** to automatically binarize the likelihood map. Otsu's method finds an optimal threshold that separates the lesion (foreground) from background by maximizing inter-class variance. The result is a binary mask of the lesion region. 

Immediately after thresholding, we retain only the largest connected component in the mask – this step removes any spurious small regions or noise that might have been thresholded as "lesion". If Otsu's initial result selects an inverted region (e.g., mostly background as lesion), the code inverts the mask to ensure the lesion is the foreground.

## Post-Processing and Segmentation Results

Using the initial mask as a base, we perform morphological post-processing to refine the lesion segmentation:

- **Hole filling:** Small gaps inside the lesion (caused by hairs crossing the lesion) are filled by a flood-fill operation, ensuring the lesion mask is solid.
- **Closing operation:** A 15×15 elliptical kernel smooths the lesion outline and connects narrow fissures.
- **Opening operation:** A 3×3 kernel removes tiny protrusions.
- **Edge smoothing:** Gaussian blur followed by re-binarization smooths edges.

The pipeline produces multiple visualization outputs:
1. Hair-cleaned input image
2. Computed lesion likelihood map (grayscale)
3. Binary mask right after Otsu thresholding
4. Final refined lesion mask after filling and morphology
5. Lesion contour overlay on original image

## Preliminary Results & Performance

The classical pipeline was tested on dermoscopic images to qualitatively assess segmentation performance. Key findings:

- **Otsu threshold:** Selected threshold values around 63 (on 0–255 scale) for the lesion likelihood map
- **Mask coverage:** Final lesion masks cover approximately 15–25% of image area, consistent with visible lesion boundaries
- **Accuracy:** The deterministic approach produces clean segmentations suitable for feature analysis

Visual inspection indicates that the pipeline accurately delineates the main lesion. Minor errors are possible in areas of low contrast or where very fine structures were removed by morphological smoothing. Future evaluations on datasets with ground-truth masks (using metrics like Dice coefficient or Jaccard index) would further quantify accuracy.

## Project Structure

```
├── Final.ipynb           # Main Jupyter notebook with complete pipeline
├── lesion_batch.py       # Batch processing script for multiple images
├── lesion_charts/        # Output directory for segmentation results
│   ├── Charts/           # Visualization outputs
│   └── ISIC_*/           # Processed image folders
└── README.md             # This file
```

## Usage

### Requirements
- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Jupyter Notebook

### Running the Pipeline

1. **Single Image Processing:** Open `Final.ipynb` and run the cells sequentially:
   - Step 1: Hair removal
   - Step 2: Lesion segmentation

2. **Batch Processing:** Use `lesion_batch.py` to process multiple images automatically.

### Example
```python
import cv2
import numpy as np

# Load image
bgr = cv2.imread("ISIC_0000042.jpg")

# Hair removal (Step 1)
# ... [see Final.ipynb for implementation]

# Lesion segmentation (Step 2)
# ... [see Final.ipynb for implementation]
```

## Limitations

- **Color-based heuristics:** The approach relies on fixed heuristics and may struggle with lesions that have atypical color contrast or very irregular borders.
- **Contrast dependency:** If the lesion's color is not sufficiently distinct from surrounding skin, the Lab-based heuristic might fail to produce a clear likelihood map.
- **Hair removal limitations:** Very small or light-colored hairs might not be fully removed by the current DullRazor method (which targets dark hairs), potentially leaving artifacts.
- **Fixed parameters:** The morphological operations use fixed kernel sizes – these may not optimally suit lesions of vastly different sizes or shapes, potentially causing over-smoothing or leftover noise.

## Future Work & Improvements

1. **Adaptive parameters:** Implement adaptive threshold percentiles and dynamic morphology kernel sizes based on lesion size.
2. **Machine learning integration:** Incorporate deep learning techniques (e.g., U-Net convolutional networks) which have shown superior accuracy in lesion segmentation.
3. **Hybrid approach:** Use the current preprocessing (hair removal) to aid a learned model for better overall performance.
4. **Quantitative evaluation:** Evaluate on datasets with ground-truth masks using metrics like Dice coefficient and Jaccard index.
5. **Extended artifact handling:** Improve detection and removal of additional artifacts (ruler marks, bubbles, color casts).

## References

- Dermoscopy for melanoma detection: PMC/NCBI publications on dermatologic imaging
- Classical image processing: OpenCV documentation and morphological operations
- Deep learning approaches: Recent work on U-Net and convolutional networks for medical image segmentation

## Conclusion

This project demonstrates a complete traditional pipeline for dermoscopic lesion detection, providing interpretable intermediate results at each stage. The classical approach is transparent and computationally efficient, making it suitable for real-time clinical applications. Moving forward, combining this step-by-step approach with modern data-driven methods could yield a powerful tool for automated dermoscopic analysis.
