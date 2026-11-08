import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calculate_metrics(processed, reference=None):
    """Calculate basic statistics and optional reference-based quality metrics."""
    if processed is None or processed.size == 0:
        raise ValueError("No processed image is available.")

    output_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    values = {
        "width": processed.shape[1],
        "height": processed.shape[0],
        "mean": float(np.mean(output_gray)),
        "std_dev": float(np.std(output_gray))
    }

    if reference is None:
        return values

    if reference.shape[:2] != processed.shape[:2]:
        raise ValueError(
            "Reference and processed images must have the same dimensions. "
            "Use the same scene and align the images before comparison."
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    difference = reference_gray.astype(np.float32) - output_gray.astype(np.float32)
    mse = float(np.mean(difference ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
    ssim_value = float(structural_similarity(reference_gray, output_gray, data_range=255))

    values.update({"mse": mse, "psnr": psnr, "ssim": ssim_value})
    return values
