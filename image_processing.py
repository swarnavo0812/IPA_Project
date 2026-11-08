import cv2
import numpy as np


def apply_technique(image, technique, gamma=1.4, kernel_size=5):
    """Apply one image-processing method to an image in BGR format."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if technique == "Histogram Equalization":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elif technique == "CLAHE":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        light, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        light = clahe.apply(light)
        return cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    elif technique == "Gamma Correction":
        gamma = max(float(gamma), 0.01)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    elif technique == "Median Filter":
        return cv2.medianBlur(image, kernel_size)

    elif technique == "Gaussian Filter":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    elif technique == "Unsharp Mask":
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    elif technique in ("DFT Low-Pass Filter", "DFT High-Pass Filter"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        center_y, center_x = rows // 2, cols // 2
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        shifted = np.fft.fftshift(dft)

        mask = np.zeros((rows, cols, 2), dtype=np.float32)
        radius = max(5, min(rows, cols) // 8)

        if technique == "DFT Low-Pass Filter":
            cv2.circle(mask, (center_x, center_y), radius, (1, 1), -1)
        else:
            mask[:, :] = 1
            cv2.circle(mask, (center_x, center_y), radius, (0, 0), -1)

        filtered = shifted * mask
        inverse = cv2.idft(np.fft.ifftshift(filtered))
        magnitude = cv2.magnitude(inverse[:, :, 0], inverse[:, :, 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(magnitude.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError("Select a valid processing technique.")
