import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import torch


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == "Match Frame 0 RGB":
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == "Match Frame 0 HSV":
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def unsharp_mask(img, kernel_size=(5, 5), sigma=0, amount=0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


img = cv2.cvtColor(cv2.imread("./data/sky.jpg"), cv2.COLOR_BGR2RGB)
color_match_sample = np.array(
    cv2.cvtColor(cv2.imread("./data/blur.png"), cv2.COLOR_BGR2RGB)
)

result = maintain_colors(img, color_match_sample, None)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img)
axs[1].imshow(result)
plt.show()
