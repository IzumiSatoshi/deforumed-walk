import os
import cv2
import math
import numpy as np
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from numpngw import write_png
from einops import rearrange, repeat
from PIL import Image

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet


class DepthModel:
    def __init__(self, device, model_path, half_precision=False):
        self.device = device
        self.half_precision = half_precision
        self.depth_min = 1000
        self.depth_max = -1000

        # load model
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = T.Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if half_precision and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)

    def predict(self, prev_img_cv2) -> torch.Tensor:
        # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
        img = prev_img_cv2.astype(np.float32) / 255.0
        img_input = self.transform({"image": img})["image"]

        # MiDaS depth estimation implementation
        sample = torch.from_numpy(img_input).float().to(self.device).unsqueeze(0)
        if self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            if self.half_precision:
                sample = sample.half()
        with torch.no_grad():
            depth_map = self.model.forward(sample)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = depth_map.cpu().numpy()
        torch.cuda.empty_cache()

        # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
        depth_map = np.subtract(50.0, depth_map)
        depth_map = depth_map / 19.0

        depth_map = np.expand_dims(depth_map, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)

        return depth_tensor

    def save(self, filename: str, depth: torch.Tensor):
        bit_depth_output = 16

        depth = depth.cpu().numpy()
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=0)
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        print(f"  depth min:{depth.min()} max:{depth.max()}")
        denom = max(1e-8, self.depth_max - self.depth_min)
        denom_bitdepth_multiplier = {
            8: 255,
            16: 255 * 255,
            32: 1,  # This one is 1 because 32bpc is float32 and isn't converted to uint, like 8bpc and 16bpc are
        }
        temp_image = rearrange(
            (depth - self.depth_min)
            / denom
            * denom_bitdepth_multiplier[bit_depth_output],
            "c h w -> h w c",
        )
        temp_image = repeat(temp_image, "h w 1 -> h w c", c=3)
        if bit_depth_output == 16:
            write_png(filename, temp_image.astype(np.uint16))
        elif bit_depth_output == 32:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            cv2.imwrite(filename.replace(".png", ".exr"), temp_image)
        else:  # 8 bit
            Image.fromarray(temp_image.astype(np.uint8)).save(filename)


if __name__ == "__main__":
    model = DepthModel(torch.device("cuda"), "./models/dpt_large_384.pt", False)
    img = cv2.imread("./data/girl.jpg")

    depth = model.predict(img)
    model.save("./data/test_depth.png", depth)
