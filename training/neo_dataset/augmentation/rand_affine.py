import math
import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as func2
from torchvision.transforms import InterpolationMode

from training.neo_dataset.neo_data_point import NeoDataPoint


class RandomAffine:
    def __init__(
            self,
            degrees,
            scale=None,
            translate=None,
            shear=None,
            p_rot=0.0,
            p_scale=0.0,
            p_trans=0.0,
            p_shear=0.0,
            image_mean=0.0,
            image_interpolation="bicubic",
    ):
        """
        The same random affine is applied to all frames.
        """
        self.degrees = degrees if isinstance(degrees, (list, tuple)) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, (list, tuple)) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self._p_rot = p_rot
        self._p_scale = p_scale
        self._p_trans = p_trans
        self._p_shear = p_shear
        self.fill_img = image_mean

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: NeoDataPoint, **kwargs):
        return self.transform_datapoint(datapoint)

    def transform_datapoint(self, datapoint: NeoDataPoint):
        _, height, width = func2.get_dimensions(datapoint.image)
        img_size = [width, height]

        # Compute center of image for rotation/scaling
        center = [img_size[1] * 0.5, img_size[0] * 0.5]

        degrees = self.degrees if random.random() < self._p_rot else (0, 0)
        translate = self.translate if random.random() < self._p_trans else None
        scale_ranges = self.scale if random.random() < self._p_scale else None
        shears = self.shear if random.random() < self._p_shear else None
        if tuple(degrees) == (0, 0) \
                and translate is None \
                and scale_ranges is None \
                and shears is None:
            return datapoint

        # Create a random affine transformation
        affine_params = T.RandomAffine.get_params(
            degrees=degrees,
            translate=translate,
            scale_ranges=scale_ranges,
            shears=shears,
            img_size=img_size,
        )

        # Transform per frame
        image_list = list()
        contour_list = list()
        for image, contour in datapoint.get_frames():
            image_list.append(func2.affine(
                image,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
                center=center
            ))

            contour_list.append(self._affine_contour(
                contour,
                *affine_params,
                center))

        return NeoDataPoint(torch.stack(image_list, axis=1),
                            torch.stack(contour_list, axis=1),
                            datapoint.neo_data_info)

    def _affine_contour(self, points, angle, translate, scale, shear, center):
        """
        Apply affine transform to [N,2] points using the same convention as torchvision RandomAffine.
        """
        if (points == 0).all():
            return points

        if isinstance(shear, (int, float)):
            shear = [shear, 0.0]
        shear_x, shear_y = shear

        # Convert degrees to radians
        angle = math.radians(angle)
        shear_x = math.radians(shear_x)
        shear_y = math.radians(shear_y)

        # Build affine matrix as in torchvision
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale

        tan_shear_x = math.tan(shear_x)
        tan_shear_y = math.tan(shear_y)

        a = cos_a - sin_a * tan_shear_y
        b = -cos_a * tan_shear_x - sin_a
        c = sin_a + cos_a * tan_shear_y
        d = -sin_a * tan_shear_x + cos_a

        cx, cy = center
        tx, ty = translate

        # Construct matrix
        M = torch.tensor([
            [a, b],
            [c, d]
        ], dtype=points.dtype, device=points.device)

        # Apply transform: shift to center, affine, shift back + translate
        shifted = points - torch.tensor(center, dtype=points.dtype, device=points.device)
        transformed = shifted @ M.T
        transformed += (torch.tensor(center, dtype=points.dtype, device=points.device)
                        + torch.tensor([tx, ty], dtype=points.dtype, device=points.device))

        return transformed
