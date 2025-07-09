import random

import torch
import torchvision.transforms.v2.functional as func2

from training.neo_dataset.neo_data_point import NeoDataPoint


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, datapoint: NeoDataPoint, **kwargs):
        if datapoint.is_lax():
            if random.random() < self.p:
                datapoint = self._hflip(datapoint)

        return datapoint

    def _hflip(self, datapoint: NeoDataPoint):
        _, height, width = func2.get_dimensions(datapoint.image)

        image_list = list()
        contour_list = list()
        for image, contour in datapoint.get_frames():
            image_list.append(func2.hflip(image))
            contour_list.append(self._hflip_contour(contour, width))

        return NeoDataPoint(torch.stack(image_list, axis=1),
                            torch.stack(contour_list, axis=1),
                            datapoint.neo_data_info)

    def _hflip_contour(self, contour, width):
        contour[..., 0] = width - 1 - contour[..., 0]
        return contour
