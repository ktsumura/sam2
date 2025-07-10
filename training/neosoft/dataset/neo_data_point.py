import numpy as np
import skimage
import torch

from training.neosoft.dataset.neo_data_info import NeoDataInfo
from training.neosoft.utils.plot_utils import PlotUtils


class NeoDataPoint:
    def __init__(self,
                 image: np.ndarray,
                 contour: np.ndarray,
                 neo_data_info: NeoDataInfo):
        self._device = torch.device('cuda:0')

        if isinstance(image, np.ndarray):
            image = torch.Tensor(image).to(self._device)
        if isinstance(contour, np.ndarray):
            contour = torch.Tensor(contour).to(self._device)

        # [Channel, Frame, Height, Width]
        self._image = image
        # [Channel, Frame, The number of points, 2(=x,y)]
        self._contour = contour
        # [Channel, Frame, Height, Width]
        self._label = None
        self._neo_data_info = neo_data_info

    def is_lax(self):
        return self._neo_data_info.is_lax()

    def get_frames(self):
        images = [self.image[:, i, ...] for i in range(self.image.shape[1])]
        contours = [self.contour[:, i, ...] for i in range(self.contour.shape[1])]

        # return images, contours
        return list(zip(images, contours))

    def create_label(self):
        label_list = list()

        for image, contour in self.get_frames():
            if torch.any(contour>0):
                contour_np = np.squeeze(contour.to('cpu').numpy(), axis=0)
                polygon_np = np.fliplr(contour_np)
                image_shape = (image.shape[-2], image.shape[-1])
                label_np = np.expand_dims(skimage.draw.polygon2mask(image_shape, polygon_np).astype(np.int32), axis=0)
                label_list.append(torch.tensor(label_np, dtype=torch.int, device=self.device))
            else:
                label_list.append(torch.zeros_like(image, dtype=torch.int, device=self.device))

        self._label = torch.stack(label_list, dim=1)

    @property
    def device(self):
        return self._device

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, i):
        self._image = i

    @property
    def contour(self):
        return self._contour

    @contour.setter
    def contour(self, c):
        self._contour = c

    @property
    def neo_data_info(self):
        return self._neo_data_info

    @property
    def label(self):
        return self._label

    def display(self, title='', is_grayscale=True, vmin=None, vmax=None):
        PlotUtils.display(self._image, self._contour, self._label)


