import threading

import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from matplotlib.widgets import Slider

from training.neo_dataset.neo_data_info import NeoDataInfo


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

        self._image = image
        self._contour = contour
        self._neo_data_info = neo_data_info
        self._label = None

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
            contour_np = np.squeeze(contour.to('cpu').numpy(), axis=0)
            polygon_np = np.fliplr(contour_np)
            image_shape = (image.shape[-2], image.shape[-1])
            label_np = np.expand_dims(skimage.draw.polygon2mask(image_shape, polygon_np).astype(np.int32), axis=0)
            label_list.append(torch.tensor(label_np, dtype=torch.int, device=self.device))

        self._label = torch.stack(label_list, dim=1)
        return self._label

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
        if not threading.current_thread() is threading.main_thread():
            return

        image3d = np.squeeze(self._image.to('cpu').numpy(), axis=0)
        contour3d = np.squeeze(self._contour.to('cpu').numpy(), axis=0)
        label3d = np.squeeze(self._label.to('cpu').numpy(), axis=0) if self._label is not None else np.zeros_like(image3d)

        # Turn interactive plotting on
        plt.ion()

        # Calculate the image value min and max
        img_vmin = np.min(image3d) if vmin is None else vmin
        img_vmax = np.max(image3d) if vmax is None else vmax

        lb_vmin = np.min(label3d)
        lb_vmax = np.max(label3d)

        # Display the image
        def _imshow(ax, im3d, idx, vmin=None, vmax=None):
            ax.cla()

            if (idx >= 0) and (idx < len(im3d)):
                im = im3d[idx]

                if len(im) > 0 and im.ndim == 2:
                    # Adjust x and y ranges
                    height, width = im.shape
                    ax.set_autoscale_on(False)
                    ax.set_xlim([0, width])
                    ax.set_ylim([0, height])

                    if is_grayscale is True:
                        ax.imshow(im, cmap=plt.get_cmap('gray'), interpolation='none', vmin=vmin, vmax=vmax)
                    else:
                        ax.imshow(im, interpolation='none', vmin=vmin, vmax=vmax)

                # Plot a contour
                contour = contour3d[idx]
                if not np.all(contour == 0):
                    if contour.shape[0] > 1:
                        ax.plot(contour[:, 0], contour[:, 1], 'r')
                    elif contour.shape[0] > 0:
                        ax.plot(contour[:, 0], contour[:, 1], 'ro')

                # Invert y axis
                ax.invert_yaxis()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        _imshow(ax1, image3d, 0, img_vmin, img_vmax)
        _imshow(ax2, label3d, 0, lb_vmin, lb_vmax)

        axcolor = 'yellow'
        ax_slider = plt.axes([0.20, 0.02, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, 'Image#', 1, len(image3d), valinit=0, valstep=1)

        def _update(val):
            idx = int(val - 1)
            _imshow(ax1, image3d, idx, img_vmin, img_vmax)
            _imshow(ax2, label3d, idx, lb_vmin, lb_vmax)
            fig.canvas.draw_idle()

        # Set on_changed callback
        slider.on_changed(_update)

        # Set the tile
        if len(title) > 0:
            plt.title(title)

        plt.show(block=True)
