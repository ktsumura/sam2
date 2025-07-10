import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class PlotUtils:
    @staticmethod
    def display(image, contour=None, label=None, title='', is_grayscale=True, vmin=None, vmax=None):
        if not threading.current_thread() is threading.main_thread():
            return

        image3d = np.squeeze(image.detach().to('cpu').numpy(), axis=0)
        contour3d = np.squeeze(contour.detach().to('cpu').numpy(), axis=0) if contour is not None else None
        label3d = np.squeeze(label.detach().to('cpu').numpy(), axis=0) if label is not None else np.zeros_like(image3d)

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
                if contour3d is not None:
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
