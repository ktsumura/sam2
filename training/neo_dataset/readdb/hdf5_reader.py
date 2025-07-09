"""
HDF5 reader

@author: ktsumura
"""
import random
from abc import ABCMeta

import h5py
import numpy as np
import torch
import torchvision.transforms.v2.functional as funcv2
from monai.transforms import RandGaussianSmoothd, RandGaussianNoised, RandScaleIntensityd, RandScaleIntensityFixedMeand, \
    RandAdjustContrastd

from training.neo_dataset.neo_data_info import NeoDataInfo
from training.neo_dataset.neo_data_point import NeoDataPoint
from training.neo_dataset.readdb.db_reader import DbReader
from training.neo_dataset.sam2_contour_type import SAM2ContourType


class HDF5Reader(DbReader):
    """
    HDF5 reader
    """
    __metaclass__ = ABCMeta
    """
    HDF5 writer
    """
    def __init__(self, db_path, db_reader_state, transforms, image_size):
        DbReader.__init__(self, db_path)
        self._state = db_reader_state
        self._db_images_labels = self._read_db()
        self._transforms = transforms
        self._image_size = image_size
        self._max_num_frames = 10

    def __getitem__(self, neo_data_info: NeoDataInfo):
        return self._read(self._db_images_labels, neo_data_info)

    def _read_db(self):
        return h5py.File(self._db_path, 'r')

    def _close_db(self, db):
        db.close()

    def _read(self, db_images_labels, neo_data_info: NeoDataInfo):
        # Get the image from the DB
        image = self._read_image(db_images_labels, neo_data_info.data_id)

        # Get the contour from the DB
        contour, contour_flag = self._read_contour(db_images_labels, neo_data_info.data_id, neo_data_info.sam2_contour_type)

        # Get the image index range from the DB
        image_index_range = self._read_image_index_range(db_images_labels, neo_data_info.data_id)

        # Pre-process image, label, etc.
        return self._preprocess(image,
                                contour,
                                contour_flag,
                                image_index_range,
                                neo_data_info)

    # noinspection PyMethodMayBeStatic
    def _read_image(self, db_images_labels, data_id):
        return db_images_labels['image'][data_id]

    # noinspection PyMethodMayBeStatic
    def _read_label(self, db_images_labels, data_id):
        return db_images_labels['label'][data_id]

    # noinspection PyMethodMayBeStatic
    def _read_contour(self, db_images_labels, data_id: int, sam2_contour_type: SAM2ContourType):
        contour = None

        contour_key = sam2_contour_type.get_key()
        contour_flag_key = sam2_contour_type.get_flag_key()

        if contour_key in db_images_labels \
                and contour_flag_key in db_images_labels:
            contour_flag = db_images_labels[contour_flag_key][data_id]
            if not np.all(contour_flag == 0):
                contour = db_images_labels[contour_key][data_id]

        return contour, contour_flag

    # noinspection PyMethodMayBeStatic
    def _read_image_index_range(self, db_images_labels, data_id):
        return db_images_labels['image_index_range'][data_id]

    # noinspection PyMethodMayBeStatic
    def _create_meta_label_dict(self, db_images_labels, data_id):
        return None

    def _preprocess(self, image, contour, contour_flag, image_index_range, neo_data_info):
        # Extract
        # image: [1, depth, height, width]
        # contour: [1, depth, length, 2(=x,y)]
        image = self._extract_volume(image_index_range, image)
        contour = self._extract_volume(image_index_range, contour)
        contour_flag = self._extract_volume(image_index_range, contour_flag)
        datapoint = NeoDataPoint(image, contour, neo_data_info)

        # Sanity check
        if self._sanity_check_image(image):
            # Augment
            with torch.no_grad():
                for transform in self._transforms.transforms:
                    if self._is_monai(transform):
                        self._monai_transform(transform, datapoint)
                    else:
                        datapoint = transform(datapoint)

            # Reorder
            # The first frame must have a ground truth contour
            self._reorder(datapoint, contour_flag, neo_data_info.net_dim)

            # Limit the number of frames to avoid OOM
            self._clip_frames(datapoint)

            # Resize
            self._resize(datapoint)

            # Create label (mask)
            datapoint.create_label()
        else:
            raise Exception('Failed the image sanity check.')

        return datapoint

    def _extract_volume(self, image_index_range, volume):
        if volume is None:
            return None

        # Note the last index is not included
        first_index = int(image_index_range[0, 0])
        last_index = int(image_index_range[0, 1])
        return volume[:, first_index:last_index, ...]

    def _sanity_check_image(self, image):
        """
        Sanity-check the image
        :param image:
        :return:
        """
        # if images are black, it may cause Nan loss. Return None and don't pass this dataset for training
        return False if np.all(image == 0) else True

    def _is_monai(self, transform):
        return isinstance(transform, RandGaussianSmoothd) \
            or isinstance(transform, RandGaussianNoised) \
            or isinstance(transform, RandScaleIntensityd) \
            or isinstance(transform, RandScaleIntensityFixedMeand) \
            or isinstance(transform, RandAdjustContrastd)

    def _monai_transform(self, transform, datapoint):
        if isinstance(transform, RandGaussianSmoothd):
            # Transform per frame
            # The edge frames look darker if 3D smoothing is used even though sigma_z is 0.0.
            image_list = list()
            for image, _ in datapoint.get_frames():
                out_dict = transform({
                    'data': image
                })
                # Use torch.as_tensor to convert monai.data.meta_tensor.MetaTensor to torch.Tensor
                image_list.append(out_dict['data'].as_tensor())

            datapoint.image = torch.stack(image_list, axis=1)
        else:
            out_dict = transform({
                'data': datapoint.image
            })

            # Use torch.as_tensor to convert monai.data.meta_tensor.MetaTensor to torch.Tensor
            datapoint.image = out_dict['data'].as_tensor()

    def _reorder(self, datapoint, contour_flag, net_dim):
        """
        Re-order the frames using alternating nearest neighbor order
        The first frame must have a ground truth
        If the first frame is frame#10, the order will be...
        10, 9, 11, 8, 12, 7, 13, 6, 14....
        """
        # As contour_flag is [1, depth], gt_indices returns tuple of 2,
        # where the second represents column indexes of nonzero elements.
        gt_indices = np.nonzero(contour_flag)
        gt_indices = [int(x) for x in gt_indices[1]]
        gt_index = random.choice(gt_indices)
        _, num_frames = contour_flag.shape
        wrap = net_dim == '3D_SLICE'
        order = self._alternating_nearest_neighbor_order(num_frames, gt_index, wrap)

        order_tensor = torch.tensor(order, dtype=torch.int, device=datapoint.device)
        datapoint.image = datapoint.image[:, order_tensor, ...]
        datapoint.contour = datapoint.contour[:, order_tensor, ...]

        return datapoint

    def _alternating_nearest_neighbor_order(self, num_frames, start_index, wrap=False):
        """
        Generate an order starting from start_index, alternating nearest neighbors.

        Returns:
            list of int: ordered indices
        """
        order = [start_index]
        offset = 1

        while len(order) < num_frames:
            left = start_index - offset
            left = left + num_frames if wrap and (left < 0) else left

            right = start_index + offset
            right = right - num_frames if wrap and (right >= num_frames) else right

            if (left >= 0) and (left < num_frames) and (left not in order):
                order.append(left)
            if (right >= 0) and (right < num_frames) and (right not in order):
                order.append(right)

            offset += 1

        return order

    def _clip_frames(self, datapoint):
        if datapoint.neo_data_info.depth > self._max_num_frames:
            datapoint.image = datapoint.image[:, 0:self._max_num_frames, ...]
            datapoint.contour = datapoint.contour[:, 0:self._max_num_frames, ...]

    def _resize(self, datapoint: NeoDataPoint):
        resized_image_list = list()
        resized_contour_list = list()

        for image, contour in datapoint.get_frames():
            resized_image_list.append(funcv2.resize(image, self._image_size, antialias=True))

            resized_contour = torch.zeros_like(contour)
            scale_x = self._image_size / image.shape[-1]
            scale_y = self._image_size / image.shape[-2]
            resized_contour[..., 0] = contour[..., 0] * scale_x
            resized_contour[..., 1] = contour[..., 1] * scale_y
            resized_contour_list.append(resized_contour)

        datapoint.image = torch.stack(resized_image_list, dim=1)
        datapoint.contour = torch.stack(resized_contour_list, dim=1)

    @property
    def state(self):
        return self._state
