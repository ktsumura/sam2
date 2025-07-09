"""
DB reader state

@author: ktsumura
"""
import math
import random
from itertools import repeat
from typing import List

import h5py
import numpy as np

from training.neo_dataset.neo_data_info import NeoDataInfo
from training.neo_dataset.sam2_contour_type import SAM2ContourType


class DbReaderState(object):
    """
    This class keeps the state of DB reader and controls which data should be consumed in each training phase.
    """

    def __init__(self,
                 db_config_path_list,
                 num_gpus,
                 batch_size,
                 epoch_size,
                 shuffle=True):
        """
        Constructor
        """
        # Create DB path list
        self._db_path_list = self._create_db_path_list(db_config_path_list)
        self._num_gpus = num_gpus
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Epoch-related parameters
        self._epoch_index = 0
        self._num_epochs = epoch_size

        # Get the number of DB files
        self._db_index = 0
        self._num_dbs = len(self._db_path_list)

        # Create DB reader state for each DB file
        self._db_state_dict = self._create_db_state_dict(num_gpus, batch_size, shuffle)

        # The NEO data id list that contains data
        self._data_info_list: List[NeoDataInfo] = list()

    def _create_db_path_list(self, db_config_path_list):
        """
        Create a DB path list
        :param db_config_path:
        :return:
        """
        db_path_list = []

        for db_config_path in db_config_path_list:
            with open(db_config_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    db_path = line.strip()
                    db_path_list.append(db_path)

        return db_path_list

    def _create_db_state_dict(self, num_gpus, batch_size, shuffle):
        db_state_dict = {}
        for db_path in self._db_path_list:
            state = self._create_reader_state(db_path, num_gpus, batch_size, shuffle)
            db_state_dict[db_path] = state

        return db_state_dict

    def _create_reader_state(self, db_path, num_gpus, batch_size, shuffle):
        return _DbReaderState(db_path, num_gpus, batch_size, shuffle)

    def shuffle(self):
        if self._shuffle:
            for _, state in self._db_state_dict.items():
                state.shuffle_data_id_list_dict()

    def calc_steps_per_epoch(self):
        """
        Calculate the number of steps per epoch
        :return:
        """
        steps_per_epoch = 0
        for _, state in self._db_state_dict.items():
            steps_per_epoch += state.get_steps_per_epoch()

        return steps_per_epoch

    def init_epoch(self, epoch_index):
        self.epoch_index = epoch_index
        self.shuffle()
        self._init_data_info_list()

    def _init_data_info_list(self):
        """
        Initialize the data info list
        """
        # Create a list of batched data id, where an element corresponds to a batch,
        # and shaffle it, so that the 3d slice and 3 phase data elements are selected randomly.
        batched_data_id_list = list()
        for _, state in self._db_state_dict.items():

            data_info_list_in_batch = list()
            for data_idx in range(0, state.get_effective_num_samples()):
                data_info_list_in_batch.append(state.calc_data_info(data_idx))

                if (len(data_info_list_in_batch) == self._batch_size) and self._is_valid_batch(data_info_list_in_batch):
                    batched_data_id_list.append(data_info_list_in_batch)
                    data_info_list_in_batch = list()

        random.shuffle(batched_data_id_list)

        # Unfold the batched data ID list
        self._data_info_list = list()
        for batched_data_info in batched_data_id_list:
            self._data_info_list.extend(batched_data_info)

    def _is_valid_batch(self, data_id_list_in_batch: List[NeoDataInfo]):
        is_valid = True
        depth = None
        for data_id in data_id_list_in_batch:
            depth = data_id.depth if depth is None else depth
            if data_id.depth != depth:
                is_valid = False
                raise Exception('The depth of data elements in a batch must be the same.')

        return is_valid

    def get_neo_data_info(self, data_idx) -> NeoDataInfo:
        if data_idx < len(self._data_info_list):
            return self._data_info_list[data_idx]
        else:
            raise Exception('The data index exceeds the length of the data ID list.')

    @property
    def db_path_list(self):
        return self._db_path_list

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epoch_index(self):
        return self._epoch_index

    @epoch_index.setter
    def epoch_index(self, e):
        self._epoch_index = e

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def db_index(self):
        return self._db_index

    @db_index.setter
    def db_index(self, d):
        self._db_index = d

    @property
    def num_dbs(self):
        return self._num_dbs

    @property
    def db_state_dict(self):
        return self._db_state_dict

class _DbReaderState(object):
    def __init__(self, db_path, num_gpus, batch_size, shuffle=True):
        self._db_path = db_path
        self._shuffle = shuffle
        self._num_gpus = num_gpus
        self._batch_size = batch_size
        self._epoch_index = 0

        self._data_info_list_dict = {}
        self._actual_num_samples_dict = {}
        self._effective_num_samples_dict = dict()
        self._num_steps_per_epoch_dict = dict()
        self._is_sorted_by_depth = False

        # Create a dictionary of data ID lists
        self._create_data_id_list_dict()

        # Calculate a dictionary of the number of data elements
        self._calc_actual_num_samples_dict()
        self._calc_number_of_steps_per_epoch_and_effective_number_of_samples()

    def shuffle_data_id_list_dict(self):
        # Shuffle the ID list
        if self._shuffle:
            for _, data_info_list in self._data_info_list_dict.items():
                random.shuffle(data_info_list)

    def _create_data_id_list_dict(self):
        self._data_info_list_dict = {}
        self._is_sorted_by_depth = False

        # Get the length of the data set
        with h5py.File(self._db_path, 'r') as db_images_labels:
            len_list = list()
            len_list.append(db_images_labels["image"].len())
            len_list.append(db_images_labels["label"].len())

            length = len_list[0]
            repeated = list(repeat(length, len(len_list)))
            if len_list == repeated:
                # logger().info('{}'.format(self._db_path))
                # logger().info('dataset size: {}'.format(length))
                net_dim = self._get_net_dim_str(db_images_labels)

                # Sort data ID list by depth,
                # as the shape of images in a batch must be the same.
                for data_id in list(range(0, length)):
                    # Get depth
                    # 3D slice: phase
                    # 3D phase: slice
                    depth = self._calc_depth(db_images_labels, data_id)

                    # Get the contour type list
                    contour_type_list = self._calc_contour_type_list(db_images_labels, data_id)
                    if contour_type_list:
                        # Create a key
                        depth_key = str(depth)
                        if depth_key not in self._data_info_list_dict:
                            self._data_info_list_dict[depth_key] = []

                        # Add data ID
                        for sam2_contour_type, contour_type in contour_type_list:
                            self._data_info_list_dict[depth_key].append(NeoDataInfo(self._db_path, data_id, sam2_contour_type, contour_type, depth, net_dim))

                    self._is_sorted_by_depth = True
            else:
                raise Exception('ERROR!!! The size of data set is different.')

    def _calc_depth(self, db_images_labels, data_id):
        # Get the image index range from the DB
        image_index_range = db_images_labels['image_index_range'][data_id]

        # Note the last index is not included
        first_index = int(image_index_range[0, 0])
        last_index = int(image_index_range[0, 1])
        return last_index - first_index

    def _calc_contour_type_list(self, db_images_labels, data_id):
        contour_type_list = list()
        for sam2_contour_type in SAM2ContourType:
            contour_flag_key = sam2_contour_type.get_flag_key()
            contour_type_key = sam2_contour_type.get_contour_type_key()

            if contour_flag_key in db_images_labels \
                and contour_type_key in db_images_labels:
                contour_flag = db_images_labels[contour_flag_key][data_id]
                contour_type = db_images_labels[contour_type_key][data_id].decode('utf-8')

                if not np.all(contour_flag == 0):
                    contour_type_list.append((sam2_contour_type, contour_type))

        return contour_type_list

    def _get_net_dim_str(self, db_images_labels) -> str:
        net_dim_str = None
        for contour_type in SAM2ContourType:
            contour_key = contour_type.get_key()
            if (contour_key in db_images_labels) and ('NetworkDimMode' in db_images_labels[contour_key].attrs):
                net_dim_str = db_images_labels[contour_key].attrs['NetworkDimMode']
                break

        return net_dim_str

    def _calc_actual_num_samples_dict(self):
        self._actual_num_samples_dict = {}
        for depth_key, data_id_list in self._data_info_list_dict.items():
            self._actual_num_samples_dict[depth_key] = len(data_id_list)

    def _calc_number_of_steps_per_epoch_and_effective_number_of_samples(self):
        """
        Calculate the number of steps per epoch and the effective number of samples
        :return:
        """
        for d, actual_num_samples in self._actual_num_samples_dict.items():
            num_steps, effective_num_samples = self.calc_steps_and_samples_per_epoch(actual_num_samples)
            self._num_steps_per_epoch_dict[d] = num_steps
            self._effective_num_samples_dict[d] = effective_num_samples

    def calc_steps_and_samples_per_epoch(self, actual_num_samples):
        """
        Calculate the number of steps per epoch and the effective number of samples given the acutual number of samples

        The effective number of samples must be a multiple of the batch size if the DB is sorted by depth,
        because the shape of images in a batch must be the same.
        :param db_size:
        :return:
        """
        effective_batch_size = self._num_gpus * self._batch_size
        num_iters_per_epoch_all_gpus = self._num_gpus * int(math.ceil(float(actual_num_samples) / float(effective_batch_size)))
        num_samples_per_epoch_per_gpu = int(num_iters_per_epoch_all_gpus * self._batch_size)

        return num_iters_per_epoch_all_gpus, num_samples_per_epoch_per_gpu

    def calc_data_info(self, data_idx) -> NeoDataInfo:
        """
        Calculate the data ID corresponding to the data index

        """
        data_info = None

        cur_num_samples = 0
        for depth_key, data_info_list in self._data_info_list_dict.items():
            # Get the number of samples in the list before taking the batch size into account
            actual_num_samples = self._actual_num_samples_dict[depth_key]

            # Get the number of samples in the list after taking the batch size into account
            # The effective number of samples is always equal to or larger than the actual number of samples
            effective_num_samples = self._effective_num_samples_dict[depth_key]

            # Check if the data index corresponds to an element of the data ID list
            nxt_num_samples = cur_num_samples + effective_num_samples
            if data_idx < nxt_num_samples:
                data_idx_in_list = data_idx - cur_num_samples
                wrapped_data_idx = data_idx_in_list % actual_num_samples
                data_info = data_info_list[wrapped_data_idx]
                break

            cur_num_samples = nxt_num_samples

        if data_info is None:
            raise Exception("The data ID can not be null")

        return data_info

    def get_steps_per_epoch(self):
        return sum(self._num_steps_per_epoch_dict.values())

    def get_effective_num_samples(self):
        return sum(self._effective_num_samples_dict.values())

    @property
    def db_path(self):
        return self._db_path

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data_id_list_dict(self):
        return self._data_info_list_dict

    @property
    def db_size_dict(self):
        return self._actual_num_samples_dict

    @property
    def is_sorted_by_depth(self):
        return self._is_sorted_by_depth

    @property
    def data_index_dict(self):
        return self._data_index_dict

    @property
    def epoch_index(self):
        return self._epoch_index

    @epoch_index.setter
    def epoch_index(self, e):
        self._epoch_index = e
