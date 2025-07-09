"""
DB reader to read multiple HDF5 files

@author: ktsumura
"""

from training.dataset.transforms import ComposeAPI
from training.neo_dataset.readdb.hdf5_reader import HDF5Reader


class HDF5MultiReader:
    """
    HDF5 writer
    """

    def __init__(self, db_reader_state, transforms: ComposeAPI, image_size):
        self._state = db_reader_state
        self._reader_dict = dict()
        self._transforms = transforms
        self._image_size = image_size

    def __getitem__(self, data_idx):
        if self._state is None:
            return None, None

        # Get the DB path and data index in the DB
        neo_data_info = self._state.get_neo_data_info(data_idx)
        db_path = neo_data_info.db_path

        if db_path not in self._reader_dict:
            reader = self._get_hdf5_reader(db_path)
            self._reader_dict[db_path] = reader
        else:
            reader = self._reader_dict[db_path]

        return reader[neo_data_info]

    def _get_hdf5_reader(self, db_path):
        db_state = self._state.db_state_dict[db_path]
        return HDF5Reader(db_path, db_state, self._transforms, self._image_size)
