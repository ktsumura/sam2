from training.neosoft.constants.sam2_contour_type import SAM2ContourType


class NeoDataInfo():
    def __init__(self,
                 db_path: str,
                 data_id: int,
                 sam2_contour_type: SAM2ContourType,
                 contour_type: str,
                 depth: int,
                 net_dim: str):
        self._db_path = db_path
        self._data_id = data_id
        # SAM2 contur type (C1, C2, C3...)
        self._sam2_contour_type = sam2_contour_type
        # Contour type (FXN_SAX_LV_ENDO, FXN_SAX_LV_EPI, FXN_SAX_RV_ENDO...)
        self._contour_type = contour_type
        self._depth = depth
        self._size = 128
        self._net_dim = net_dim

    def is_lax(self):
        return 'LAX' in self._contour_type

    @property
    def db_path(self):
        return self._db_path

    @property
    def data_id(self):
        return self._data_id

    @property
    def sam2_contour_type(self):
        return self._sam2_contour_type

    @property
    def contour_type(self):
        return self._contour_type

    @property
    def depth(self):
        return self._depth

    @property
    def net_dim(self):
        return self._net_dim

    @property
    def size(self):
        return self._size
