"""
Image type
"""
from enum import Enum, auto


class SAM2ContourType(Enum):
    """
    Image type
    """
    C1 = auto()
    C2 = auto()
    C3 = auto()
    C4 = auto()
    C5 = auto()
    C6 = auto()
    C7 = auto()
    C8 = auto()
    C9 = auto()
    C10 = auto()

    def to_integer(self):
        return self.value - 1

    def __str__(self):
        if self == SAM2ContourType.C1:
            return 'C1'
        if self == SAM2ContourType.C2:
            return 'C2'
        if self == SAM2ContourType.C3:
            return 'C3'
        if self == SAM2ContourType.C4:
            return 'C4'
        if self == SAM2ContourType.C5:
            return 'C5'
        if self == SAM2ContourType.C6:
            return 'C6'
        if self == SAM2ContourType.C7:
            return 'C7'
        if self == SAM2ContourType.C8:
            return 'C8'
        if self == SAM2ContourType.C9:
            return 'C9'
        if self == SAM2ContourType.C10:
            return 'C10'

        return None

    def get_key(self):
        return str(self)

    def get_flag_key(self):
        return str(self) + '_flag'

    def get_contour_type_key(self):
        return str(self) + '_contour_type'

    __repr__ = __str__
