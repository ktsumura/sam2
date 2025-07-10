"""
DB reader

@author: ktsumura
"""
import os
from abc import ABCMeta

class DbReader(object):
    """
    DB reader
    """
    __metaclass__ = ABCMeta

    def __init__(self, db_path):
        self._db_name = os.path.split(db_path)[-1]
        self._db_path = db_path
        self._id_list = None
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def __str__(self):
        return '{}'.format(self._db_name)
