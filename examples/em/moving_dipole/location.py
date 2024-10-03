"""
    Tools for locations
    TODO: mtpy or mth5 has something like this already -- check it and consider replacing.
"""
from loguru import logger
import copy
import numpy as np

class Location():
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def __str__(self):
        tmp = f"Location: {self.x}X {self.y}Y {self.z}Z"
        return tmp

    def __repr__(self):
        return  self.__str__()

    def _clone(self):
        return copy.deepcopy(self)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    @property
    def z(self):
        return self._z

    def __add__(self, other):
        tmp = self._clone()
        if isinstance(other, Location):
            tmp._x += other._x
            tmp._y += other._y
            tmp._z += other._z
        elif isinstance(other, np.ndarray):
            tmp._x += other[0]
            tmp._y += other[1]
            tmp._z += other[2]
        else:
            msg = f"__add__ not defined for object of type {type(other)}"
            logger.error(msg)
            raise TypeError(msg)
        return tmp

    def __sub__(self, other):
        if isinstance(other, Location):
            xyz = other.to_array()
        else:
            xyz = other

        return self.__add__(-1. * xyz)

    def to_array(self):
        return np.array([self.x, self.y, self.z])