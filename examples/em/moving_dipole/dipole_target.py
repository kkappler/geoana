"""
    Module with functions to describe magnetic dipole targets.

    Note that dipole targets are completely described by three parameters: a.k.a. the dipole-moment vector.

    Of course there are a couple real-world complications that we are skirting here
    1. the target is only approximatley a dipole
    2. The targe may have remnent magnetization
    3. The actual (even approximate dipole moment) for a given target In this case
    1. magnetic moment
    2. orienation(note that this will depend on the individual target, the earth magnetic field, and
         some other
"""
from geoana import utils
from typing import Optional, Union
from loguru import logger
import numpy as np



B_FREMONT = np.array([5128., 22291., 41876.])  # magnetic field in nT

DEFAULT_GRID_PARAMETERS = {
    "x_min": -36,
    "x_max": 36,
    "nx": 100,
    "y_min": -36,
    "y_max": 36,
    "ny": 100,
}

class Location():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        tmp = f"Location: {self.x}X {self.y}Y {self.z}Z"
        return tmp

class MagneticDipoleTarget(object):

    def __init__(
            self,
            dipole_moment: float,
            location: np.ndarray,  # 3 component XYZ vector
            **kwargs
    ):
        """
        - magnetic moment (note that this will depend on the individual target, the earth magnetic field, and
         some other
        """
        self._dipole_moment = dipole_moment
        self._location = location

    def __str__(self):
        tmp = "Magnetic Dipole\n"
        tmp += f"Moment: {self.dipole_moment} \n"
        tmp += f"Location: {Location(*self.location)}"
        return tmp

    @property
    def location(self):
        return self._location

    @property
    def dipole_moment(self):
        return self._dipole_moment

class ExperimentParameters():

    def __init__(
            self,
            targets: Optional[Union[list, tuple, None]] = None,
            ambient_static_field: Optional[np.ndarray] = B_FREMONT,
            grid_parameters: Optional[Union[dict, None]] = None,
            **kwargs
    ):
        """
        Constructor.

        """
        self.targets = targets  # TODO: support iterable of targets
        self.ambient_static_field = ambient_static_field
        #self.

        self._ambient_static_field_orientation = None
        self._grid_parameters = grid_parameters
        self.grid = self._grid_parameters.make_grid()
        self._targets = None
        self._ambient_static_field = None

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @property
    def ambient_static_field(self):
        return self._ambient_static_field

    @ambient_static_field.setter
    def ambient_static_field(self, value):
        self._ambient_static_field = value


    def ambient_static_field_orientation(self) -> np.ndarray:
        """ Returns a unit vector pointing in the direction of the ambient field"""
        if self._ambient_static_field_orientation  is None:
            orientation = self.ambient_static_field / np.linalg.norm(self.ambient_static_field)
            self._ambient_static_field_orientation = orientation
        return self._ambient_static_field_orientation


class GridParameters():
    PARAMETER_NAMES = [
        "x_min", "x_max", "dx", "nx", "y_min", "y_max", "dy", "ny"
    ]
    def __init__(self,
                 x_min: Optional[Union[float, None]] = None,
                 x_max: Optional[Union[float, None]] = None,
                 dx: Optional[Union[float, None]] = None,
                 nx: Optional[Union[int, None]] = None,
                 y_min: Optional[Union[float, None]] = None,
                 y_max: Optional[Union[float, None]] = None,
                 dy: Optional[Union[float, None]] = None,
                 ny: Optional[Union[int, None]] = None
                 ):
        self._x_min = x_min
        self._x_max = x_max
        self._dx = dx
        self._nx = nx
        self._y_min = y_min
        self._y_max = y_max
        self._dy = dy
        self._ny = ny

        # class vars
        self._x_nodes = None
        self._y_nodes = None

    def __str__(self):
        output = ""
        for k in self.PARAMETER_NAMES:
            output += f"{k}: {self.__getattribute__(k)}\n"
        return output

    def __repr__(self):
        return self.__str__()

    def from_dict(self, params: dict):
        for k, v in params.items():
            if k in self.PARAMETER_NAMES:
                self.__setattr__(k, v)



    # ---- X-axis Properties --- #

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def dx(self):
        if self._dx is None:
            dx = 1.0 * (self.x_max - self.x_min) / self.nx
            self._dx = dx
        return self._dx

    @property
    def nx(self):
        return self._nx

    @property  # TODO: consider merging x_nodes, y_nodes, z_nodes into a dict
    def x_nodes(self):
        if self._x_nodes is None:
            x_nodes = np.linspace(self.x_min, self.x_max, self.nx)
            self._x_nodes = x_nodes
        return self._x_nodes

    # ---- X-axis Setters --- #

    @nx.setter
    def nx(self, value):
        self._nx = value

    @x_min.setter
    def x_min(self, value):
        self._x_min = value

    @x_max.setter
    def x_max(self, value):
        self._x_max = value


    # ---- Y-axis Properties --- #

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @property
    def dy(self):
        if self._dy is None:
            self._dy = 1.0 * (self.y_max - self.y_min) / self.ny
        return self._dy

    @property
    def ny(self):
        return self._ny

    @property
    def y_nodes(self):
        if self._y_nodes is None:
            y_nodes = np.linspace(self.y_min, self.y_max, self.ny)
            self._y_nodes = y_nodes
        return self._y_nodes

    # ---- Y-axis Setters --- #

    @y_min.setter
    def y_min(self, value):
        self._y_min = value

    @y_max.setter
    def y_max(self, value):
        self._y_max = value

    @ny.setter
    def ny(self, value):
        self._ny = value

    def make_grid(self):
        x = self.x_nodes
        y = self.y_nodes
        xyz = utils.ndgrid([x, y, np.r_[1.]])
        logger.debug("WARNING: Check out hardcoded z=1.0 in make_grid ")
        return xyz

def test_grid_parameters():
    """

    :return:
    """
    params = DEFAULT_GRID_PARAMETERS
    grid_params = GridParameters()
    grid_params.from_dict(DEFAULT_GRID_PARAMETERS)
    grid = grid_params.make_grid()
    print(grid)
    params = DEFAULT_GRID_PARAMETERS.copy()
    params["x_min"] = -5.0
    params["x_max"] = 5.0
    params["y_min"] = -5.0
    params["y_max"] = 5.0
    grid_params.from_dict(params)
    grid = grid_params.make_grid()
    print(grid_params)
    print(grid)
    return grid_params

def test_reference_dipole():
    experiment_params = ExperimentParameters(
        targets=[test_target(),],
        grid_parameters = test_grid_parameters()
    )
    # Write out default parameters in a dict

    pass    # th

def test_target():
    target = MagneticDipoleTarget(
        dipole_moment=1.0,
        location= [0, 0, 0.0]
    )
    return target

def main():
    test_target()
    test_reference_dipole()
    test_grid_parameters()



if __name__ == "__main__":
    main()