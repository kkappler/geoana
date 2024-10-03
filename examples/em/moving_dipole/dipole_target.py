"""
    Module with functions to describe magnetic dipole targets.

    Note that dipole targets are completely described by three parameters: a.k.a. the dipole-moment vector.

    Of course there are a some real-world complications that we are skirting here
    1. the target is only approximatley a dipole
    2. The target may have remanent magnetization


"""
import copy

import pandas as pd
from geoana import utils
from geoana.em import static
from loguru import logger
from location import Location
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, Union
import discretize
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import xarray as xr

matplotlib.use('TkAgg')

B_FREMONT = np.array([5128., 22291., 41876.])  # magnetic field in nT

DEFAULT_GRID_PARAMETERS = {
    "x_min": -36,
    "x_max": 36,
    "nx": 100,
    "y_min": -36,
    "y_max": 36,
    "ny": 100,
}




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

    def __repr__(self):
        return  self.__str__()

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
        # Intialize some vars to None
        self._targets = None
        self._ambient_static_field = None
        self._grid_parameters = None
        self._ambient_static_field_orientation = None

        self.targets = targets  # TODO: support iterable of targets
        self.ambient_static_field = ambient_static_field

        self.grid_parameters = grid_parameters
        self.grid = self.grid_parameters.make_grid()

    def __str__(self):
        tmp = "Experiment Parameters"
        tmp += f"Targets: {self.targets}"
        tmp += f"Grid Parameters: {self.grid_parameters}"
        tmp += f"Ambient Static Field: {self.ambient_static_field}"
        return tmp
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @property
    def grid_parameters(self):
        return self._grid_parameters

    @grid_parameters.setter
    def grid_parameters(self, value):
        self._grid_parameters = value

    @property
    def ambient_static_field(self):
        return self._ambient_static_field

    @ambient_static_field.setter
    def ambient_static_field(self, value):
        self._ambient_static_field = value


    @property
    def ambient_static_field_orientation(self) -> np.ndarray:
        """ Returns a unit vector pointing in the direction of the ambient field"""
        if self._ambient_static_field_orientation  is None:
            orientation = self.ambient_static_field / np.linalg.norm(self.ambient_static_field)
            self._ambient_static_field_orientation = orientation
        return self._ambient_static_field_orientation


class GridParameters():
    """
    TODO: replace make_grid with discretize.TensorMesh
    
    """
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
        xyz = utils.ndgrid([x, y, np.r_[-1.]])
        logger.debug("WARNING: Check out hardcoded z=1.0 in make_grid ")
        return xyz


class Transect():
    """
    Class to hold r(t), a 3-dimensional path, with a time axis bound to it
    We will use a constant sample_rate.
    Use xarray


    """
    def __init__(
            self,
            x,
            y,
            z,
            time,
            label: Optional[str] = "",
            start_point: Optional[Union[Location, np.ndarray, None]] = None,
            # sample_rate,
    ):
        self._start_point = start_point
        self.label = label
        self.path = None  # TODO: make this an xarray
        n_observations = 100
        data = {
            "x": x,
            "y": y,
            "z": z,
            "time": time,
        }
        df = pd.DataFrame(data=data)
        df.set_index("time", inplace=True)
        path = df.to_xarray()
        self.path = path

    def start_point(self):
        if self._start_point is None:
            self._start_point = Location(self.path.x[0], self.path.y[0], self.path.z[0])



class DipoleField():
    def __init__(
            self,
            experiment_parameters: ExperimentParameters
    ):
        # Init some class vars
        self._b_vec = None
        self._b_total = None

        self.params = experiment_parameters
        # TODO: could make this iterate over targets -- for now take the zero'th

        target = self.params.targets[0]
        self.dipole = static.MagneticDipoleWholeSpace(
            location=target.location,
            orientation=self.params.ambient_static_field_orientation,
            moment=self.params.targets[0].dipole_moment
        )

    @property
    def b_vec(self):
        return self._b_vec

    @property
    def b_total(self):
        return self._b_total

    def compute_field(self):
        self._b_vec = self.dipole.magnetic_flux_density(self.params.grid)
        self._b_total = self.dipole.dot_orientation(self.b_vec)
        return

    def plot_total_field_amplitude(
            self,
            savefig_path=None,
            transects=None
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # plot dipole vector potential
        # _plot_amplitude(ax[0], self.b_total)
        _plot_amplitude(
            ax,
            self.b_total,
            self.params.grid_parameters.x_nodes,
            self.params.grid_parameters.y_nodes,
        )
        colors = ["blue", "orange", "green", "red", "purple", "pink", "brown"]

        if transects is not None:
            if isinstance(transects, dict):
                logger.warning("OLD Hacky way of adding transect overlays")
                try:
                    y_transects = transects["y"]
                except KeyError:
                    msg = "no y transects found"
                    y_transects = None
                xlim = ax.get_xlim()
                for y_transect in y_transects:
                    ax.plot(
                        np.asarray([xlim[0], xlim[1]]),
                        np.asarray([y_transect, y_transect]),
                        label=f"{y_transect:.2f}",
                        linewidth=2,
                    )
            else:
                for transect in transects:
                    ax.plot(
                        transect.path.x,
                        transect.path.y,
                        label=f"{transect.label}",
                        linewidth=2,
                    )

                #plt.hlines(y_transects, xlim[0], xlim[1], colors=colors, linestyles= )
        #ax[0].set_title("Total field: dipole")
        plt.legend()
        if savefig_path:
            plt.savefig(savefig_path)
        plt.show()

    def plot_transects(
            self,
            x = None,
            y_indices = None,  # TODO: deprecate
            transects = None,
    ):
        """
        Tool intended to make a pair of plots.
        1. total field amplitude (with some transects over it)
        2. the field values at the transects

        :param x:
        :param y:
        :return:
        """
        fig, ax  = plt.subplots(figsize=(8, 8))
        # fig, ax = plt.subplots(figsize=(8, 6))
        if y_indices is not None:
            msg = "Using old hack method for transect definition"
            logger.warning(msg)
            for y_ind in y_indices:
                ax.plot(self.params.grid_parameters.x_nodes,
                        self.b_total[y_ind * 100: (y_ind + 1) * 100],
                        label=f"{self.params.grid_parameters.y_nodes[y_ind]:.1f}m"
                        )

        if transects is not None:
            for transect in transects:
                msg = "TODO: Add new transect plotter"
                logger.error(msg)
        #for n in [0, 7, 13, 23, 33, 43, 44]:
        #     print(x[n], y[n])
        #     ax.plot(y[0:100], b_total_dipole[n * 100: (n + 1) * 100], label=f"{x[n]:.1f}m")
        # 
        fig.suptitle("Typical traces of Total Magnetic Field Intensity as a target passes")
        fig.text(0.5, 0.915, "(Legend denotes lateral offset between target and receiver)",
                  horizontalalignment="center")
        # # ax.set_title("\n\n Legend denotes lateral offset between target and receiver", fontsize=10)
        plt.legend()
        plt.xlabel("Distance from target (m)")
        plt.ylabel("Magnetic Field Intensity [T]")
        plt.savefig("transects.png")
        plt.show()

def _plot_amplitude(ax, v, x, y):
    """

    :type ax: matplotlib.axes._subplots.AxesSubplot
    :param ax: Axes object to receive the plot

    :param v:
    :return:
    """
    pcm = ax.pcolormesh(
            x, y, v.reshape(len(x), len(y), order='F')
        )
    plt.colorbar(
        pcm,
        ax=ax,
        pad=0.05,
        fraction=0.046
        # cax=inset_axes(ax, width="5%", height="90%", loc="right")
    )
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    ax.axis('square')
    ax.set_xlabel('y (east,  m)')
    ax.set_ylabel('x (north,  m)')
    ax.set_title("Total field: dipole")
    # ax[1].set_title("Total field: pole")

    # format so text doesn't overlap
    plt.tight_layout()


def test_grid_parameters():
    """

    :return:
    """
    params = DEFAULT_GRID_PARAMETERS
    grid_params = GridParameters()
    grid_params.from_dict(DEFAULT_GRID_PARAMETERS)
    grid = grid_params.make_grid()
    # print(grid)
    params = DEFAULT_GRID_PARAMETERS.copy()
    params["x_min"] = -5.0
    params["x_max"] = 5.0
    params["y_min"] = -5.0
    params["y_max"] = 5.0
    grid_params.from_dict(params)
    grid = grid_params.make_grid()
    # print(grid_params)
    # print(grid)
    return grid_params

def test_experiment_params():
    """ 
        This is supposed to be an example of a class that has all the information 
         needed to generate dipole fields for one or more targets as a numpy array.
    """
    experiment_params = ExperimentParameters(
        targets=[test_target(),],
        grid_parameters = test_grid_parameters()
    )
    # Write out default parameters in a dict
    return experiment_params

def test_target():
    target = MagneticDipoleTarget(
        dipole_moment=1.0,
        location= [0, 0, -10.0]
    )
    return target


def test_transect(
        start_point: Optional[Union[Location, np.ndarray, None]] = None,

):
    """

    :return:
    """
    if start_point is None:
        start_point = Location(
            x=-36,
            y=-0.05,
            z = 0,
        )
    end_point = Location(
        x=36,
        y=-0.05,
        z = 0,
    )
    # end_point = start_point +np.array([72., 0., 0.])
    n_observations = 100
    speed = 1.0
    t0 = 0
    
    total_displacement = (end_point - start_point).to_array()
    total_distance = np.linalg.norm(total_displacement)
    # v  = d / t
    # t = d / v
    total_time = total_distance / speed
    time = np.linspace(t0, total_time+t0, n_observations)
    x = np.linspace(start_point.x, end_point.x, n_observations)
    y = np.linspace(start_point.y, end_point.y, n_observations)
    z = np.linspace(start_point.z, end_point.z, n_observations)
    # an example of a straight line transect
    
    transect = Transect(
        x=x, y=y, z=z, time=time, label= "0.05m"
    )
    return transect

def test_reference_dipole():
    params = test_experiment_params()
    dipole = DipoleField(experiment_parameters=params)
    dipole.compute_field()

    # HACKY TRANSECTS
    # Each transect is a curve r(t) = x(t)i_hat + y(t) j_hat + z(t) k_hat
    # These are hackaround transects defined by selecting a single index of the y-axis and holding it
    # constant, fixed z and letting x run.
    transect_location_y_indices = [1,7,13,23, 33, 43,44]  # hacky assumes nx=ny=100
    y_transects = dipole.params.grid_parameters.y_nodes[transect_location_y_indices]

    dipole.plot_total_field_amplitude(
        savefig_path="test.png",
        transects = {"y":y_transects}
    )
    dipole.plot_transects(
        y_indices=transect_location_y_indices
    )
    
    # LESS HACKY TRANSECTS
    transect = test_transect()
    dipole.plot_total_field_amplitude(
        savefig_path="test.png",
        transects=[transect, ]
    )

    print("TODO: Convert transect into a parametric s(t)")
    # Assuming constant velocity
    time = dipole.params.grid_parameters.x_nodes  # equivalent to 1m/s
    x_of_t = dipole.params.grid_parameters.x_nodes
    y_of_t = y_transects[0]
    #x_of_t =


def main():
    test_transect()
    test_target()
    test_experiment_params()
    test_grid_parameters()
    test_reference_dipole()





if __name__ == "__main__":
    main()

