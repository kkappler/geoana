import pytest
import numpy as np
import discretize
from geoana.em import fdem
from scipy.constants import mu_0, epsilon_0


class TestHarmonicPlaneWave:

    def test_defaults(self):
        frequencies = np.logspace(1, 4, 3)
        amplitude = 1.0
        sigma = 1.0
        w = 2 * np.pi * frequencies
        hpw = fdem.HarmonicPlaneWave(frequency=frequencies, amplitude=amplitude, sigma=sigma)
        assert np.all(hpw.frequency == np.logspace(1, 4, 3))
        assert hpw.amplitude == 1.0
        assert np.all(hpw.orientation == np.r_[1., 0., 0.])
        assert np.all(hpw.omega == w)
        assert np.all(hpw.wavenumber == np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0))
        assert hpw.sigma == 1.0
        assert hpw.mu == mu_0
        assert hpw.epsilon == epsilon_0

    def test_errors(self):
        frequencies = np.logspace(1, 4, 3)
        amplitude = 1.0
        hpw = fdem.HarmonicPlaneWave(frequency=frequencies, amplitude=amplitude)
        with pytest.raises(TypeError):
            hpw.frequency = "string"
        with pytest.raises(ValueError):
            hpw.frequency = -1
        with pytest.raises(TypeError):
            hpw.frequency = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            hpw.orientation = 1
        with pytest.raises(ValueError):
            hpw.orientation = np.r_[1., 0.]
        with pytest.raises(ValueError):
            hpw.orientation = np.r_[0., 0., 1.]

    def test_electric_field(self):
        frequencies = np.logspace(1, 4, 3)
        amplitude = 1.0
        hpw = fdem.HarmonicPlaneWave(frequency=frequencies, amplitude=amplitude)

        # test x orientation
        w = 2 * np.pi * frequencies
        k = np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        kz = np.outer(k, z)
        ikz = 1j * kz

        ex = amplitude * np.exp(ikz)
        ey = np.zeros_like(z)
        ez = np.zeros_like(z)

        np.testing.assert_equal(ex, hpw.electric_field(xyz)[0])
        np.testing.assert_equal(ey, hpw.electric_field(xyz)[1])
        np.testing.assert_equal(ez, hpw.electric_field(xyz)[2])

        # test y orientation
        hpw.orientation = 'Y'

        ex = np.zeros_like(z)
        ey = amplitude * np.exp(ikz)
        ez = np.zeros_like(z)

        np.testing.assert_equal(ex, hpw.electric_field(xyz)[0])
        np.testing.assert_equal(ey, hpw.electric_field(xyz)[1])
        np.testing.assert_equal(ez, hpw.electric_field(xyz)[2])

    def test_current_density(self):
        frequencies = np.logspace(1, 4, 3)
        amplitude = 1.0
        sigma = 2.0
        hpw = fdem.HarmonicPlaneWave(frequency=frequencies, amplitude=amplitude, sigma=sigma)

        # test x orientation
        w = 2 * np.pi * frequencies
        k = np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0 * 2)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        kz = np.outer(k, z)
        ikz = 1j * kz

        jx = 2 * amplitude * np.exp(ikz)
        jy = np.zeros_like(z)
        jz = np.zeros_like(z)

        np.testing.assert_equal(jx, hpw.current_density(xyz)[0])
        np.testing.assert_equal(jy, hpw.current_density(xyz)[1])
        np.testing.assert_equal(jz, hpw.current_density(xyz)[2])

        # test y orientation
        hpw.orientation = 'Y'

        jx = np.zeros_like(z)
        jy = 2 * amplitude * np.exp(ikz)
        jz = np.zeros_like(z)

        np.testing.assert_equal(jx, hpw.current_density(xyz)[0])
        np.testing.assert_equal(jy, hpw.current_density(xyz)[1])
        np.testing.assert_equal(jz, hpw.current_density(xyz)[2])

    """
    def test_magnetic_field(self):
        frequencies = np.logspace(1, 4, 3)
        amplitude = 1.0
        hpw = fdem.HarmonicPlaneWave(frequency=frequencies, amplitude=amplitude)

        # test x orientation
        w = 2 * np.pi * frequencies
        k = np.sqrt(w ** 2 * mu_0 * epsilon_0 - 1j * w * mu_0)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        kz = np.outer(k, z)
        ikz = 1j * kz
        Z = w * mu_0 / k

        hx = amplitude / Z * np.exp(ikz)
        hy = np.zeros_like(z)
        hz = np.zeros_like(z)

        np.testing.assert_equal(hx, hpw.magnetic_field(xyz)[0])
        np.testing.assert_equal(hy, hpw.magnetic_field(xyz)[1])
        np.testing.assert_equal(hz, hpw.magnetic_field(xyz)[2])

        # test y orientation
        hpw.orientation = 'Y'

        hx = np.zeros_like(z)
        hy = amplitude / Z * np.exp(ikz)
        hz = np.zeros_like(z)

        np.testing.assert_equal(hx, hpw.magnetic_field(xyz)[0])
        np.testing.assert_equal(hy, hpw.magnetic_field(xyz)[1])
        np.testing.assert_equal(hz, hpw.magnetic_field(xyz)[2])
    """








