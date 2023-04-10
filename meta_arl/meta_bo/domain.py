from typing import Optional

import numpy as np
from numpy.typing import NDArray


class ContinuousDomain:
    def __init__(self, l: NDArray[np.float64], u: NDArray[np.float64]):
        """A multi-dimensional continuous box-domain implementation.

        Args:
            l (NDArray[np.float64]): Lower corner of the domain.
            u (NDArray[np.float64]): Upper corner of the domain.
        """
        assert l.ndim == u.ndim == 1 and l.shape[0] == u.shape[0]
        assert np.all(l < u)
        self._l = l
        self._u = u
        self._range = self._u - self._l
        self._d = l.shape[0]
        self._bounds = np.vstack((self._l, self._u)).T

    @property
    def l(self) -> NDArray[np.float64]:
        return self._l

    @property
    def u(self) -> NDArray[np.float64]:
        return self._u

    @property
    def bounds(self) -> NDArray[np.float64]:
        return self._bounds

    @property
    def range(self) -> NDArray[np.float64]:
        return self._range

    @property
    def d(self) -> int:
        return self._d

    def normalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize x to [0, 1]^d

        Args:
            x (NDArray[np.float64]): The points to be normalized.

        Returns:
            NDArray[np.float64]: The normalized points.
        """
        return (x - self._l) / self._range

    def denormalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Denormalize x from [0, 1]^d to the domain.

        Args:
            x (NDArray[np.float64]): The points to be denormalized.

        Returns:
            NDArray[np.float64]: The denormalized points.
        """
        return x * self._range + self._l

    def project(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project X into domain rectangle.

        Args:
            X (NDArray[np.float64]): The points to be projected.

        Returns:
            NDArray[np.float64]: The projected points.
        """
        return np.minimum(np.maximum(X, self.l), self.u)

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def default_x0(self) -> NDArray[np.float64]:
        """The default starting point for the domain.

        Returns:
            NDArray[np.float64]: The default starting point.
        """
        return self._l + self._range / 2
        # use random initial point
        # return np.random.uniform(low=self.l, high=self.u, size=(1, self.d))


class DiscreteDomain:
    def __init__(self, points: NDArray[np.float64], d: Optional[int] = None):
        """A multi-dimensional discrete domain implementation.

        Args:
            points (NDArray[np.float64]): The points in the domain.
            d (Optional[int], optional): The dimension of the domain. Defaults to None.
        """
        if points.ndim == 1:
            # LUKAS FIX
            # points = np.expand_dims(points, axis=-1)
            points = np.expand_dims(points, axis=0)
        assert points.ndim == 2
        self._points = points
        if d is None:
            self._d = points[0].shape[0]
        else:
            self._d = d

    @property
    def points(self) -> NDArray[np.float64]:
        return self._points

    @property
    def d(self) -> int:
        return self._d

    @property
    def num_points(self) -> int:
        return len(self._points)

    def normalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    def denormalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def default_x0(self) -> NDArray[np.float64]:
        return self.points[0]
