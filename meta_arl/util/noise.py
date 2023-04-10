from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# from meta_arl.environments.abstract import EnvironmentBase
from meta_arl.util.myrandom import set_all_seeds

if TYPE_CHECKING:
    from meta_arl.environments.abstract import EnvironmentBase


class Noise2D:
    """A base class to create 2d noise over a specified box environment"""

    def __init__(self, min_corner: Iterable, max_corner: Iterable, seed: Optional[int] = None):
        """Instantiate the noise class

        Args:
            min_corner (Iterable): The smallest corner of the box environment.
            max_corner (Iterable): The largest corner of the box environment.
            seed (Optional[int], optional): A seed for the random number generator.
                Defaults to None.

        Raises:
            ValueError: If min_corner contains larger values than max_corner
        """
        if seed is not None:
            set_all_seeds(seed)

        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)

        if not (self.min_corner <= self.max_corner).all():
            raise ValueError("'min_corner' must be <= than 'max_corner'")

        self.x_min, self.y_min = self.min_corner
        self.x_max, self.y_max = self.max_corner

    def __call__(self, pos: Iterable) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position. Basically just performs a few safety checks and then calls the _get method.

        Args:
            pos (Iterable): An array of positions in the environment

        Raises:
            ValueError: If pos is not a numpy 1d/2d array
            ValueError: If the last dimension of pos is not 2
            ValueError: If not all values inside pos are inside the environment

        Returns:
            np.ndarray: An array of noise values
        """
        pos_np = np.array(pos)
        if len(pos_np.shape) > 2:
            raise ValueError("'pos' needs to be either a 1d or 2d array.")
        if pos_np.shape[-1] != 2:
            raise ValueError(
                f"The last dimension of 'pos' must be 2 but got {pos_np.shape[-1]} instead"
            )
        # Check that pos is inside the 2d range
        if not (self.min_corner <= pos_np).all() or not (pos_np <= self.max_corner).all():
            raise ValueError(
                "All values inside 'pos' must be >= than "
                f"{self.min_corner} and <= than {self.max_corner}"
            )

        # Make sure that 'pos' is a 2d array
        if len(pos_np.shape) == 1:
            pos_np = pos_np[np.newaxis, ...]

        return self._get(pos_np)

    def _get(self, pos: np.ndarray) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position. This is the method that needs to be implemented by the child classes.

        Args:
            pos (np.ndarray): An array of positions in the environment

        Returns:
            np.ndarray: An array of noise values
        """
        raise NotImplementedError()


class Noise2DPiecewise(Noise2D):
    """A smooth, continuous 2d noise function.
    Uses the idea from the video "Painting a Landscape with Maths", by  Inigo Quilez
    see here: https://www.youtube.com/watch?v=BFld4EBO2RE (starting at 02:00 minutes)
    """

    def __init__(
        self,
        min_corner: Iterable,
        max_corner: Iterable,
        num_rows: int,
        num_cols: int,
        min_val: int,
        max_val: int,
        seed: Optional[int] = None,
    ):
        """Instantiate the noise class

        Args:
            min_corner (Iterable): The smallest corner of the box environment.
            max_corner (Iterable): The largest corner of the box environment.
            num_rows (int): The number of rows in the noise grid
            num_cols (int): The number of columns in the noise grid
            min_val (int): The minimum noise value
            max_val (int): The maximum noise value
            seed (Optional[int], optional): A seed for the random number generator.
                Defaults to None.

        Raises:
            ValueError: If 'num_rows' or 'num_cols' is not a positive integer
            ValueError: If 'min_val' is larger than 'max_val'
        """
        super(Noise2DPiecewise, self).__init__(min_corner, max_corner, seed)
        self.num_rows = np.array(num_rows)
        self.num_cols = np.array(num_cols)
        self.min_val = np.array(min_val)
        self.max_val = np.array(max_val)

        if (
            not np.issubdtype(self.num_rows.dtype, np.integer)
            or not self.num_rows > 0
            or not np.issubdtype(self.num_cols.dtype, np.integer)
            or not self.num_cols > 0
        ):
            raise ValueError(
                "'num_rows' and 'num_cols' need to be positive integers, but got "
                f"{num_rows=}, {num_cols=}."
            )

        if not self.min_val <= self.max_val:
            raise ValueError(
                "'min_val' must be <= than 'max_val' but got " f"{self.min_val=}, {self.max_val=}"
            )

        # Get the dimensions of a single grid cell
        self.cell_dims = (self.max_corner - self.min_corner) / (
            self.num_rows,
            self.num_cols,
        )

        # Build the grid of corner vertices
        self.grid_values = np.random.uniform(
            self.min_val, self.max_val, size=(self.num_rows + 1, self.num_cols + 1)
        )

    def _smooth_step(self, x: np.ndarray, a: int = 0, b: int = 1) -> np.ndarray:
        """Computes the noise at position x, y in a grid cell of size a, b.

        Args:
            x (np.ndarray): The position in the grid cell
            a (int, optional): The width of the grid cell. Defaults to 0.
            b (int, optional): The height of the grid cell. Defaults to 1.

        Returns:
            np.ndarray: The noise value at position x, y
        """
        theta = np.minimum(1, np.maximum(0, (x - a) / (b - a)))
        return 3 * (theta**2) - 2 * (theta**3)

    def _get(self, pos: np.ndarray) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position.

        Args:
            pos (np.ndarray): An array of positions in the environment

        Returns:
            np.ndarray: An array of noise values
        """
        x_dim, y_dim = self.cell_dims

        # Compute for each tuple in pos the 4 corners of the tile it is contained in
        a_pos = ((pos - self.min_corner) // self.cell_dims).astype(np.int32)

        # a-corners can't take the maximum value. => fix that
        first_maximum = np.where(a_pos[:, 0] == len(self.grid_values) - 1)
        second_maximum = np.where(a_pos[:, 1] == len(self.grid_values[0]) - 1)
        a_pos[first_maximum] -= (1, 0)
        a_pos[second_maximum] -= (0, 1)

        # Compute positions of other vertices
        b_pos = (a_pos + (1, 0)).astype(np.int32)
        c_pos = (a_pos + (0, 1)).astype(np.int32)
        d_pos = (a_pos + (1, 1)).astype(np.int32)

        # Extract the corresponding heights of each corner
        a = self.grid_values[a_pos[:, 0], a_pos[:, 1]]
        b = self.grid_values[b_pos[:, 0], b_pos[:, 1]]
        c = self.grid_values[c_pos[:, 0], c_pos[:, 1]]
        d = self.grid_values[d_pos[:, 0], d_pos[:, 1]]

        # Compute the two smooth steps
        x_smooth_step = self._smooth_step(
            pos[:, 0] - self.min_corner[0] - a_pos[:, 0] * x_dim, 0, x_dim
        )
        y_smooth_step = self._smooth_step(
            pos[:, 1] - self.min_corner[1] - a_pos[:, 1] * y_dim, 0, y_dim
        )

        # Compute the final noise values
        noise_values = (
            a
            + (b - a) * x_smooth_step
            + (c - a) * y_smooth_step
            + (a - b - c + d) * x_smooth_step * y_smooth_step
        )

        return noise_values


class Noise2DRandom(Noise2D):
    """A pointwise noise class that generates random noise values for each point in the environment."""

    def __init__(
        self,
        min_corner: Iterable,
        max_corner: Iterable,
        mean: Iterable,
        std: Iterable,
        seed: Optional[int] = None,
    ):
        """Instantiate the noise class

        Args:
            min_corner (Iterable): The smallest corner of the box environment.
            max_corner (Iterable): The largest corner of the box environment.
            mean (Iterable): The mean of the noise distribution
            std (Iterable): The standard deviation of the noise distribution
            seed (Optional[int], optional): A seed for the random number generator.
                Defaults to None.

        Raises:
            ValueError: If 'mean' is not a scalar value
            ValueError: I 'std' is not a scalar value
            ValueError: If 'std' is negative
        """
        super(Noise2DRandom, self).__init__(min_corner, max_corner, seed)

        self.mean = np.array(mean)
        self.std = np.array(std)

        if not len(self.mean.shape) == 0:
            raise ValueError("'mean' has to be a scalar value!")

        if not len(self.std.shape) == 0:
            raise ValueError("'std' has to be a scalar value!")

        if not self.std >= 0:
            raise ValueError("'std' needs to be a positive value!")

    def _get(self, pos: np.ndarray) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position.

        Args:
            pos (np.ndarray): An array of positions in the environment

        Returns:
            np.ndarray: An array of normal distributed noise values
        """
        return np.random.normal(self.mean, self.std, pos.shape[0])


# def get_noise(config: dict[str, Any], environment: Optional[EnvironmentBase] = None) -> Noise2D:
def get_noise(config: dict[str, Any], environment: Optional["EnvironmentBase"] = None) -> Noise2D:
    """Given a configuration dictionary, returns the corresponding noise class.

    Args:
        config (dict[str, Any]): A configuration dictionary
        environment (Optional[&quot;EnvironmentBase&quot;], optional): The environment the noise
            class is used in. Defaults to None.

    Raises:
        KeyError: If the configuration dictionary misses the 'name' key
        Exception: If the specified noise class is not implemented

    Returns:
        Noise2D: The noise class
    """
    # Check if the config contains a 'name' entry
    if "name" not in config:
        raise KeyError("The noise config requires an entry 'name'!")

    # Extract the name
    name_orig = config["name"]
    name = name_orig.lower()

    # Delete the name from the dictionary. This allows to pass the remaining
    # parameters directly to the initialization function of the reward model
    parameter_config = dict(config)
    del parameter_config["name"]

    # Get the observation space of the environment (if it is present)
    if environment is not None:
        obs = environment.observation_space

    noise: Union[Noise2DPiecewise, Noise2DRandom]
    if name == "noise2dpiecewise":
        noise = Noise2DPiecewise(obs.low, obs.high, **parameter_config)
    elif name == "noise2drandom":
        noise = Noise2DRandom(obs.low, obs.high, **parameter_config)
    else:
        raise Exception(f"No implementation of the noise class {name_orig} exists!")

    return noise


if __name__ == "__main__":
    min_corner = np.array([-5, -5])
    max_corner = np.array([5, 5])
    num_rows, num_cols = 8, 8
    min_val, max_val = -1, 1

    noise = Noise2DPiecewise(min_corner, max_corner, num_rows, num_cols, min_val, max_val)

    # noise = Noise2DRandom(min_corner, max_corner, 0, 1)

    x_points_grid = np.meshgrid(
        np.linspace(min_corner[0], max_corner[0], 500),
        np.linspace(min_corner[1], max_corner[1], 500),
    )
    x_points = np.array([np.concatenate(x_points_grid[0]), np.concatenate(x_points_grid[1])]).T

    y_points = noise(x_points)

    mesh = plt.pcolormesh(
        x_points_grid[0],
        x_points_grid[1],
        y_points.reshape(500, 500),
        shading="auto",
        alpha=0.6,
    )

    plt.colorbar(mesh)

    plt.show()
