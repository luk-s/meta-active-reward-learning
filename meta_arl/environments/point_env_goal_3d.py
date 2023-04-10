import itertools
from optparse import Option
from typing import Any, Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from meta_arl.environments.point_env_goal import PointEnv, PointTasks
from meta_arl.meta_bo.domain import ContinuousDomain
from meta_arl.reward_models.abstract import RewardModelGPBase


def sample_plane_point(
    point: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    obs_space: gym.spaces.Box,
    rng: np.random.Generator,
    num_points: int = 1,
    max_tries: int = 1000,
    homomorphic: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Randomly sample 'num_points' points from a plane defined by 'point' and 'vec1' and 'vec2'.
    Make sure that the sampled points are within the observation space.

    Args:
        point (np.ndarray): A point on the plane.
        vec1 (np.ndarray): A vector defining the first axis of the plane.
        vec2 (np.ndarray): A vector defining the second axis of the plane.
        obs_space (gym.spaces.Box): The observation space.
        rng (np.random.Generator): A random number generator.
        num_points (int, optional): The number of points to sample. Defaults to 1.
        max_tries (int, optional): The maximum number of tries to sample a point. Defaults to 1000.
        homomorphic (bool, optional): If True, the sampled points use homomorphic coordinates.
            Defaults to False.

    Raises:
        ValueError: If the plane specified by 'point', 'vec1' and 'vec2' does not intersect with
            the observation space or only intersects with the observation space in a single point.

    Returns:
        Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]: The sampled points, the corners of the
            observation space and the corners of the observation space in transformed plane
            coordinates.
    """
    # Create an orthogonal basis
    vec3 = np.cross(vec1, vec2)
    vec2 = np.cross(vec1, vec3)

    # Normalize the basis to make it orthonormal
    vec1_norm, vec2_norm, vec3_norm = (
        np.linalg.norm(vec1),
        np.linalg.norm(vec2),
        np.linalg.norm(vec3),
    )
    vec1 = vec1 / vec1_norm
    vec2 = vec2 / vec2_norm
    vec3 = vec3 / vec3_norm

    # Build the transformation matrix from the [vec1, vec2, vec3] coordinate system
    # to the [e1, e2, e3] coordinate system
    inv_transformation = np.vstack([vec1, vec2, vec3, point]).T
    inv_transformation = np.vstack([inv_transformation, np.zeros(len(vec1) + 1)])
    inv_transformation[-1][-1] = 1

    # Now, invert this matrix to get the transformation matrix from [e1, e2, e3] to
    # [vec1, vec2, vec3]
    transformation = np.linalg.inv(inv_transformation)

    # Get all 8 corner points of the cube (in homomorphic coordinates)
    limits = np.array([obs_space.low, obs_space.high]).T
    corners = np.array(list(itertools.product(*limits)))
    corners = np.hstack([corners, np.ones((len(corners), 1))])

    # Transform the cube corner coordinates into the new coordinate system of the plane
    corners_transformed = np.dot(transformation, corners.T).T

    # Each tuple represents the indices of two corners in the 'corners' and
    # 'corners_transformed' lists which are connected by a line. For example,
    # 'corners[0]' and 'corners[1]' are connected by a line.
    line_pairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    # Find all pairs of corners whose connecting vector intersects the plane
    intersection_points_list = []
    for corner1, corner2 in [
        (corners_transformed[i], corners_transformed[j]) for i, j in line_pairs
    ]:
        # If the multiplication of z the axis values is <= 0 this means that the
        # connecting line between corner1 and corner2 intersects the plane
        if corner1[2] * corner2[2] <= 0:
            # Find the point where the line intersects the plane
            connecting_vector = corner2 - corner1

            # Assume that corner1 + alpha * connecting_vector intersects the plane
            if corner1[2] != 0:  # corner1 does not lie on the plane
                alpha = -corner1[2] / connecting_vector[2]
                assert connecting_vector[2] != 0, "This should not happen!"
            else:  # corner1 lies on the plane
                alpha = 0

            # Get the resulting intersection point
            intersection_points_list.append(corner1 + alpha * connecting_vector)

    if len(intersection_points_list) <= 1:
        raise ValueError(
            "The specified plane does either not intersect the cube, "
            "or only intersects it in a single point!"
        )

    # Bound the intersection points with a rectangle
    intersection_points = np.array(intersection_points_list)
    max_point = np.max(intersection_points, axis=0)
    min_point = np.min(intersection_points, axis=0)

    bounding_box = gym.spaces.Box(low=min_point, high=max_point, seed=rng)

    # Sample from this bounding square until a point inside the cube has been found
    points_list = []
    for sample_idx in range(max_tries):
        sample = bounding_box.sample()

        sample_transformed = np.dot(inv_transformation, sample)
        if (sample_transformed[:-1] >= obs_space.low).all() and (
            sample_transformed[:-1] <= obs_space.high
        ).all():
            points_list.append(sample_transformed)
            if len(points_list) == num_points:
                points = np.array(points_list)

                if not homomorphic:
                    points = points[:, :-1]
                    corners = corners[:, :-1]
                    corners_transformed = corners_transformed[:, :-1]
                return points, corners, corners_transformed

    print(
        f"{max_tries} samplings only found {len(points)} points! "
        "Maybe try increase 'max_tries'?"
    )

    return None, corners, corners_transformed


class PointEnv3d(PointEnv):
    """A simple 3D point environment."""

    def __init__(
        self,
        rng: np.random.Generator,
        arena_size: float = 5.0,
        arena_dim: int = 3,
        done_bonus: int = 0,
        num_tasks: int = 1,
        max_episode_length: int = 200,
        action_dist: float = 0.1,
        target_pos: Optional[np.ndarray] = None,
        reward_model: Optional[RewardModelGPBase] = None,
        use_upper_conf_bound: bool = False,
        visualize: bool = False,
        plane_point: Optional[np.ndarray] = None,
        plane_vector1: Optional[np.ndarray] = None,
        plane_vector2: Optional[np.ndarray] = None,
    ):
        """Initialize a 3d point environment. The observation space is either a 3d cube or a
        2d plane intersecting a 3d cube.

        Args:
            rng (np.random.Generator): The random number generator.
            arena_size (float, optional): The length of the sides of the cube. Defaults to 5.0.
            arena_dim (int, optional): The dimension of the observation space. Defaults to 3.
            done_bonus (int, optional): The reward bonus for reaching the target. Defaults to 0.
            num_tasks (int, optional): The number of tasks of this environment. Defaults to 1.
            max_episode_length (int, optional): The maximum episode length. Defaults to 200.
            action_dist (float, optional): The distance the agent moves in one step. Defaults to
                0.1.
            target_pos (Optional[np.ndarray], optional): The target position. Defaults to None.
            reward_model (Optional[RewardModelGPBase], optional): The reward model. Defaults to
                None.
            use_upper_conf_bound (bool, optional): Whether to use the upper confidence bound of
                the reward model. Defaults to False.
            visualize (bool, optional): Whether to visualize the environment. Defaults to False.
            plane_point (Optional[np.ndarray], optional): A point on the plane. Defaults to None.
            plane_vector1 (Optional[np.ndarray], optional): A vector defining the plane. Defaults
                to None.
            plane_vector2 (Optional[np.ndarray], optional): A second vector defining the plane.
                Defaults to None.
        """
        super(PointEnv, self).__init__(rng)

        # DEBUG START
        self._reward_times = []
        # DEBUG END

        assert arena_dim == 3, (
            f"This environment requires 'arena_dim' == 3! "
            f"Got 'arena_dim'=={arena_dim} instead."
        )

        self._arena_size = arena_size
        self._arena_dim = arena_dim
        self._done_bonus = done_bonus
        self._num_tasks = num_tasks
        self._max_episode_length = max_episode_length
        self._use_upper_conf_bound = use_upper_conf_bound
        self._visualize = visualize

        self._observation_space = spaces.Box(
            -arena_size,
            arena_size,
            shape=[
                arena_dim,
            ],
            seed=int(rng.integers(1 << 32 - 1)),
        )
        self._action_space = spaces.Box(
            -action_dist,
            action_dist,
            shape=[
                arena_dim,
            ],
            seed=int(rng.integers(1 << 32 - 1)),
        )

        # Used for compatibility with pacoh
        self._domain = ContinuousDomain(
            l=np.repeat(-arena_size, arena_dim), u=np.repeat(arena_size, arena_dim)
        )

        self._task: PointTasks = self.sample_tasks(
            num_tasks=num_tasks,
            target_pos=target_pos,
            plane_point=plane_point,
            plane_vector1=plane_vector1,
            plane_vector2=plane_vector2,
        )

        self._reward_model = reward_model

    def sample_tasks(
        self,
        num_tasks: int,
        target_pos: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        plane_point: Optional[np.ndarray] = None,
        plane_vector1: Optional[np.ndarray] = None,
        plane_vector2: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> PointTasks:
        """Sample 'num_tasks' tasks for this environment.

        Args:
            num_tasks (int): The number of tasks to sample.
            target_pos (Optional[np.ndarray], optional): The target position. Defaults to None.
            seed (Optional[int], optional): The seed for the random number generator. Defaults to
                None.
            plane_point (Optional[np.ndarray], optional): A point on the plane. Defaults to None.
            plane_vector1 (Optional[np.ndarray], optional): A vector defining the plane. Defaults
                to None.
            plane_vector2 (Optional[np.ndarray], optional): A second vector defining the plane.
                Defaults to None.

        Raises:
            ValueError: If not all of 'plane_point', 'plane_vector1' and 'plane_vector2' are
                defined
            ValueError: If not all of 'plane_point', 'plane_vector1' and 'plane_vector2' have
                the same dimension

        Returns:
            PointTasks: A 'PointTasks' object containing the sampled tasks.
        """
        if seed is not None:
            self.set_seed(seed)

        # Do some input checking on the plane parameters
        plane_params = [plane_point, plane_vector1, plane_vector2]
        specified = [param is not None for param in plane_params]
        if not all(specified) and any(specified):
            raise ValueError("You either need to specify all plane parameters or none of them!")

        # Sample points which lie on this plane
        if target_pos is None and all(specified):
            # Convert the values to np arrays
            plane_point, plane_vector1, plane_vector2 = (
                np.array(plane_point),
                np.array(plane_vector1),
                np.array(plane_vector2),
            )

            # Check that they all have the same dimensions
            if not (
                plane_point.shape == plane_vector1.shape
                and plane_point.shape == plane_vector2.shape
            ):
                raise ValueError("All plane parameters must have the same dimensions!")

            target_pos, corners, corners_transformed = sample_plane_point(
                plane_point,
                plane_vector1,
                plane_vector2,
                self.observation_space,
                self._rng,
                num_points=num_tasks,
                max_tries=1_000_000,
                homomorphic=False,
            )

        tasks = PointTasks(
            self._observation_space,
            num_tasks=num_tasks,
            target_pos=target_pos,
        )
        return tasks


if __name__ == "__main__":
    """
    resolution = 100
    env = PointEnv3d(
        num_tasks=5,
        plane_point=[1.5, 0, -0.5],
        plane_vector1=[1, 1, -2],
        plane_vector2=[-3, 1.2, 3.3],
    )

    x = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
    states = np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    rewards_map = env._reward(states).reshape(resolution, resolution)

    # plot rollouts
    from matplotlib import pyplot as plt

    c = plt.pcolormesh(x[0], x[1], rewards_map, shading="auto", alpha=0.3)
    plt.colorbar(
        c,
    )
    plt.show()
    """
    # CONFIG
    arena_size = 2
    arena_dim = 3
    point = np.array([1.5, 1.5, 1.5])
    vec1 = np.array([1, 1, 1])
    vec2 = np.array([-0.2, 0.7, -1])
    # CONFIG END

    obs_space = spaces.Box(
        -arena_size,
        arena_size,
        shape=[
            arena_dim,
        ],
    )

    points, corners, corners_transformed = sample_plane_point(
        point,
        vec1,
        vec2,
        obs_space,
        rng=np.random.default_rng(42),
        num_points=1000,
        max_tries=1_000_000,
    )

    assert isinstance(points, np.ndarray)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    line_pairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    lines1 = [np.vstack([corners[i], corners[j]]).T for (i, j) in line_pairs]
    lines2 = [
        np.vstack([corners_transformed[i], corners_transformed[j]]).T for (i, j) in line_pairs
    ]

    for line in lines1:
        ax1.plot3D(*line)

    vec1_translated = np.vstack([point, point + vec1]).T
    vec2_translated = np.vstack([point, point + vec2]).T

    ax1.plot3D(*vec1_translated)
    ax1.plot3D(*vec2_translated)
    ax1.scatter(*points.T)

    for line in lines2:
        ax2.plot3D(line[0], line[1], line[2])

    plt.show()
