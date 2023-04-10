from typing import Any, Iterable, Optional, Union, cast

import matplotlib
import matplotlib.colors as mcolors
import mpl_toolkits
import numpy as np
from matplotlib import pyplot as plt

NO_SIMULTANEOUS_MEASUREMENTS = len(list(mcolors.TABLEAU_COLORS))


def plot_pointmass_rollouts(
    replay_buffer: list[dict],
    grid: list[np.ndarray],
    reward_map_true: np.ndarray,
    reward_map_pred: np.ndarray,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plot 2d or 3d pointmass rollouts as well as the true and predicted reward landscapes.

    Args:
        replay_buffer (list[dict]): A list of dictionaries where each dictionary
            contains the keys 'observations', 'actions', 'rewards', 'next_observations',
            corresponding to a single rollout.
        grid (list[np.ndarray]): A list of 2d/3d numpy arrays containing the x and y coordinates
            of the 2d/3d grid.
        reward_map_true (np.ndarray): A 2d/3d numpy array containing the true reward landscape.
        reward_map_pred (np.ndarray): A 2d/3d numpy array containing the predicted reward
            landscape.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If too many rollouts are provided.
        ValueError: If the number of dimensions is not 2 or 3.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    dim = len(grid)
    assert dim in [2, 3], "Only 2- or 3-dimensional environments are supported!"
    # Check if only the true reward map has been provided
    only_true = reward_map_pred is None

    # Make sure that replay_buffer is a list
    if not isinstance(replay_buffer, list):
        replay_buffer = [replay_buffer]

    # Check that we don't plot too many rollouts
    if len(replay_buffer) > NO_SIMULTANEOUS_MEASUREMENTS:
        raise ValueError(
            f"Only {NO_SIMULTANEOUS_MEASUREMENTS} rollouts " "can be plotted in a single plot!"
        )

    # Get a list of available colors
    colors = list(mcolors.TABLEAU_COLORS)

    # Create the figure
    fig = plt.figure()

    # Create the axes
    if only_true:
        if dim == 3:
            ax_true = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax_true = fig.add_subplot(1, 1, 1)
        min_value, max_value = reward_map_true.min(), reward_map_true.max()

    else:
        if dim == 3:
            ax_true = fig.add_subplot(1, 2, 1, projection="3d")
            ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        else:
            ax_true = fig.add_subplot(1, 2, 1)
            ax_pred = fig.add_subplot(1, 2, 2)

        # Get the minimum and maximum values of the reward maps
        combined = np.concatenate([reward_map_true, reward_map_pred])
        min_value, max_value = combined.min(), combined.max()

    def plot_data(
        ax: Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D],
        grid: list[np.ndarray],
        reward_map: np.ndarray,
    ) -> Union[matplotlib.collections.PathCollection, matplotlib.collections.QuadMesh]:
        """A helper function which creates the plot conditioned on the dimensionality of
        the environment.

        Args:
            ax (Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D]): The axis
                on which to plot.
            grid (list[np.ndarray]): A list of 2d/3d numpy arrays containing the x, y (and z)
                coordinates of the 2d/3d grid.
            reward_map (np.ndarray): A 2d/3d numpy array containing the reward landscape.

        Raises:
            ValueError: If the number of dimensions is not 2 or 3.

        Returns:
            Union[matplotlib.collections.PathCollection, matplotlib.collections.QuadMesh]: The
                plot.
        """
        if dim == 2:
            return ax.pcolormesh(
                *grid,
                reward_map,
                alpha=0.6,
                vmin=min_value,
                vmax=max_value,
            )
        elif dim == 3:
            return ax.scatter(
                *grid,
                c=reward_map,
                alpha=0.6,
                vmin=min_value,
                vmax=max_value,
            )
        else:
            raise ValueError(f"Dimension {dim} is not supported!")

    # Plot the reward landscapes
    if dim == 2:
        mesh_true = plot_data(ax_true, grid, reward_map_true)

    if only_true:
        fig.colorbar(mesh_true, ax=ax_true)
    else:
        mesh_pred = plot_data(ax_pred, grid, reward_map_pred)
        if dim == 2:
            fig.colorbar(mesh_pred, ax=np.array([ax_true, ax_pred]).ravel().tolist())
        else:
            fig.colorbar(mesh_pred, ax=ax_pred)

    # Add the rollouts
    for idx, rollout_data in enumerate(replay_buffer):
        observations = np.array(rollout_data["prev_state"] + [rollout_data["next_state"][-1]])
        if dim == 3:
            ax_true.plot(
                observations[:, 0],
                observations[:, 1],
                observations[:, 2],
                color=colors[idx],
            )
        elif only_true:
            ax_true.plot(observations[:, 0], observations[:, 1], color=colors[idx])
        else:
            ax_pred.plot(observations[:, 0], observations[:, 1], color=colors[idx])

    # Add the titles
    fig.suptitle("Policy rollouts")
    ax_true.set_title("True reward landscape")

    if not only_true:
        ax_pred.set_title("Predicted reward landscape")

    if show:
        plt.show()
    return fig


def plot_reward_landscapes(
    grid: list[np.ndarray],
    reward_map_true: Optional[np.ndarray],
    reward_map_pred: Optional[np.ndarray],
    slices: bool = False,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plots the true and predicted reward landscapes. 2d and 3d environments are supported.
    3d environments can be plotted as a single 3d plot or as a series of 2d slices.

    Args:
        grid (list[np.ndarray]): A list of 2d/3d numpy arrays containing the x, y (and z)
            coordinates
        reward_map_true (Optional[np.ndarray]): A 2d/3d numpy array containing the true reward
            landscape.
        reward_map_pred (Optional[np.ndarray]): A 2d/3d numpy array containing the predicted
            reward landscape.
        slices (bool, optional): If True, the 3d reward landscape is plotted as a series of 2d
            slices. Defaults to False.
        show (bool, optional): If True, the plot is shown. Defaults to True.

    Raises:
        ValueError: If 'slices' is true and both reward maps are provided
        ValueError: If the number of dimensions is not 2 or 3

    Returns:
        matplotlib.figure.Figure: The plot.
    """

    # Get the dimension of the data
    dim = len(grid)

    if slices and dim == 2:
        print("For 2d environments the 'slices' parameter is ignored")
        slices = False
    is_provided = [reward_map_true is not None, reward_map_pred is not None]
    if slices and ((not any(is_provided)) or all(is_provided)):
        raise ValueError(
            "If 'slices' is true, only one of 'reward_map_true' "
            "and 'reward_map_pre' must be provided!"
        )

    # Check if only the true reward map has been provided
    only_true = reward_map_pred is None

    # Create the figure
    fig = plt.figure()

    # Create the axes
    if slices:
        # Compute the number of rows and columns
        num_plots = grid[0].shape[0]
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int((num_plots - (num_plots % num_cols)) / num_cols) + 1
        # for index in range(num_plots):
        #    axes.append(
        #        fig.add_subplot(
        #            num_rows,
        #            num_cols,
        #            index + 1,
        #        )
        #    )
        fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(8, 6), squeeze=True)
        axes = list(axes.flatten())
        for to_remove in reversed(range(num_cols, num_rows * (num_cols + 1), num_cols + 1)):
            axes[to_remove].remove()
            del axes[to_remove]
        [ax.remove() for ax in axes[num_plots:]]
        axes = axes[:num_plots]

        reward_map = reward_map_true if reward_map_true is not None else reward_map_pred
        assert reward_map is not None
        reward_map_name = "True" if reward_map_true is not None else "Predicted"
        min_value, max_value = reward_map.min(), reward_map.max()
    elif only_true:
        assert reward_map_true is not None
        if dim == 3:
            ax_true = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax_true = fig.add_subplot(1, 1, 1)
        min_value, max_value = reward_map_true.min(), reward_map_true.max()

    else:
        assert reward_map_true is not None
        assert reward_map_pred is not None
        if dim == 3:
            ax_true = fig.add_subplot(1, 2, 1, projection="3d")
            ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        else:
            ax_true = fig.add_subplot(1, 2, 1)
            ax_pred = fig.add_subplot(1, 2, 2)

        # Get the minimum and maximum values of the reward maps
        combined: np.ndarray = np.concatenate([reward_map_true, reward_map_pred])
        min_value, max_value = combined.min(), combined.max()

    def plot_data(
        ax: Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D],
        grid: list[np.ndarray],
        reward_map: np.ndarray,
    ) -> Union[matplotlib.collections.PathCollection, matplotlib.collections.QuadMesh]:
        """A helper function which creates the plot conditioned on the dimensionality of
        the environment.

        Args:
            ax (Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D]): The axis
                on which to plot.
            grid (list[np.ndarray]): A list of 2d/3d numpy arrays containing the x, y (and z)
                coordinates
            reward_map (np.ndarray): A 2d/3d numpy array containing the reward landscape.

        Raises:
            ValueError: If the number of dimensions is not 2 or 3

        Returns:
            Union[matplotlib.collections.PathCollection, matplotlib.collections.QuadMesh]: The
                plot.
        """
        if dim == 2:
            return ax.pcolormesh(
                *grid,
                reward_map,
                alpha=0.6,
                vmin=min_value,
                vmax=max_value,
            )
        elif dim == 3:
            return ax.scatter(
                *grid,
                c=reward_map,
                alpha=0.6,
                vmin=min_value,
                vmax=max_value,
                linewidths=3,
            )
        else:
            raise ValueError(f"Dimension {dim} is not supported!")

    if slices:
        assert reward_map is not None
        for index, ax in enumerate(axes):
            height = grid[1][index][0][0]
            mesh = ax.pcolormesh(
                grid[0][index],
                grid[2][index],
                reward_map[:, index, :],
                alpha=0.6,
                vmin=min_value,
                vmax=max_value,
            )
            ax.set_title(f"Height = {height:.2f}")
        cb_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        fig.colorbar(mesh, cax=cb_ax)

        fig.suptitle(reward_map_name + " reward landscapes")
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        fig.tight_layout()

    else:
        # Plot the reward landscapes
        assert reward_map_true is not None
        mesh_true = plot_data(ax_true, grid, reward_map_true)

        if only_true:
            fig.colorbar(mesh_true, ax=ax_true)
        else:
            assert reward_map_pred is not None
            mesh_pred = plot_data(ax_pred, grid, reward_map_pred)
            fig.colorbar(mesh_pred, ax=np.array([ax_true, ax_pred]).ravel().tolist())

        # Add the titles
        fig.suptitle("Reward landscapes")
        ax_true.set_title("True reward landscape")

        if not only_true:
            ax_pred.set_title("Predicted reward landscape")

    if show:
        plt.show()

    return fig


def plot_buffer_uncertainty(
    grid: list[np.ndarray],
    reward_map_true: np.ndarray,
    reward_map_pred: np.ndarray,
    context_x: np.ndarray,
    uncertainties: np.ndarray,
    observed_samples: Optional[np.ndarray] = None,
    scale: int = 15,
    cmap: str = "Reds",
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Given a set of points in the context space, plot the uncertainty of the reward
    model at those points as well as the predicted and true reward landscacpes.

    Args:
        grid (list[np.ndarray]): A list of 2d/3d numpy arrays containing the x, y (and z)
        reward_map_true (np.ndarray): A 2d/3d numpy array containing the true reward landscape.
        reward_map_pred (np.ndarray): A 2d/3d numpy array containing the predicted reward
            landscape.
        context_x (np.ndarray): A 2d/3d numpy array containing the points in the context space.
        uncertainties (np.ndarray): A 1d numpy array containing the uncertainty of the reward model
            at the points in the context space.
        observed_samples (Optional[np.ndarray], optional): A 2d numpy array containing the points
            in the context space which have been observed. Defaults to None.
        scale (int, optional): The scale of the uncertainty points. Defaults to 15.
        cmap (str, optional): The colormap to use. Defaults to "Reds".
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure.
    """

    # Create the figure
    fig = plt.figure()

    # Create the axes
    ax_true = fig.add_subplot(1, 2, 1)
    ax_pred = fig.add_subplot(1, 2, 2)

    # Get the minimum and maximum values of the reward maps
    combined = np.concatenate([reward_map_true, reward_map_pred])
    min_value, max_value = combined.min(), combined.max()

    # Plot the reward landscapes
    _ = ax_true.pcolormesh(
        grid[0],
        grid[1],
        reward_map_true,
        shading="auto",
        alpha=0.6,
        vmin=min_value,
        vmax=max_value,
    )
    mesh_pred = ax_pred.pcolormesh(
        grid[0],
        grid[1],
        reward_map_pred,
        shading="auto",
        alpha=0.6,
        vmin=min_value,
        vmax=max_value,
    )
    fig.colorbar(mesh_pred, ax=np.array([ax_true, ax_pred]).ravel().tolist())

    # Predict the uncertainty of the context of the reward model
    # Transpose the context
    x, y = context_x.T[0], context_x.T[1]
    ax_pred.scatter(
        x,
        y,
        s=scale * uncertainties,
        c=uncertainties,
        alpha=0.5,
        cmap=plt.get_cmap(cmap),
    )

    # Add the observed samples if they exist
    if observed_samples is not None:
        x2, y2 = observed_samples.T[0], observed_samples.T[1]
        ax_pred.scatter(x2, y2, s=20, c="black")

    # Add the titles
    fig.suptitle("Reward model context and uncertainty")
    ax_true.set_title("True reward landscape")
    ax_pred.set_title("Predicted reward landscape")

    if show:
        plt.show()
    return fig


def plot_agent_policy(
    grid_reward_landscape: list[np.ndarray],
    grid_policy: list[np.ndarray],
    reward_map_true: np.ndarray,
    reward_map_pred: np.ndarray,
    policy_map: np.ndarray,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plots the entire policy of an RL agent trained on a 2d reward landscape as a
    vector field.

    Args:
        grid_reward_landscape (list[np.ndarray]): A list of 2d numpy arrays containing the x, and
            y coordinates of the reward landscape.
        grid_policy (list[np.ndarray]): A list of 2d numpy arrays containing the x, and y
            coordinates where the policy is evaluated.
        reward_map_true (np.ndarray): A 2d numpy array containing the true reward landscape.
        reward_map_pred (np.ndarray): A 2d numpy array containing the predicted reward landscape.
        policy_map (np.ndarray): A 2d numpy array containing the policy of the agent evaluated
            at the points in the policy grid.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    # Check if only the true reward map has been provided
    only_true = reward_map_pred is None

    # Create the figure
    fig = plt.figure()

    # Create the axes
    if only_true:
        ax_true = fig.add_subplot(1, 1, 1)
        min_value, max_value = reward_map_true.min(), reward_map_true.max()
    else:
        ax_true = fig.add_subplot(1, 2, 1)
        ax_pred = fig.add_subplot(1, 2, 2)

        # Get the minimum and maximum values of the reward maps
        combined = np.concatenate([reward_map_true, reward_map_pred])
        min_value, max_value = combined.min(), combined.max()

    # Plot the reward landscapes
    mesh_true = ax_true.pcolormesh(
        grid_reward_landscape[0],
        grid_reward_landscape[1],
        reward_map_true,
        shading="auto",
        alpha=0.6,
        vmin=min_value,
        vmax=max_value,
    )
    if only_true:
        fig.colorbar(mesh_true, ax=ax_true)
    else:
        mesh_pred = ax_pred.pcolormesh(
            grid_reward_landscape[0],
            grid_reward_landscape[1],
            reward_map_pred,
            shading="auto",
            alpha=0.6,
            vmin=min_value,
            vmax=max_value,
        )
        fig.colorbar(mesh_pred, ax=np.array([ax_true, ax_pred]).ravel().tolist())

    # Plot the policy map
    if only_true:
        ax_true.quiver(
            grid_policy[0],
            grid_policy[1],
            policy_map.T[0],
            policy_map.T[1],
        )
    else:
        ax_pred.quiver(
            grid_policy[0],
            grid_policy[1],
            policy_map.T[0],
            policy_map.T[1],
        )

    # Add the titles
    fig.suptitle("Policy vector field")
    ax_true.set_title("True reward landscape")

    if not only_true:
        ax_pred.set_title("Policy vector field")

    if show:
        plt.show()
    return fig


def get_fig_axes(
    num_plots: int, plot_arrangement_shape: Union[str, tuple[int, int]] = "auto"
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Given the number of plots and an optional subplot-arrangement shape, create a figure
    with corresponding axes in the desired arrangement shape.

    Args:
        num_plots (int): The total number of plots
        plot_arrangement_shape (Union[str, tuple[int, int]], optional): Either "auto" or a 2d
            tuple specifying a grid structure. Defaults to "auto".

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: (fig, axes) where:
            fig: is a matplotlib figure object and
            axes: is a list of matplotlib axes objects (len(axes) = num_plots)
    """
    # Automatically determine the appropriate shape how plots should be arranged
    if plot_arrangement_shape == "auto":
        # find smallest square number larger than 'num_plots'
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int(np.ceil(num_plots / num_cols))
    else:
        assert isinstance(plot_arrangement_shape, tuple) and len(plot_arrangement_shape) == 2
        num_cols, num_rows = plot_arrangement_shape[0], plot_arrangement_shape[1]

    # Create the figure
    fig = plt.figure()

    # Create the axes
    axes = []
    for index in range(num_plots):
        axes.append(fig.add_subplot(num_rows, num_cols, index + 1))

    return fig, axes


def check_variable_length(
    same_length: list[list[tuple[str, Any]]], fix_lengths: Optional[list[int]] = None
) -> list[list[Any]]:
    """Given a set of arrays, determine if they all have the same length. If 'fix_lengths'
    is not provided then raise an exception if two arrays don't have same length. If
    'fix_lengths' is set, pad/cut them to a fixed size.

    Args:
        same_length (list[list[tuple[str, Any]]]): A list of different equivalence classes where
            each equivalence class is a list of arrays which should have same length. To make error
            messages easier, instead of just providing a list of arrays, a user should
            provide a list of (name, array) tuples where 'name' is the name of the
            'array' variable. An example instantiation of same_length could look like
            this:
                same_length = [
                    [
                        ("int_array", [1, 2, 3, 4]),
                        ("string_array", ["This", "is", "an", "example"])
                    ],
                    [
                        ("float_array1", [4.2, 2.1]),
                        ("string_array2", ["Hello", "world"]),
                        ("last_one", [-4, -9])
                    ]
                ]
        fix_lengths (Optional[list[int]], optional): Either 'None' or an integer list of same
            length like the 'same_length' variable containing the length to which each individual
            array in 'same_length' should be padded/cut.

    Raises:
        ValueError: if two arrays don't have the same size and 'fix_lengths' is not
        provided by the user.

    Returns:
        list[list[Any]]: The raw arrays of the 'same_length' input list. Raw in this context means
        that while the 'same_input' array contained lists of (name, array) tuples, the output
        is a list of arrays.
    """
    # The array to store the raw output arrays
    results = []

    # Iterate over every equivalence class (i.e. over every list which
    # should contain arrays of same size)
    for index, equivalence_class in enumerate(same_length):
        eq_result = []

        # Extract the names and lengths of every (name, array) tuple in the list
        names = [
            e[0] for e in equivalence_class if hasattr(e[1], "__iter__") and type(e[1]) is not str
        ]
        lengths = [
            len(e[1])
            for e in equivalence_class
            if hasattr(e[1], "__iter__") and type(e[1]) is not str
        ]

        # If fix_lengths was provided, check for each array if it has the length
        # specified in this array
        if fix_lengths is not None:
            length = fix_lengths[index]

        # otherwise just check if all arrays in the list have same size
        else:
            length = lengths[0]  # length of the first array

        # Iterate over all (name, array) tuples in the equivalence_class
        for index2, [name, array] in enumerate(equivalence_class):

            # Check if:
            #   1) the array value has even be specified (i.e. is actually an iterable
            #      and not just 'None' or some other value)
            #   2) if it has been specified if the array has the correct length
            if (
                array is not None
                and hasattr(array, "__iter__")
                and type(array) is not str
                and len(array) != length
            ):

                if fix_lengths is not None:
                    # If 'fix_lengths' has been specified, pad/cut the array to
                    # the correct length
                    fixed: Union[list[Any], np.ndarray]
                    if len(array) > length:
                        fixed = array[:length]
                    else:
                        fixed = list(array) + [array[-1] for _ in range(length - len(array))]
                        if type(array) == np.ndarray:
                            fixed = np.array(fixed)

                    # Append the fixed array to the output list
                    eq_result.append(fixed)
                else:
                    # If 'fixed_lengths' has not been specified, just raise an exception
                    raise ValueError(
                        f"All the following variables need to have the same lenght! "
                        f"{names} But got the following lengths instead: {lengths}"
                    )
            else:
                # If everything is ok, just append the raw array to the output array
                eq_result.append(array)

        # Append the raw output equivalence class list to the result list
        results.append(eq_result)

    return results


def wrap_variables(wrap_variables: list[Any]) -> list[list[Any]]:
    """Wrap every variable inside 'wrap_variables' in a list if the
    variable isn't already a list

    Args:
        wrap_variables (list[Any]): A list of variables to wrap in a list

    Returns:
        list[list[Any]]: wrap_variables but every entry is now guaranteed to have
        the '__iter__' attribute (i.e. is iterable)
    """
    # Iterate over all variables in the 'wrap_variables' array
    for index in range(len(wrap_variables)):
        # Check if the variable is already iterable (and not a string)
        if wrap_variables[index] is not None and (
            not hasattr(wrap_variables[index][0], "__iter__") or type(wrap_variables[index]) == str
        ):
            # Wrap the variable
            wrap_variables[index] = [wrap_variables[index]]

    return wrap_variables


def line_plots(
    num_plots: int,
    x_arrays: Union[list[float], list[list[float]], np.ndarray],
    y_arrays: Union[list[float], list[list[float]], list[list[np.ndarray]]],
    y_error_bars: Optional[Union[list[float], list[list[float]], list[list[np.ndarray]]]] = None,
    y_error_bars_layout: str = "continuous",
    scatter: Optional[list[bool]] = None,
    scatter_configs: Optional[list[dict[str, Union[str, float, int]]]] = None,
    sup_title: Optional[str] = None,
    titles: Optional[list[str]] = None,
    x_axis_names: Optional[Union[str, list[str]]] = None,
    y_axis_names: Optional[Union[str, list[str]]] = None,
    labels: Optional[list[Union[str, list[str]]]] = None,
    x_limits: Optional[tuple[Optional[float], Optional[float]]] = None,
    y_limits: Optional[tuple[Optional[float], Optional[float]]] = None,
    plot_arrangement_shape: Union[str, tuple[int, int]] = "auto",
    save_path: Optional[str] = None,
    plot_legends: bool = False,
    save: bool = False,
    plot: bool = True,
) -> matplotlib.figure.Figure:
    """A function which makes it possible to create a decent plot wihtout having to write
    100 lines of code every time. The parameters specify plot parameters.

    Args:
        num_plots (int): The number of plots to create
        x_arrays (Union[list[float], list[list[float]], np.ndarray]): Either just a single array
            which is used as x-values for every sub-plot, or a list of lists, containing one
            list of x-values per sub-plot.
        y_arrays (Union[list[float], list[list[float]], list[list[np.ndarray]]]): A list. This
            list contains one list per subplot. Each list contains
            one list per data series which shall be plotted on this subplot.
                For example, if you have 3 subplots and you want to plot 1 data series
                in the first and 2 data series in the second and third subplot, it could
                look something like this:
                    y_array = [[[1, 2, 3, 4]], [[5, 3], [1, 2]], [[8, 9, 0], [1, 2, 3]]]
        y_error_bars (
            Optional[Union[list[float], list[list[float]], list[list[np.ndarray]]]], optional
        ):  A list (of lists) of strings. Must have the same shape as 'y_arrays'.
            This variable contains optional error bars for the plot points. Defaults to None.
        y_error_bars_layout (str, optional): One of ["continuous", "pointwise"]. This option
            controls how the error bars shall be displayed
        scatter (Optional[list[bool]], optional): A list of booleans where index 'i' denotes
            whether plot 'i' will be a scatter plot. Defaults to None.
        scatter_configs (Optional[list[dict[str, Union[str, float, int]]]], optional): If any
            entry of the 'scatter' parameter above is True, needs to contain a list of
            dictionaries (or None entries) containing the parameters of the scatter plot
            (including x- and y-values) for each plot. Defaults to None.
        sup_title (Optional[str], optional): The main title of the plot. Defaults to None.
        titles (Optional[list[str]], optional): A list of strings of length 'num_plots' containing
            one title per subplot. Defaults to None.
        x_axis_names (Optional[Union[str, list[str]]], optional): A list of strings with the same
            length as 'x_arrays' containing x-axis names for each x-axis defined in 'x_arrays'.
            Defaults to None.
        y_axis_names (Optional[Union[str, list[str]]], optional): A list of strings of length
            'num_plots' containing names for the y-axis of every subplot. Defaults to None.
        labels (Optional[list[Union[str, list[str]]]], optional): A list (of lists) of strings.
            Must have the same shape as 'y_arrays'. This variable contains a label for every
            single data-series stored in 'y_arrays'. This is used for subplot legends.
        x_limits (Optional[tuple[Optional[float], Optional[float]]], optional): A list with same
            length as 'x_arrays' containing tuples (l,u) indicating the lower or upper bounds
            of the x-axes. 'l' and 'b' can either be integers or 'None'. Defaults to None.
        y_limits (Optional[tuple[Optional[float], Optional[float]]], optional): A list with same
            length as 'y_arrays' containing tuples (l,u) indicating the lower or upper bounds of
            the y-axes. 'l' and 'b' can either be integers or 'None'. Defaults to None.
        plot_arrangement_shape (Union[str, tuple[int, int]], optional): Either "auto" or a 1d/2d
            tuple specifying the grid strucutre in which the different subplots shall be arranged.
            Defaults to "auto".
        save_path (Optional[str], optional): Contains the path into which the plot shall be saved.
            Defaults to None.
        plot_legends (bool, optional): Either a boolean indicating if all sub-plots shall have
            legends or a list of booleans of length 'num_plots' containing for each individual
            subplot a boolean indicating if its legend shall be printed. Defaults to False.
        save (bool, optional): A boolean indicating if the plot shall be saved. If 'True' then the
            'save_path' must be provided. Defaults to False.
        plot (bool, optional): A boolean indicating if the created figure shall be plotted. Defaults
            to True.

    Raises:
        TypeError: If 'x_arrays' or 'y_arrays' are not iterable
        ValueError: If 'y_error_bars' is not in ['continuous', 'pointwise']
        ValueError: If 'y_arrays' contain too many data series
        ValueError: If 'save' is True but 'save_path' is None

    Returns:
        matplotlib.figure.Figure: The figure which is created by this function
    """
    ############################################
    # Data sanity checks
    ############################################

    # Check if x_array and y_arrays are both iterable (and not just strings)
    if (
        not hasattr(x_arrays, "__iter__")
        or type(x_arrays) == str
        or not hasattr(y_arrays, "__iter__")
        or type(y_arrays) == str
    ):
        raise TypeError(
            f"x_arrays and y_arrays must be iterable! "
            f"Got {type(x_arrays)} and {type(y_arrays)}"
        )

    # Check that 'y_error_bars_layout' has the correct value
    error_bars_layouts = ["continuous", "pointwise"]
    if y_error_bars_layout not in error_bars_layouts:
        raise ValueError(f"'y_error_bars_layout' must be in {error_bars_layouts}")

    # Check for all the following variables if they have the same length
    # (if they have been specified)
    same_length = [
        [
            ("x_arrays", x_arrays),
            ("x_axis_names", x_axis_names),
            ("x_limits", x_limits),
        ],
        [
            ("y_arrays", y_arrays),
            ("y_error_bars", y_error_bars),
            ("y_axis_names", y_axis_names),
            ("labels", labels),
            ("y_limits", y_limits),
        ],
        [
            ("plot_legends", plot_legends),
            ("num_plots_list", [0 for _ in range(num_plots)]),
            ("titles", titles),
        ],
    ]
    assert all([isinstance(same_length[i], list) for i in range(len(same_length))])
    same_length2 = cast(list[list[tuple[str, Any]]], same_length)
    check_variable_length(same_length2)

    # Wrap all the following variables if they arent' already wrapped
    variables_to_wrap = [
        x_arrays,
        x_axis_names,
        x_limits,
        y_arrays,
        y_error_bars,
        y_axis_names,
        labels,
    ]

    (
        x_arrays,  # type: ignore
        x_axis_names,  # type: ignore
        x_limits,  # type: ignore
        y_arrays,  # type: ignore
        y_error_bars,  # type: ignore
        y_axis_names,  # type: ignore
        labels,  # type: ignore
    ) = wrap_variables(  # type: ignore
        wrap_variables=variables_to_wrap  # type: ignore
    )  # type: ignore

    # Make sure that x_arrays, x_axis_names and x_limits have a length of num_plots
    [[x_arrays, x_axis_names, x_limits]] = check_variable_length(
        [
            [
                ("x_arrays", x_arrays),
                ("x_axis_names", x_axis_names),
                ("x_limits", x_limits),
            ]
        ],
        fix_lengths=[num_plots],
    )

    # Make sure that y_arrays, y_axis_names and y_limits have a length of num_plots
    [[y_arrays, y_axis_names, y_limits]] = check_variable_length(
        [
            [
                ("y_arrays", y_arrays),
                ("y_axis_names", y_axis_names),
                ("y_limits", y_limits),
            ]
        ],
        fix_lengths=[num_plots],
    )

    # Check if we are trying to plot too many data series in one plot
    for data_series in y_arrays:
        assert not isinstance(data_series, float)
        if len(data_series) > NO_SIMULTANEOUS_MEASUREMENTS:
            raise ValueError(
                f"Only {NO_SIMULTANEOUS_MEASUREMENTS} data series "
                "can be plotted in a single plot!"
            )

    # Raise an error if the plot shall be saved and no save-path has been provided
    if save and save_path is None:
        raise ValueError("You need to specify a save path if you want to save your plots.")

    ############################################
    # Creating the plot
    ############################################

    # Get a list of available colors
    colors = list(mcolors.TABLEAU_COLORS)

    font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}

    matplotlib.rc("font", **font)

    # Create the figure and its corresponding axes
    fig, axes = get_fig_axes(num_plots, plot_arrangement_shape)

    # Plot the main title of the figure (if one has been provided)
    if sup_title is not None:
        fig.suptitle(sup_title)

    # Iterate over all sub-plot axes
    for index, axis in enumerate(axes):
        # Set the limits of this axis
        if x_limits is not None and x_limits[index] is not None:
            assert isinstance(x_limits, list)
            bottom, top = x_limits[index]
            axis.set_xlim(left=bottom, right=top)

        if y_limits is not None and y_limits[index] is not None:
            assert isinstance(y_limits, list)
            bottom, top = y_limits[index]
            axis.set_ylim(bottom=bottom, top=top)

        # If this plot should be a scatter plot, plot the scatter plot
        if scatter is not None and scatter[index] and scatter_configs is not None:
            axis.scatter(**scatter_configs[index])
        # Create normal line plot
        else:
            # For each sub-plot, plot all specified data series
            y_arrays = cast(list[list[float]], y_arrays)
            for index2, y_array in enumerate(y_arrays[index]):
                if y_error_bars is None:
                    axis.plot(
                        x_arrays[index],
                        y_array,
                        colors[index2],
                        label=labels[index][index2] if labels is not None else labels,
                    )
                else:
                    y_error_bars = cast(list[list[float]], y_error_bars)
                    if y_error_bars_layout == "pointwise":
                        axis.errorbar(
                            x_arrays[index],
                            y_array,
                            yerr=y_error_bars[index][index2],
                            color=colors[index2],
                            label=labels[index][index2] if labels is not None else labels,
                        )
                    elif y_error_bars_layout == "continuous":
                        axis.plot(
                            x_arrays[index],
                            y_array,
                            colors[index2],
                            label=labels[index][index2] if labels is not None else labels,
                        )

                        axis.fill_between(
                            x_arrays[index],
                            (y_array - y_error_bars[index][index2]),
                            (y_array + y_error_bars[index][index2]),
                            color=colors[index2],
                            alpha=0.3,
                        )

        # If desired, plot a legend on this subplot
        if (isinstance(plot_legends, bool) and plot_legends) or (
            isinstance(plot_legends, list) and plot_legends[index]
        ):
            axis.legend()

        # Create a grid
        axis.grid(which="both")

        # If a title for this specific sub-plot has been provided set it.
        if titles is not None:
            axis.set_title(titles[index])

        # If axis names have been provided, plot them
        if x_axis_names is not None:
            axis.set_xlabel(x_axis_names[index])

        if y_axis_names is not None:
            axis.set_ylabel(y_axis_names[index])

    # plt.tight_layout()

    # Save the plot if desired
    if save:
        plt.savefig(
            save_path,
            dpi=fig.dpi,
            bbox_inches="tight",
        )

    # Plot the figure if desired
    if plot:
        plt.show()

    return fig


if __name__ == "__main__":
    """
    config = {
        "num_plots": 2,
        "x_arrays": [1, 2, 3],
        "y_arrays": [[[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [4, 5, 6]]],
        "titles": ["test plot 1", "test plot 2"],  # list of strings
        "x_axis_names": "x values",  # list of strings
        "y_axis_names": "y values",  # list of strings
        "labels": [
            ["first data series", "second data series"],
            ["first data series", "second data series"],
        ],  # list (of lists) of strings
        "plot_arrangement_shape": "auto",  # 'auto' or a 1d/2d tuple of integers
        "save_path": None,  # string
        "plot_legends": [True, True],  # boolean or list of booleans
        "save": False,  # boolean
        "plot": True,  # boolean
    }

    line_plots(**config)
    """
