"""Matplotlib visualization helpers for COLORING samples."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_image(
    image: NDArray[np.unsignedinteger],
    ax: Axes | None = None,
    cmap: str = "tab10",
    title: str | None = None,
) -> Axes:
    """Plot a two-dimensional integer COLORING image.

    Parameters
    ----------
    image:
        Per-pixel color IDs from a :class:`dataset.ColoringDataSample`.
    ax:
        Optional Matplotlib axes. A new axes is created when omitted.
    cmap:
        Discrete-compatible Matplotlib colormap name.
    title:
        Optional axes title. No title is set by default.
    """
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional integer matrix.")
    if ax is None:
        _, ax = plt.subplots()
    n_colors = max(1, int(image.max()) + 1)
    ax.imshow(image, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_colors - 1)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax


def plot_graph(
    adjacency_matrix: NDArray[np.bool_],
    node_colors: Sequence[int] | NDArray[np.unsignedinteger],
    node_positions: NDArray[np.floating] | None = None,
    ax: Axes | None = None,
    cmap: str = "tab10",
    title: str | None = None,
) -> Axes:
    """Plot an unpadded COLORING graph.

    A Kamada--Kawai layout is computed unless ``node_positions`` is supplied.
    The generated dataset does not store Voronoi centroid coordinates, but
    :func:`utils.get_graph` can estimate positions from an image.
    """
    if (
        adjacency_matrix.ndim != 2
        or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]
    ):
        raise ValueError("adjacency_matrix must be square.")
    colors = np.asarray(node_colors)
    if colors.ndim != 1 or colors.size != adjacency_matrix.shape[0]:
        raise ValueError("node_colors must contain exactly one value per graph node.")
    if ax is None:
        _, ax = plt.subplots()

    n_nodes = adjacency_matrix.shape[0]
    if node_positions is None:
        graph = nx.from_numpy_array(adjacency_matrix)
        layout_positions = nx.kamada_kawai_layout(graph)
        positions = np.asarray([layout_positions[node] for node in range(n_nodes)])
    else:
        positions = np.asarray(node_positions, dtype=float).copy()
        positions[:, 1] = 1 - positions[:, 1]  # Flip y-axis for Matplotlib
        if positions.shape != (n_nodes, 2):
            raise ValueError("node_positions must have shape (n_nodes, 2).")
    first_nodes, second_nodes = np.where(np.triu(adjacency_matrix, k=1))
    for first, second in zip(first_nodes, second_nodes):
        ax.plot(
            positions[[first, second], 0],
            positions[[first, second], 1],
            color="0.35",
            linewidth=1.25,
            zorder=1,
        )

    n_colors = max(1, int(colors.max()) + 1)
    ax.scatter(
        positions[:, 0],
        positions[:, 1], 
        c=colors,
        cmap=cmap,
        vmin=0,
        vmax=n_colors - 1,
        edgecolors="black",
        linewidths=0.75,
        s=250,
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax
