"""Utilities for reconstructing COLORING graphs from integer images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True)
class ColoringGraph:
    """A COLORING graph represented by node colors and an adjacency matrix."""

    node_colors: NDArray[np.integer]
    adjacency_matrix: NDArray[np.bool_]


def _component_map(
    image: NDArray[np.integer],
) -> tuple[NDArray[np.int64], NDArray[np.integer], NDArray[np.float64]]:
    """Identify same-color 4-connected regions and their normalized centers."""
    height, width = image.shape
    component_ids = np.full(image.shape, -1, dtype=np.int64)
    node_colors: list[int] = []
    node_positions: list[tuple[float, float]] = []
    cross_structure = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=bool)

    component_index = 0
    for color in np.unique(image):
        labels, n_components = ndimage.label(image == color, structure=cross_structure)
        for component_label in range(1, n_components + 1):
            component = labels == component_label
            component_ids[component] = component_index
            rows, columns = np.nonzero(component)
            # Pixel-center coordinates normalized to the unit square, ordered x then y.
            node_positions.append(
                ((columns.mean() + 0.5) / width, (rows.mean() + 0.5) / height)
            )
            node_colors.append(int(color))
            component_index += 1

    return (
        component_ids,
        np.asarray(node_colors, dtype=image.dtype),
        np.asarray(node_positions, dtype=np.float64),
    )


def _adjacency_from_component_map(
    component_ids: NDArray[np.int64], n_nodes: int
) -> NDArray[np.bool_]:
    """Build an undirected adjacency matrix from component-boundary pixels."""
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    for first, second in (
        (component_ids[:, :-1], component_ids[:, 1:]),
        (component_ids[:-1, :], component_ids[1:, :]),
    ):
        different = first != second
        adjacency_matrix[first[different], second[different]] = True
        adjacency_matrix[second[different], first[different]] = True
    return adjacency_matrix


def get_graph(
    image: NDArray[np.integer], return_node_pos: bool = True
) -> (
    tuple[ColoringGraph, NDArray[np.bool_], NDArray[np.float64]]
    | tuple[ColoringGraph, NDArray[np.bool_]]
):
    """Reconstruct the graph encoded by a COLORING integer image.

    Each node is one 4-connected component of equal-valued pixels. This is
    necessary because distinct, non-adjacent Voronoi regions may share a color.
    An edge is added when two components share a horizontal or vertical pixel
    boundary. Therefore the recovered graph is the stored graph up to node
    permutation. Node positions are estimated by the normalized `(x, y)` mean
    of each component's pixel centers; they need not equal original centroids.

    Parameters
    ----------
    image:
        Two-dimensional integer color-ID matrix.
    return_node_pos:
        When true, return ``(graph, adjacency_matrix, node_positions)``.
        Otherwise return ``(graph, adjacency_matrix)``.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional color-ID matrix.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if image.dtype.kind not in "iu":
        raise TypeError("image must have an integer dtype.")

    component_ids, node_colors, node_positions = _component_map(image)
    adjacency_matrix = _adjacency_from_component_map(component_ids, len(node_colors))
    graph = ColoringGraph(
        node_colors=node_colors,
        adjacency_matrix=adjacency_matrix,
    )
    if return_node_pos:
        return graph, adjacency_matrix, node_positions
    return graph, adjacency_matrix
