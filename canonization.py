"""Position-free WL graph ordering adapted from GraViti.

This implementation is based on the ``ordering="wl"`` canonization code from
GraViti by Roman Bresson: https://github.com/RomanBresson/GraViti

Only the WL, degree, and BFS-distance keys are retained.  GraViti's geometric
position tie-breakers are intentionally omitted because COLORING graphs do not
store node positions.  See ``DOC/canonization.md`` for the precise algorithm
and its limitations.
"""

from __future__ import annotations

from collections import deque
from typing import Union

import numpy as np
from numpy.typing import NDArray

try:  # Support both package and direct repository-module imports.
    from .utils import ColoringGraph
except ImportError:  # pragma: no cover - exercised by notebook-style imports.
    from utils import ColoringGraph


GraphTuple = tuple[NDArray[np.generic], NDArray[np.generic]]
GraphLike = Union[GraphTuple, ColoringGraph]


def _normalize_graph(
    graph: GraphLike,
) -> tuple[NDArray[np.bool_], NDArray[np.integer]]:
    """Validate and normalize a simple undirected colored graph."""
    if isinstance(graph, ColoringGraph):
        adjacency_matrix = np.asarray(graph.adjacency_matrix)
        node_colors = np.asarray(graph.node_colors)
    else:
        try:
            adjacency_matrix, node_colors = graph
        except (TypeError, ValueError) as error:
            raise ValueError(
                "graph must be (adjacency_matrix, node_colors)."
            ) from error
        adjacency_matrix = np.asarray(adjacency_matrix)
        node_colors = np.asarray(node_colors)

    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be square.")
    n_nodes = adjacency_matrix.shape[0]
    if node_colors.ndim != 1 or node_colors.size != n_nodes:
        raise ValueError("node_colors must contain exactly one value per node.")
    if adjacency_matrix.dtype.kind not in "biuf":
        raise TypeError("adjacency_matrix must be numeric or boolean.")
    if adjacency_matrix.dtype.kind == "f" and not np.isfinite(adjacency_matrix).all():
        raise ValueError("adjacency_matrix must contain finite values.")
    if not np.all((adjacency_matrix == 0) | (adjacency_matrix == 1)):
        raise ValueError("adjacency_matrix must be binary.")

    normalized_adjacency = adjacency_matrix.astype(bool, copy=False)
    if not np.array_equal(normalized_adjacency, normalized_adjacency.T):
        raise ValueError("adjacency_matrix must be symmetric.")
    if np.any(np.diag(normalized_adjacency)):
        raise ValueError("self-loops are not supported.")
    if node_colors.dtype.kind not in "iu":
        raise TypeError("node_colors must have an integer dtype.")
    return normalized_adjacency, node_colors


def _wl_refinement_labels(
    adjacency_matrix: NDArray[np.bool_],
    node_colors: NDArray[np.integer],
    iterations: int,
) -> NDArray[np.int64]:
    """Run deterministic labeled 1-WL refinement for at most ``iterations``."""
    labels = node_colors.astype(np.int64, copy=True)
    neighbors = [
        np.flatnonzero(adjacency_matrix[node]) for node in range(labels.size)
    ]

    for _ in range(max(iterations, 1)):
        signatures = [
            (
                int(labels[node]),
                tuple(sorted(int(labels[neighbor]) for neighbor in neighbors[node])),
            )
            for node in range(labels.size)
        ]
        vocabulary = {
            signature: label
            for label, signature in enumerate(sorted(set(signatures)))
        }
        refined_labels = np.fromiter(
            (vocabulary[signature] for signature in signatures),
            dtype=np.int64,
            count=labels.size,
        )

        partition_is_stable = (
            np.unique(refined_labels).size == np.unique(labels).size
        )
        labels = refined_labels
        if partition_is_stable:
            break
    return labels


def _bfs_distances(
    adjacency_matrix: NDArray[np.bool_], source: int
) -> NDArray[np.int32]:
    """Return unweighted distances, using ``n_nodes + 1`` for unreachable nodes."""
    n_nodes = adjacency_matrix.shape[0]
    unreachable = n_nodes + 1
    distances = np.full(n_nodes, unreachable, dtype=np.int32)
    distances[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(adjacency_matrix[node]):
            if distances[neighbor] == unreachable:
                distances[neighbor] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _wl_canonical_permutation(
    adjacency_matrix: NDArray[np.bool_],
    node_colors: NDArray[np.integer],
    wl_iterations: int,
) -> NDArray[np.int32]:
    """Build GraViti's WL/BFS ordering without geometric tie-breakers."""
    n_nodes = node_colors.size
    if n_nodes <= 1:
        return np.arange(n_nodes, dtype=np.int32)

    degrees = adjacency_matrix.sum(axis=1).astype(np.int32, copy=False)
    wl_labels = _wl_refinement_labels(
        adjacency_matrix, node_colors, wl_iterations
    )

    root = min(
        range(n_nodes),
        key=lambda node: (
            int(wl_labels[node]),
            int(degrees[node]),
            node,
        ),
    )
    root_distances = _bfs_distances(adjacency_matrix, root)

    anchor = max(
        range(n_nodes),
        key=lambda node: (
            int(root_distances[node]),
            int(degrees[node]),
            -int(wl_labels[node]),
            node,
        ),
    )
    anchor_distances = _bfs_distances(adjacency_matrix, anchor)

    return np.asarray(
        sorted(
            range(n_nodes),
            key=lambda node: (
                int(wl_labels[node]),
                int(root_distances[node]),
                int(anchor_distances[node]),
                int(degrees[node]),
                node,
            ),
        ),
        dtype=np.int32,
    )


def canonize_graph(
    graph: GraphLike, wl_iterations: int = 3
) -> tuple[NDArray[np.bool_], NDArray[np.integer]]:
    """Return a position-free WL-ordered copy of ``graph``.

    Parameters
    ----------
    graph:
        ``(adjacency_matrix, node_colors)`` or a :class:`ColoringGraph`.
    wl_iterations:
        Maximum number of labeled 1-WL refinement rounds. Values below one are
        treated as one round, matching the GraViti implementation.

    Notes
    -----
    This is the GraViti WL canonical-ordering heuristic without node-position
    tie-breakers. It is deterministic for a fixed input, but unresolved graph
    symmetries fall back to original node indices; see ``DOC/canonization.md``.
    """
    if not isinstance(wl_iterations, (int, np.integer)) or isinstance(
        wl_iterations, bool
    ):
        raise TypeError("wl_iterations must be an integer.")

    adjacency_matrix, node_colors = _normalize_graph(graph)
    permutation = _wl_canonical_permutation(
        adjacency_matrix,
        node_colors,
        int(wl_iterations),
    )
    canonized_adjacency = adjacency_matrix[np.ix_(permutation, permutation)]
    canonized_colors = node_colors[permutation]
    return canonized_adjacency, canonized_colors
