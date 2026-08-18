"""Validity and isomorphism metrics for node-colored COLORING graphs."""

from __future__ import annotations

from collections import Counter
from typing import Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

try:  # Support both package imports and direct repository-module imports.
    from .utils import ColoringGraph
except ImportError:  # pragma: no cover - exercised by notebook-style imports.
    from utils import ColoringGraph


GraphTuple = tuple[NDArray[np.generic], NDArray[np.generic]]
GraphLike = Union[GraphTuple, ColoringGraph]


class _GraphFormatError(ValueError):
    """Internal error raised for malformed graph representations."""


def _normalize_graph(
    graph: GraphLike,
) -> tuple[NDArray[np.bool_], NDArray[np.integer]]:
    """Validate and normalize a simple undirected node-colored graph."""
    if isinstance(graph, ColoringGraph):
        adjacency_matrix = np.asarray(graph.adjacency_matrix)
        node_colors = np.asarray(graph.node_colors)
    else:
        try:
            adjacency_matrix, node_colors = graph
        except (TypeError, ValueError) as error:
            raise _GraphFormatError(
                "graph must be (adjacency_matrix, node_colors)."
            ) from error
        adjacency_matrix = np.asarray(adjacency_matrix)
        node_colors = np.asarray(node_colors)

    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise _GraphFormatError("adjacency_matrix must be square.")
    n_nodes = adjacency_matrix.shape[0]
    if node_colors.ndim != 1 or node_colors.size != n_nodes:
        raise _GraphFormatError("node_colors must contain one value per node.")
    if adjacency_matrix.dtype.kind not in "biuf":
        raise _GraphFormatError("adjacency_matrix must be numeric or boolean.")
    if adjacency_matrix.dtype.kind == "f" and not np.isfinite(adjacency_matrix).all():
        raise _GraphFormatError("adjacency_matrix must contain finite values.")
    if not np.all((adjacency_matrix == 0) | (adjacency_matrix == 1)):
        raise _GraphFormatError("adjacency_matrix must be binary.")

    normalized_adjacency = adjacency_matrix.astype(bool, copy=False)
    if not np.array_equal(normalized_adjacency, normalized_adjacency.T):
        raise _GraphFormatError("adjacency_matrix must be symmetric.")
    if np.any(np.diag(normalized_adjacency)):
        raise _GraphFormatError("self-loops are not supported.")
    if node_colors.dtype.kind not in "iu":
        raise _GraphFormatError("node_colors must have an integer dtype.")
    return normalized_adjacency, node_colors


def _edge_count(adjacency_matrix: NDArray[np.bool_]) -> int:
    """Count undirected edges in a normalized adjacency matrix."""
    return int(np.count_nonzero(np.triu(adjacency_matrix, k=1)))


def _to_networkx(
    adjacency_matrix: NDArray[np.bool_], node_colors: NDArray[np.integer]
) -> nx.Graph:
    """Convert the NumPy graph representation to a labeled NetworkX graph."""
    graph = nx.from_numpy_array(adjacency_matrix)
    nx.set_node_attributes(
        graph,
        {node: int(color) for node, color in enumerate(node_colors)},
        "color",
    )
    return graph


def is_planar(adjacency_matrix: NDArray[np.generic]) -> bool:
    """Check planarity after applying the cheap planar edge-count bound.

    Malformed, directed, non-binary, or self-looped matrices return ``False``.
    """
    try:
        adjacency, _ = _normalize_graph(
            (adjacency_matrix, np.zeros(np.asarray(adjacency_matrix).shape[0], dtype=np.uint8))
        )
    except (IndexError, _GraphFormatError):
        return False

    n_nodes = adjacency.shape[0]
    n_edges = _edge_count(adjacency)
    if n_nodes >= 3 and n_edges > 3 * n_nodes - 6:
        return False
    networkx_graph = nx.from_numpy_array(adjacency)
    planar, _ = nx.check_planarity(networkx_graph, counterexample=False)
    return bool(planar)


def _wl_refine(
    adjacency_a: NDArray[np.bool_],
    colors_a: NDArray[np.integer],
    adjacency_b: NDArray[np.bool_],
    colors_b: NDArray[np.integer],
) -> bool:
    """Return whether stable labeled 1-WL fails to distinguish two graphs."""
    state_a = colors_a.astype(np.int64, copy=True)
    state_b = colors_b.astype(np.int64, copy=True)
    if sorted(state_a.tolist()) != sorted(state_b.tolist()):
        return False

    max_rounds = max(adjacency_a.shape[0], adjacency_b.shape[0])
    for _ in range(max_rounds):
        tokens_a = [
            (int(state_a[node]), tuple(sorted(state_a[adjacency_a[node]].tolist())))
            for node in range(state_a.size)
        ]
        tokens_b = [
            (int(state_b[node]), tuple(sorted(state_b[adjacency_b[node]].tolist())))
            for node in range(state_b.size)
        ]
        canonical_tokens = {
            token: token_id
            for token_id, token in enumerate(sorted(set(tokens_a + tokens_b)))
        }
        next_a = np.fromiter(
            (canonical_tokens[token] for token in tokens_a), dtype=np.int64
        )
        next_b = np.fromiter(
            (canonical_tokens[token] for token in tokens_b), dtype=np.int64
        )
        if sorted(next_a.tolist()) != sorted(next_b.tolist()):
            return False

        stable_a = np.unique(next_a).size == np.unique(state_a).size
        stable_b = np.unique(next_b).size == np.unique(state_b).size
        if stable_a and stable_b:
            return True
        state_a, state_b = next_a, next_b
    return True


def wl_may_be_isomorphic(graph_a: GraphLike, graph_b: GraphLike) -> bool:
    """Return whether labeled stable 1-WL does not reject a graph pair."""
    try:
        adjacency_a, colors_a = _normalize_graph(graph_a)
        adjacency_b, colors_b = _normalize_graph(graph_b)
    except _GraphFormatError:
        return False
    if adjacency_a.shape != adjacency_b.shape:
        return False
    return _wl_refine(adjacency_a, colors_a, adjacency_b, colors_b)


def are_planar_graphs_isomorphic(graph_a: GraphLike, graph_b: GraphLike) -> bool:
    """Run exact color-preserving VF2 on two graphs assumed to be planar."""
    try:
        adjacency_a, colors_a = _normalize_graph(graph_a)
        adjacency_b, colors_b = _normalize_graph(graph_b)
    except _GraphFormatError:
        return False
    if adjacency_a.shape != adjacency_b.shape:
        return False

    networkx_a = _to_networkx(adjacency_a, colors_a)
    networkx_b = _to_networkx(adjacency_b, colors_b)
    node_match = nx.algorithms.isomorphism.categorical_node_match("color", None)
    return bool(nx.is_isomorphic(networkx_a, networkx_b, node_match=node_match))


def _component_signatures(
    adjacency_matrix: NDArray[np.bool_], node_colors: NDArray[np.integer]
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    """Return cheap isomorphism-invariant signatures of connected components."""
    graph = nx.from_numpy_array(adjacency_matrix)
    signatures = []
    for component in nx.connected_components(graph):
        nodes = np.fromiter(component, dtype=np.int64)
        subgraph = adjacency_matrix[np.ix_(nodes, nodes)]
        signatures.append(
            (
                nodes.size,
                _edge_count(subgraph),
                tuple(sorted(subgraph.sum(axis=1).astype(int).tolist())),
                tuple(sorted(node_colors[nodes].astype(int).tolist())),
            )
        )
    return sorted(signatures)


def is_valid(graph: GraphLike) -> int:
    """Return 1 iff ``graph`` is planar and has a proper node coloring."""
    try:
        adjacency_matrix, node_colors = _normalize_graph(graph)
    except _GraphFormatError:
        return 0

    first_nodes, second_nodes = np.where(np.triu(adjacency_matrix, k=1))
    if np.any(node_colors[first_nodes] == node_colors[second_nodes]):
        return 0
    return int(is_planar(adjacency_matrix))


def is_same(predicted_graph: GraphLike, target_graph: GraphLike) -> int:
    """Return 1 iff two node-colored graphs are isomorphic.

    ``target_graph`` is assumed planar. The predicted graph is explicitly
    checked for planarity after cheap invariants and stable 1-WL filtering.
    """
    try:
        predicted_adjacency, predicted_colors = _normalize_graph(predicted_graph)
        target_adjacency, target_colors = _normalize_graph(target_graph)
    except _GraphFormatError:
        return 0

    if predicted_adjacency.shape[0] != target_adjacency.shape[0]:
        return 0
    if _edge_count(predicted_adjacency) != _edge_count(target_adjacency):
        return 0
    if not np.array_equal(
        np.sort(predicted_adjacency.sum(axis=1)),
        np.sort(target_adjacency.sum(axis=1)),
    ):
        return 0
    if Counter(map(int, predicted_colors)) != Counter(map(int, target_colors)):
        return 0
    if _component_signatures(
        predicted_adjacency, predicted_colors
    ) != _component_signatures(target_adjacency, target_colors):
        return 0
    if not _wl_refine(
        predicted_adjacency,
        predicted_colors,
        target_adjacency,
        target_colors,
    ):
        return 0
    if not is_planar(predicted_adjacency):
        return 0
    return int(
        are_planar_graphs_isomorphic(
            (predicted_adjacency, predicted_colors),
            (target_adjacency, target_colors),
        )
    )
