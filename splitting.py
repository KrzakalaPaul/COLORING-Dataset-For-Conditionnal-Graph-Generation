"""WL-aware train/validation/test splitting for COLORING datasets.

The splitter performs an implicit depth-first 1-WL refinement tree.  A final
group is never divided between folds, so graphs indistinguishable by 1-WL are
always assigned the same fold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FoldCode = np.int8
_FOLD_NAMES = np.array(("train", "valid", "test"))


@dataclass(frozen=True)
class SplitSummary:
    """Summary returned after writing a WL-aware split."""

    n_samples: int
    n_wl_groups: int
    requested_sizes: tuple[float, float, float]
    achieved_sizes: tuple[int, int, int]


def _graph_signature(node_state: NDArray[np.int64]) -> bytes:
    """Return an exact, deterministic graph-level signature for a WL state."""
    return np.sort(node_state).tobytes()


def _initial_signature(node_colors: NDArray[np.int64]) -> bytes:
    """Return the round-zero signature based on the stored node features."""
    return _graph_signature(node_colors)


def _initial_groups(
    node_colors: np.memmap, n_nodes: np.memmap
) -> Iterable[NDArray[np.int64]]:
    """Group graph indices by their node-feature multiset (WL round zero)."""
    groups: dict[bytes, list[int]] = {}
    for index, n_nodes_in_graph in enumerate(n_nodes):
        node_count = int(n_nodes_in_graph)
        signature = _initial_signature(node_colors[index, :node_count])
        groups.setdefault(signature, []).append(index)
    for indices in groups.values():
        yield np.asarray(indices, dtype=np.int64)


def _load_initial_states(
    indices: NDArray[np.int64], node_colors: np.memmap, n_nodes: np.memmap
) -> list[NDArray[np.int64]]:
    """Load unpadded node features as the round-zero WL node states."""
    return [
        np.asarray(node_colors[index, : int(n_nodes[index])], dtype=np.int64).copy()
        for index in indices
    ]


def _refine_once(
    indices: NDArray[np.int64],
    states: list[NDArray[np.int64]],
    adjacency_matrices: np.memmap,
    n_nodes: np.memmap,
) -> list[NDArray[np.int64]]:
    """Apply one globally canonical 1-WL refinement round to an active group."""
    if len(indices) != len(states):
        raise ValueError("Each active graph index must have exactly one WL state.")
    token_ids: dict[tuple[int, tuple[int, ...]], int] = {}
    refined_states: list[NDArray[np.int64]] = []

    for index, state in zip(indices, states):
        node_count = int(n_nodes[index])
        adjacency_matrix = adjacency_matrices[index, :node_count, :node_count]
        refined = np.empty(node_count, dtype=np.int64)
        for node in range(node_count):
            neighbor_colors = tuple(
                sorted(int(color) for color in state[adjacency_matrix[node]])
            )
            token = (int(state[node]), neighbor_colors)
            refined[node] = token_ids.setdefault(token, len(token_ids))
        refined_states.append(refined)
    return refined_states


def _states_at_round(
    indices: NDArray[np.int64],
    round_index: int,
    adjacency_matrices: np.memmap,
    node_colors: np.memmap,
    n_nodes: np.memmap,
) -> list[NDArray[np.int64]]:
    """Recompute an active group's shared WL states up to ``round_index``."""
    states = _load_initial_states(indices, node_colors, n_nodes)
    for _ in range(round_index):
        states = _refine_once(indices, states, adjacency_matrices, n_nodes)
    return states


def _partition_refined_states(
    indices: NDArray[np.int64], states: list[NDArray[np.int64]]
) -> Iterable[tuple[NDArray[np.int64], list[NDArray[np.int64]]]]:
    """Partition a group by its refined graph-level WL signatures."""
    if len(indices) != len(states):
        raise ValueError("Each active graph index must have exactly one WL state.")
    partitions: dict[bytes, tuple[list[int], list[NDArray[np.int64]]]] = {}
    for index, state in zip(indices, states):
        signature = _graph_signature(state)
        if signature not in partitions:
            partitions[signature] = ([], [])
        partitions[signature][0].append(int(index))
        partitions[signature][1].append(state)

    for child_indices, child_states in partitions.values():
        yield np.asarray(child_indices, dtype=np.int64), child_states


def _validate_split_parameters(
    train_fraction: float,
    valid_fraction: float,
    test_fraction: float,
    n_max_wl_test: int,
    state_threshold: int,
) -> None:
    """Validate split proportions and WL traversal limits."""
    fractions = (train_fraction, valid_fraction, test_fraction)
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("Fold fractions must be non-negative.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("train_fraction, valid_fraction, and test_fraction must sum to 1.")
    if not isinstance(n_max_wl_test, int) or n_max_wl_test < 0:
        raise ValueError("n_max_wl_test must be a non-negative integer.")
    if not isinstance(state_threshold, int) or state_threshold < 1:
        raise ValueError("state_threshold must be a positive integer.")


def _validate_seed(seed: int | None) -> None:
    """Validate an optional random seed for fold-assignment tie-breaking."""
    if seed is not None and (
        not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None.")


def _write_split_csv(output_csv: Path, fold_codes: NDArray[FoldCode]) -> None:
    """Write exactly one fold value per sample in dataset-index order."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 100_000
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        for start in range(0, fold_codes.size, chunk_size):
            names = _FOLD_NAMES[fold_codes[start : start + chunk_size]]
            file.write("\n".join(names))
            file.write("\n")


def _assign_groups_to_folds(
    group_sizes: NDArray[np.int64],
    fractions: NDArray[np.float64],
    seed: int | None,
) -> NDArray[FoldCode]:
    """Assign indivisible groups by largest-first greedy bin packing.

    A supplied seed deterministically resolves equal-size-group and equal-fold
    ties. Without one, ties use stable dataset-index order.
    """
    target_sizes = fractions * group_sizes.sum()
    remaining_sizes = target_sizes.copy()
    group_folds = np.empty(group_sizes.size, dtype=FoldCode)
    rng = np.random.default_rng(seed) if seed is not None else None

    if rng is None:
        group_order = np.argsort(-group_sizes, kind="stable")
    else:
        group_order = np.lexsort((rng.random(group_sizes.size), -group_sizes))

    for group_id in group_order:
        largest_remaining = remaining_sizes.max()
        candidate_folds = np.flatnonzero(
            np.isclose(remaining_sizes, largest_remaining)
        )
        fold = int(candidate_folds[0]) if rng is None else int(rng.choice(candidate_folds))
        group_folds[group_id] = fold
        remaining_sizes[fold] -= group_sizes[group_id]
    return group_folds


def create_wl_split(
    dataset_directory: str | Path,
    output_csv: str | Path,
    train_fraction: float = 0.9,
    valid_fraction: float = 0.05,
    test_fraction: float = 0.05,
    n_max_wl_test: int = 10,
    state_threshold: int = 10_000,
    seed: int | None = None,
) -> SplitSummary:
    """Create a train/validation/test CSV without cross-fold 1-WL duplicates.

    The initial WL node colors are read from ``node_colors.npy``.  The function
    uses an implicit depth-first refinement tree: node states are retained only
    for active groups no larger than ``state_threshold``.  Larger groups
    recompute their state when revisited, trading CPU time for bounded memory.

    ``n_max_wl_test`` limits the number of refinement rounds.  A limit reached
    before convergence is conservative: it can combine extra graphs, but it
    cannot separate graphs that full 1-WL leaves indistinguishable.

    ``seed`` makes tie-breaking in group-to-fold assignment reproducible.
    Without a seed, stable dataset-index order resolves ties.
    """
    _validate_split_parameters(
        train_fraction,
        valid_fraction,
        test_fraction,
        n_max_wl_test,
        state_threshold,
    )
    _validate_seed(seed)
    dataset_path = Path(dataset_directory)
    adjacency_matrices = np.load(
        dataset_path / "adjacency_matrices.npy", mmap_mode="r"
    )
    node_colors = np.load(dataset_path / "node_colors.npy", mmap_mode="r")
    n_nodes = np.load(dataset_path / "n_nodes.npy", mmap_mode="r")

    n_samples = int(n_nodes.size)
    if adjacency_matrices.shape[0] != n_samples or node_colors.shape[0] != n_samples:
        raise ValueError("Dataset arrays do not have the same number of samples.")
    if (
        adjacency_matrices.shape[1] != adjacency_matrices.shape[2]
        or adjacency_matrices.shape[1] != node_colors.shape[1]
    ):
        raise ValueError("Graph and node-color array shapes are incompatible.")

    # Each sample receives exactly one terminal WL-equivalence-group ID.
    group_ids = np.empty(n_samples, dtype=np.int64)
    group_sizes: list[int] = []

    def emit_terminal_group(indices: NDArray[np.int64]) -> None:
        group_ids[indices] = len(group_sizes)
        group_sizes.append(indices.size)

    # A stack entry is: sample indices, current WL round, optional node states.
    # A state list is retained only below the caller-selected size threshold.
    stack: list[tuple[NDArray[np.int64], int, list[NDArray[np.int64]] | None]] = []
    for initial_indices in _initial_groups(node_colors, n_nodes):
        if initial_indices.size == 1 or n_max_wl_test == 0:
            emit_terminal_group(initial_indices)
        else:
            initial_states = (
                _load_initial_states(initial_indices, node_colors, n_nodes)
                if initial_indices.size <= state_threshold
                else None
            )
            stack.append((initial_indices, 0, initial_states))

    while stack:
        indices, current_round, states = stack.pop()
        if states is None:
            states = _states_at_round(
                indices, current_round, adjacency_matrices, node_colors, n_nodes
            )

        refined_states = _refine_once(indices, states, adjacency_matrices, n_nodes)
        next_round = current_round + 1
        # All graphs in an active group have the same current graph signature,
        # hence the same number of current WL colors.  Refinement only splits
        # color classes, so equal counts mean that a child is stable.
        current_color_count = np.unique(states[0]).size
        for child_indices, child_states in _partition_refined_states(indices, refined_states):
            child_is_stable = np.unique(child_states[0]).size == current_color_count
            if child_indices.size == 1 or child_is_stable or next_round >= n_max_wl_test:
                emit_terminal_group(child_indices)
            else:
                retained_states = child_states if child_indices.size <= state_threshold else None
                stack.append((child_indices, next_round, retained_states))

    group_sizes_array = np.asarray(group_sizes, dtype=np.int64)
    fractions = np.array((train_fraction, valid_fraction, test_fraction))
    group_folds = _assign_groups_to_folds(group_sizes_array, fractions, seed)
    fold_codes = group_folds[group_ids]
    _write_split_csv(Path(output_csv), fold_codes)

    achieved_sizes = tuple(int((fold_codes == fold).sum()) for fold in range(3))
    requested_sizes = tuple(float(value) for value in fractions * n_samples)
    return SplitSummary(
        n_samples=n_samples,
        n_wl_groups=len(group_sizes),
        requested_sizes=requested_sizes,
        achieved_sizes=achieved_sizes,
    )
