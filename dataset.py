"""NumPy-backed dataset access for stored COLORING datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ColoringDataSample:
    """One unpadded COLORING image/graph pair.

    ``index`` always refers to the original dataset index, including when a
    :class:`ColoringDataset` is filtered to one fold.
    """

    index: int
    image: NDArray[np.unsignedinteger]
    adjacency_matrix: NDArray[np.bool_]
    node_colors: NDArray[np.unsignedinteger]
    n_nodes: int
    fold: str | None = None


class ColoringDataset:
    """Lazy, NumPy-only access to a generated COLORING dataset.

    The arrays are opened with ``mmap_mode='r'``.  Integer indexing returns an
    unpadded :class:`ColoringDataSample`, while its image remains a view of the
    memory-mapped array whenever NumPy permits it.

    Parameters
    ----------
    dataset_directory:
        Directory containing the generated ``.npy`` files.
    folds_csv:
        Optional headerless fold CSV produced by :func:`splitting.create_wl_split`.
    fold:
        Optional ``'train'``, ``'valid'``, or ``'test'`` filter. Requires
        ``folds_csv``.
    """

    def __init__(
        self,
        dataset_directory: str | Path,
        folds_csv: str | Path | None = None,
        fold: str | None = None,
    ) -> None:
        self.dataset_directory = Path(dataset_directory)
        self.images = np.load(self.dataset_directory / "images.npy", mmap_mode="r")
        self.adjacency_matrices = np.load(
            self.dataset_directory / "adjacency_matrices.npy", mmap_mode="r"
        )
        self.node_colors = np.load(
            self.dataset_directory / "node_colors.npy", mmap_mode="r"
        )
        self.n_nodes = np.load(self.dataset_directory / "n_nodes.npy", mmap_mode="r")
        self._validate_arrays()

        self.folds: NDArray[np.str_] | None = None
        if folds_csv is not None:
            self.folds = np.atleast_1d(np.loadtxt(folds_csv, dtype=str))
            if self.folds.size != self.n_nodes.size:
                raise ValueError("The fold CSV must contain one row per dataset sample.")
            invalid_folds = set(self.folds) - {"train", "valid", "test"}
            if invalid_folds:
                raise ValueError(f"Unknown fold values: {sorted(invalid_folds)}.")

        if fold is not None and fold not in {"train", "valid", "test"}:
            raise ValueError("fold must be 'train', 'valid', 'test', or None.")
        if fold is not None and self.folds is None:
            raise ValueError("fold requires a folds_csv file.")
        self.fold = fold
        self._indices = (
            np.flatnonzero(self.folds == fold)
            if fold is not None
            else np.arange(self.n_nodes.size)
        )

    def _validate_arrays(self) -> None:
        """Validate the fixed dataset-array storage contract."""
        n_samples = self.n_nodes.size
        if self.images.shape[0] != n_samples:
            raise ValueError("images.npy and n_nodes.npy have incompatible shapes.")
        if (
            self.adjacency_matrices.shape[0] != n_samples
            or self.node_colors.shape[0] != n_samples
        ):
            raise ValueError("Graph arrays and n_nodes.npy have incompatible shapes.")
        if (
            self.adjacency_matrices.ndim != 3
            or self.adjacency_matrices.shape[1] != self.adjacency_matrices.shape[2]
        ):
            raise ValueError("adjacency_matrices.npy must contain square matrices.")
        if self.adjacency_matrices.shape[1] != self.node_colors.shape[1]:
            raise ValueError(
                "adjacency_matrices.npy and node_colors.npy have incompatible widths."
            )
        if np.any(self.n_nodes > self.adjacency_matrices.shape[1]):
            raise ValueError("n_nodes.npy exceeds the adjacency-matrix padding.")

    def __len__(self) -> int:
        """Return the number of samples visible through this dataset view."""
        return int(self._indices.size)

    def __getitem__(self, item: int) -> ColoringDataSample:
        """Return one unpadded sample using torch-style integer indexing."""
        if not isinstance(item, (int, np.integer)):
            raise TypeError("ColoringDataset supports integer sample indexing only.")
        dataset_position = int(item)
        if dataset_position < 0:
            dataset_position += len(self)
        if not 0 <= dataset_position < len(self):
            raise IndexError("ColoringDataset index out of range.")

        index = int(self._indices[dataset_position])
        node_count = int(self.n_nodes[index])
        sample_fold = None if self.folds is None else str(self.folds[index])
        return ColoringDataSample(
            index=index,
            image=self.images[index],
            adjacency_matrix=self.adjacency_matrices[index, :node_count, :node_count],
            node_colors=self.node_colors[index, :node_count],
            n_nodes=node_count,
            fold=sample_fold,
        )
