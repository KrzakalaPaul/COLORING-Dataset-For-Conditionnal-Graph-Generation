"""Generation of the official COLORING dataset.

The public :func:`create_coloring_dataset` function creates one dataset made of
four NumPy arrays: integer images, padded boolean adjacency matrices, padded
node-color vectors, and unpadded graph sizes.  It intentionally has no
split-related behaviour.
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


class ColoringSample(NamedTuple):
    """The unpadded data produced by one COLORING generation worker."""

    image: NDArray[np.integer]
    adjacency_matrix: NDArray[np.bool_]
    node_colors: NDArray[np.integer]
    n_nodes: int


def _smallest_unsigned_dtype(maximum_value: int) -> np.dtype:
    """Return the smallest standard unsigned dtype that stores ``maximum_value``."""
    dtype = np.dtype(np.min_scalar_type(maximum_value))
    if dtype.kind != "u":
        raise ValueError(f"Cannot store {maximum_value} in a NumPy unsigned integer.")
    return dtype


def _validate_seed(seed: int | None) -> None:
    """Validate an optional seed accepted by NumPy's random generators."""
    if seed is not None and (
        not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None.")


def validate_parameters(
    n_samples: int,
    n_pixels: int,
    n_nodes_max: int,
    n_nodes_min: int,
    n_colors: int,
) -> None:
    """Validate the five public COLORING generation parameters."""
    parameters = {
        "n_samples": n_samples,
        "n_pixels": n_pixels,
        "n_nodes_max": n_nodes_max,
        "n_nodes_min": n_nodes_min,
        "n_colors": n_colors,
    }
    for name, value in parameters.items():
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if n_nodes_min > n_nodes_max:
        raise ValueError("n_nodes_min must not exceed n_nodes_max.")
    if n_nodes_max > n_pixels * n_pixels:
        raise ValueError(
            "n_nodes_max cannot exceed n_pixels ** 2: each graph node needs "
            "at least one rasterized pixel."
        )


def sample_centroids(n_nodes: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Sample Voronoi centroids uniformly in the unit square."""
    return rng.uniform(0.0, 1.0, size=(n_nodes, 2))


def rasterize_voronoi(
    centroids: NDArray[np.float64], n_pixels: int
) -> NDArray[np.intp]:
    """Return the nearest-centroid region index at every pixel centre.

    A nearest-neighbour tree avoids materializing the former
    ``(n_pixels, n_pixels, n_nodes, 2)`` distance tensor.  This is both much
    faster and substantially more memory efficient for large images or graphs.
    """
    coordinates = (np.arange(n_pixels, dtype=np.float64) + 0.5) / n_pixels
    pixel_coordinates = np.empty((n_pixels * n_pixels, 2), dtype=np.float64)
    pixel_coordinates[:, 0] = np.repeat(coordinates, n_pixels)
    pixel_coordinates[:, 1] = np.tile(coordinates, n_pixels)
    _, region_indices = cKDTree(centroids).query(pixel_coordinates, workers=1)
    return region_indices.reshape(n_pixels, n_pixels)


def adjacency_from_regions(regions: NDArray[np.intp], n_nodes: int) -> NDArray[np.bool_]:
    """Build region adjacency from horizontal and vertical raster boundaries."""
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)

    horizontal_left = regions[:, :-1]
    horizontal_right = regions[:, 1:]
    vertical_top = regions[:-1, :]
    vertical_bottom = regions[1:, :]

    for first, second in (
        (horizontal_left, horizontal_right),
        (vertical_top, vertical_bottom),
    ):
        different = first != second
        adjacency_matrix[first[different], second[different]] = True
        adjacency_matrix[second[different], first[different]] = True

    return adjacency_matrix


def exact_color_graph(
    adjacency_matrix: NDArray[np.bool_], n_colors: int
) -> NDArray[np.intp] | None:
    """Find a proper coloring with a DSATUR-guided backtracking search.

    DSATUR selects the uncolored node with the greatest number of differently
    colored neighbors, breaking ties by degree.  Unlike a greedy coloring, the
    backtracking search is complete: it returns ``None`` only when no proper
    coloring using ``n_colors`` exists.  The routine is intentionally isolated
    so another coloring implementation can later replace it.
    """
    n_nodes = adjacency_matrix.shape[0]
    neighbors = [
        np.flatnonzero(adjacency_matrix[node]) for node in range(n_nodes)
    ]
    degrees = np.fromiter((len(node_neighbors) for node_neighbors in neighbors), int)
    colors = np.full(n_nodes, -1, dtype=np.intp)
    # ``neighbor_color_counts[node, color]`` supports O(degree) updates and
    # makes saturation-degree computation cheap during search.
    neighbor_color_counts = np.zeros((n_nodes, n_colors), dtype=np.intp)

    def search(n_colored: int) -> bool:
        if n_colored == n_nodes:
            return True

        uncolored = colors == -1
        saturation = (neighbor_color_counts > 0).sum(axis=1)
        saturation[~uncolored] = -1
        highest_saturation = saturation.max()
        candidates = np.flatnonzero(saturation == highest_saturation)
        node = candidates[np.argmax(degrees[candidates])]

        available_colors = np.flatnonzero(neighbor_color_counts[node] == 0)
        # Try colors that constrain the fewest remaining neighbours first.
        color_impacts = np.array(
            [
                sum(
                    colors[neighbor] == -1
                    and neighbor_color_counts[neighbor, color] == 0
                    for neighbor in neighbors[node]
                )
                for color in available_colors
            ]
        )
        for color in available_colors[np.argsort(color_impacts, kind="stable")]:
            colors[node] = color
            for neighbor in neighbors[node]:
                if colors[neighbor] == -1:
                    neighbor_color_counts[neighbor, color] += 1

            if search(n_colored + 1):
                return True

            for neighbor in neighbors[node]:
                if colors[neighbor] == -1:
                    neighbor_color_counts[neighbor, color] -= 1
            colors[node] = -1
        return False

    return colors if search(0) else None


def color_graph(
    adjacency_matrix: NDArray[np.bool_], n_colors: int, rng: np.random.Generator
) -> NDArray[np.intp] | None:
    """Color ``adjacency_matrix`` using the selected coloring strategy.

    This is the pipeline's stable coloring hook.  Replace the delegated
    implementation when evaluating another coloring method.
    """
    del rng  # Reserved in the interface for future randomized strategies.
    return exact_color_graph(adjacency_matrix, n_colors)


def generate_coloring_sample(
    n_pixels: int,
    n_nodes_min: int,
    n_nodes_max: int,
    n_colors: int,
    seed_sequence: np.random.SeedSequence,
) -> ColoringSample:
    """Generate one valid COLORING sample, retrying only failed random draws."""
    rng = np.random.default_rng(seed_sequence)
    # A failed draw can have an empty raster region or be uncolorable with the
    # requested number of colors.  Four colors suffice for these planar graphs.
    max_attempts = 1_000
    for _ in range(max_attempts):
        n_nodes = int(rng.integers(n_nodes_min, n_nodes_max + 1))
        regions = rasterize_voronoi(sample_centroids(n_nodes, rng), n_pixels)
        if np.unique(regions).size != n_nodes:
            continue

        adjacency_matrix = adjacency_from_regions(regions, n_nodes)
        colors = color_graph(adjacency_matrix, n_colors, rng)
        if colors is not None:
            return ColoringSample(colors[regions], adjacency_matrix, colors, n_nodes)

    raise RuntimeError(
        "Could not generate a properly colored sample after 1000 attempts. "
        "Increase n_colors (four colors suffice for planar graphs) or adjust "
        "the requested image and graph sizes."
    )


def _generate_worker(
    arguments: tuple[int, int, int, int, np.random.SeedSequence]
) -> ColoringSample:
    """Pickle-friendly multiprocessing entry point for one sample."""
    n_pixels, n_nodes_min, n_nodes_max, n_colors, seed_sequence = arguments
    return generate_coloring_sample(
        n_pixels, n_nodes_min, n_nodes_max, n_colors, seed_sequence
    )


def create_coloring_dataset(
    output_directory: str | Path,
    n_samples: int,
    n_pixels: int,
    n_nodes_max: int,
    n_nodes_min: int,
    n_colors: int,
    n_workers: int | None = None,
    seed: int | None = None,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    """Create, save, and return one official COLORING dataset.

    Parameters other than ``output_directory`` are the five public dataset
    parameters documented in ``DOC/instruction.md``. ``seed`` optionally makes
    all sampling deterministic, including when multiprocessing is used. The
    output directory is created if needed and receives ``images.npy``,
    ``adjacency_matrices.npy``, ``node_colors.npy``, and ``n_nodes.npy``. Existing files
    with those names are overwritten.

    Returns
    -------
    images, adjacency_matrices, node_colors, n_nodes
        Arrays with shapes ``(n_samples, n_pixels, n_pixels)``,
        ``(n_samples, n_nodes_max, n_nodes_max)``,
        ``(n_samples, n_nodes_max)``, and ``(n_samples,)``.
    """
    validate_parameters(n_samples, n_pixels, n_nodes_max, n_nodes_min, n_colors)
    _validate_seed(seed)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # NPY files created by ``open_memmap`` have the regular NumPy file format
    # while allowing sample-wise writes and later lazy/chunked reads.
    color_dtype = _smallest_unsigned_dtype(n_colors - 1)
    node_count_dtype = _smallest_unsigned_dtype(n_nodes_max)
    images = np.lib.format.open_memmap(
        output_path / "images.npy",
        mode="w+",
        dtype=color_dtype,
        shape=(n_samples, n_pixels, n_pixels),
    )
    adjacency_matrices = np.lib.format.open_memmap(
        output_path / "adjacency_matrices.npy",
        mode="w+",
        dtype=bool,
        shape=(n_samples, n_nodes_max, n_nodes_max),
    )
    node_colors = np.lib.format.open_memmap(
        output_path / "node_colors.npy",
        mode="w+",
        dtype=color_dtype,
        shape=(n_samples, n_nodes_max),
    )
    node_counts = np.lib.format.open_memmap(
        output_path / "n_nodes.npy",
        mode="w+",
        dtype=node_count_dtype,
        shape=(n_samples,),
    )
    adjacency_matrices.fill(False)
    # Padded node colors are ignored; ``n_nodes`` identifies valid entries.
    node_colors.fill(0)

    seed_sequences = np.random.SeedSequence(seed).spawn(n_samples)
    arguments = [
        (n_pixels, n_nodes_min, n_nodes_max, n_colors, seed_sequence)
        for seed_sequence in seed_sequences
    ]
    n_workers = n_workers or min(cpu_count(), n_samples)
    chunk_size = max(1, n_samples // (n_workers * 4))
    with Pool(processes=n_workers) as pool:
        generated_samples = pool.imap(_generate_worker, arguments, chunksize=chunk_size)
        for index, sample in enumerate(tqdm(
            generated_samples,
            total=n_samples,
            desc="Generating COLORING samples",
            unit="sample",
        )):
            images[index] = sample.image
            adjacency_matrices[
                index, : sample.n_nodes, : sample.n_nodes
            ] = sample.adjacency_matrix
            node_colors[index, : sample.n_nodes] = sample.node_colors
            node_counts[index] = sample.n_nodes

    images.flush()
    adjacency_matrices.flush()
    node_colors.flush()
    node_counts.flush()
    return images, adjacency_matrices, node_colors, node_counts
