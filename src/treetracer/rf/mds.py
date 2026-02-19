"""MDS embedding algorithms for distance matrices."""

import numpy as np


def compute_mds(distance_matrix: np.ndarray, n_components: int = 6,
                algorithm: str = "pcoa") -> np.ndarray:
    """Compute MDS embedding from a precomputed distance matrix.

    Args:
        distance_matrix: Square symmetric distance matrix (n x n).
        n_components: Number of dimensions to return.
        algorithm: MDS algorithm to use. Currently supported: "pcoa".

    Returns:
        Embedding array of shape (n, n_components).

    Raises:
        ValueError: If the algorithm is not recognized.
    """
    algorithms = {
        "pcoa": _pcoa,
    }
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown MDS algorithm '{algorithm}'. "
            f"Supported: {list(algorithms.keys())}"
        )
    return algorithms[algorithm](distance_matrix, n_components)


def _pcoa(distance_matrix: np.ndarray, n_components: int) -> np.ndarray:
    """Classical MDS (PCoA) via double-centering and eigendecomposition."""
    n_components = min(n_components, distance_matrix.shape[0] - 1)
    n = distance_matrix.shape[0]
    D_sq = distance_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * centering @ D_sq @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # eigh returns ascending order; reverse to get largest first
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:n_components]]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    # Clamp negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors * np.sqrt(eigenvalues)
