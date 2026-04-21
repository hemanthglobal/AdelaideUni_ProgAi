# Author: Musfir Ahmad Nazir (a1990098) — Member 2 (preprocessing.py, utils.py)

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import Generator


def _to_2d(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Convert input to a 2-D float64 array.

    Parameters
    ----------
    X    : array-like
    name : str — variable name used in error messages

    Returns
    -------
    X : ndarray, shape (n, p), dtype float64
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {X.shape}.")
    return X


def euclidean_distance(
    A: ArrayLike,
    B: ArrayLike | None = None,
    *,
    squared: bool = False,
) -> np.ndarray:
    """Compute pairwise Euclidean distances between rows of A and B.

    Uses the identity  ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    implemented as a matrix multiply to avoid an explicit outer loop.
    Floating-point noise (tiny negatives) is clipped to 0 before sqrt.

    Parameters
    ----------
    A       : array-like (n, p)
    B       : array-like (m, p) or None — if None, pairwise within A
    squared : bool, default False — return squared distances (skip sqrt)

    Returns
    -------
    D : ndarray (n, m) — D[i, j] = ||A[i] - B[j]||

    Raises
    ------
    ValueError : if A and B have different number of columns

    Time  : O(n * m * p)
    Space : O(n * m)
    """
    A = _to_2d(A, "A")
    B = A if B is None else _to_2d(B, "B")

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A has {A.shape[1]} columns; B has {B.shape[1]}. Must match.")

    AA = np.einsum("ij,ij->i", A, A)
    BB = np.einsum("ij,ij->i", B, B)
    AB = A @ B.T

    D2 = AA[:, np.newaxis] + BB[np.newaxis, :] - 2.0 * AB
    np.clip(D2, 0.0, None, out=D2)

    return D2 if squared else np.sqrt(D2)


def cosine_distance(
    A: ArrayLike,
    B: ArrayLike | None = None,
) -> np.ndarray:
    """Compute pairwise cosine distances (1 - cosine similarity).

    Rows are L2-normalised before the dot product.
    Zero-norm rows are assigned distance 1 (undefined similarity).
    Result is clipped to [0, 2] to handle floating-point edge cases.

    Parameters
    ----------
    A : array-like (n, p)
    B : array-like (m, p) or None — if None, pairwise within A

    Returns
    -------
    D : ndarray (n, m) — values in [0, 2]; 0 = identical direction

    Raises
    ------
    ValueError : if A and B have different number of columns

    Time  : O(n * m * p)
    Space : O(n * m)
    """
    A = _to_2d(A, "A")
    B = A if B is None else _to_2d(B, "B")

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A has {A.shape[1]} columns; B has {B.shape[1]}. Must match.")

    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)

    norm_A[norm_A == 0] = 1.0
    norm_B[norm_B == 0] = 1.0

    sim = (A / norm_A) @ (B / norm_B).T
    np.clip(sim, -1.0, 1.0, out=sim)

    return 1.0 - sim


def logsumexp(
    X: ArrayLike,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    """Compute log(sum(exp(X))) in a numerically stable way.

    Stability trick — subtract the per-slice maximum before exponentiating,
    then add it back:  LSE(x) = max(x) + log(sum(exp(x - max(x))))
    This prevents overflow for large values (e.g. x = 1000).

    Parameters
    ----------
    X        : array-like, any shape
    axis     : int or None — reduction axis; None reduces entire array
    keepdims : bool, default False

    Returns
    -------
    out : ndarray — same shape as X with axis removed (or kept if keepdims)

    Time  : O(X.size)
    Space : O(X.size)
    """
    X = np.asarray(X, dtype=np.float64)
    x_max = np.max(X, axis=axis, keepdims=True)
    x_max_safe = np.where(np.isneginf(x_max), 0.0, x_max)
    out = np.log(np.sum(np.exp(X - x_max_safe), axis=axis, keepdims=True))
    out += x_max_safe
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(
    X: ArrayLike,
    axis: int = -1,
) -> np.ndarray:
    """Numerically stable softmax along the given axis.

    Uses max-shift:  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    Output values are in (0, 1) and sum to 1 along the given axis.

    Parameters
    ----------
    X    : array-like, any shape
    axis : int, default -1

    Returns
    -------
    out : ndarray — same shape as X, sums to 1 along axis

    Time  : O(X.size)
    Space : O(X.size)
    """
    X = np.asarray(X, dtype=np.float64)
    x_max = np.max(X, axis=axis, keepdims=True)
    e_x = np.exp(X - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax(
    X: ArrayLike,
    axis: int = -1,
) -> np.ndarray:
    """Numerically stable log-softmax along the given axis.

    Computed as  X - logsumexp(X)  rather than  log(softmax(X))
    to preserve precision for very small probabilities.

    Parameters
    ----------
    X    : array-like, any shape
    axis : int, default -1

    Returns
    -------
    out : ndarray — same shape as X, log-probabilities

    Time  : O(X.size)
    Space : O(X.size)
    """
    X = np.asarray(X, dtype=np.float64)
    return X - logsumexp(X, axis=axis, keepdims=True)


def sigmoid(X: ArrayLike) -> np.ndarray:
    """Element-wise sigmoid (logistic) function.

    Uses a two-branch stable implementation to avoid overflow:
        x >= 0 :  1 / (1 + exp(-x))
        x <  0 :  exp(x) / (1 + exp(x))
    Both branches are mathematically identical but each avoids
    overflow in the other's domain.

    Parameters
    ----------
    X : array-like, any shape

    Returns
    -------
    out : ndarray — same shape as X, values in (0, 1)

    Time  : O(X.size)
    Space : O(X.size)
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    pos = X >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-X[pos]))
    exp_x = np.exp(X[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def relu(X: ArrayLike) -> np.ndarray:
    """Element-wise Rectified Linear Unit:  relu(x) = max(0, x).

    Parameters
    ----------
    X : array-like, any shape

    Returns
    -------
    out : ndarray — same shape as X, values >= 0

    Time  : O(X.size)
    Space : O(X.size)
    """
    return np.maximum(0.0, np.asarray(X, dtype=np.float64))


def batch_iterator(
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    """Yield mini-batches of (X,) or (X, y) without copying the data.

    The last batch may be smaller than batch_size if the data does not
    divide evenly. Uses index slicing so no data is copied for contiguous arrays.

    Parameters
    ----------
    X            : array-like (n_samples, ...)
    y            : array-like (n_samples, ...) or None
    batch_size   : int, default 32
    shuffle      : bool, default False — shuffle indices before batching
    random_state : int, Generator, or None — seed for reproducible shuffles

    Yields
    ------
    (X_batch,)          when y is None
    (X_batch, y_batch)  when y is provided

    Raises
    ------
    ValueError : if batch_size < 1 or X and y have different n_samples

    Time  : O(n_samples) per full pass
    Space : O(1) beyond the yielded slices
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    X = np.asarray(X)
    n = X.shape[0]

    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != n:
            raise ValueError(f"X has {n} samples; y has {y.shape[0]}. Must match.")

    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        if y is None:
            yield (X[batch_idx],)
        else:
            yield (X[batch_idx], y[batch_idx])
