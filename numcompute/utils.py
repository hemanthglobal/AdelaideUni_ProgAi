# Author: Musfir Ahmad Nazir (a1990098) — Member 2 (preprocessing.py, utils.py)

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import Generator


def _to_2d(X: ArrayLike, name: str = "X") -> np.ndarray:
    # Converts any input into a 2D float64 array.
    # 1D inputs are treated as a single row.
    # Raises a clear error if the shape is completely wrong.
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
    # Computes straight-line distances between every row of A and every row of B.
    # If B is not given, computes pairwise distances within A itself.
    #
    # Instead of looping over every pair of rows, uses the algebraic identity:
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a dot b)
    # The dot product part becomes a matrix multiply (A @ B.T) which NumPy
    # runs much faster than any Python loop.
    #
    # Floating point arithmetic can produce tiny negatives like -1e-15 before
    # the sqrt, so we clip anything below 0 to exactly 0 to avoid NaN.
    # Use squared=True to skip the sqrt entirely.
    A = _to_2d(A, "A")
    B = A if B is None else _to_2d(B, "B")

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A has {A.shape[1]} columns; B has {B.shape[1]}. Must match.")

    AA = np.einsum("ij,ij->i", A, A)   # squared row norms of A
    BB = np.einsum("ij,ij->i", B, B)   # squared row norms of B
    AB = A @ B.T                        # cross term via matrix multiply

    D2 = AA[:, np.newaxis] + BB[np.newaxis, :] - 2.0 * AB
    np.clip(D2, 0.0, None, out=D2)     # remove floating point noise before sqrt

    return D2 if squared else np.sqrt(D2)


def cosine_distance(
    A: ArrayLike,
    B: ArrayLike | None = None,
) -> np.ndarray:
    # Measures the angle between vectors rather than straight-line distance.
    # Doesn't care about the length of vectors, only the direction they point.
    # Result is always between 0 and 2:
    #   0 = same direction, 1 = perpendicular, 2 = completely opposite
    #
    # Each row is normalised to length 1 first, then we do a dot product.
    # Zero-norm rows (all zeros) have no direction — we set their norm to 1
    # so the output is 1 (neutral) instead of crashing with division by zero.
    A = _to_2d(A, "A")
    B = A if B is None else _to_2d(B, "B")

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A has {A.shape[1]} columns; B has {B.shape[1]}. Must match.")

    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)

    # Replace zero norms with 1 to avoid division by zero
    norm_A[norm_A == 0] = 1.0
    norm_B[norm_B == 0] = 1.0

    sim = (A / norm_A) @ (B / norm_B).T
    np.clip(sim, -1.0, 1.0, out=sim)   # clamp for floating point safety

    return 1.0 - sim                    # convert similarity to distance


def logsumexp(
    X: ArrayLike,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    # Computes log(sum(exp(x))) without overflowing.
    #
    # The problem: exp(1000) is infinity on a computer.
    # The fix (max-shift trick):
    #   1. Find the max value in x
    #   2. Subtract it from everything before exp()
    #   3. Add it back after log()
    # The maths works out exactly the same but it never overflows.
    # This is used internally by softmax and log_softmax.
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
    # Converts raw scores into probabilities that sum to 1.
    # Used in classification — the final layer of a neural network uses this
    # to turn scores into "probability of each class".
    #
    # Uses the same max-shift trick as logsumexp to prevent overflow.
    # Every row of the output sums to exactly 1.0.
    X = np.asarray(X, dtype=np.float64)
    x_max = np.max(X, axis=axis, keepdims=True)
    e_x = np.exp(X - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax(
    X: ArrayLike,
    axis: int = -1,
) -> np.ndarray:
    # Log of the softmax probabilities.
    # Computed as x - logsumexp(x) rather than log(softmax(x)).
    # Both give the same answer mathematically but this way keeps more
    # precision when probabilities are very small — log(tiny number) loses
    # accuracy quickly, this approach avoids that problem entirely.
    X = np.asarray(X, dtype=np.float64)
    return X - logsumexp(X, axis=axis, keepdims=True)


def sigmoid(X: ArrayLike) -> np.ndarray:
    # Squishes any number into the range (0, 1).
    # Used in binary classification to output a probability.
    # Formula: 1 / (1 + exp(-x))
    #
    # Stability issue: for very negative x, exp(-x) overflows to infinity.
    # Fix — use two branches that are mathematically identical:
    #   x >= 0 : 1 / (1 + exp(-x))       safe for positive x
    #   x <  0 : exp(x) / (1 + exp(x))   safe for negative x
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    pos = X >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-X[pos]))
    exp_x = np.exp(X[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def relu(X: ArrayLike) -> np.ndarray:
    # Rectified Linear Unit — the simplest activation function.
    # Anything negative becomes 0, positive values stay unchanged.
    # Widely used between layers in neural networks.
    return np.maximum(0.0, np.asarray(X, dtype=np.float64))


def batch_iterator(
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    # Splits the dataset into small chunks (mini-batches) and yields them one
    # at a time. Training on the full dataset at once can exhaust memory —
    # batching solves that by feeding data in manageable pieces.
    #
    # This is a generator so it never loads all data into memory at once.
    # It slices using index arrays so no data is copied for contiguous arrays.
    # The last batch may be smaller if data doesn't divide evenly — that's fine.
    #
    # shuffle=True randomises the order before splitting into batches.
    # random_state lets you set a seed so results are reproducible.
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
