"""
sort_search.py
==============

Sorting, partial sorting (top-k), selection, and searching utilities built on
NumPy. All public functions validate inputs and raise informative errors.

Conventions
-----------
- Inputs are coerced to ``np.ndarray`` via ``np.asarray``.
- 1D arrays are assumed unless stated otherwise.
- Stable sorts preserve the original order of equal elements.
- Top-k functions return the k largest by default (``largest=True``).

Complexity notes are given in each function's docstring.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_1d_array(a, name: str = "array") -> np.ndarray:
    """Coerce input to a 1D ``np.ndarray`` and validate."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1D, got shape {arr.shape} (ndim={arr.ndim})."
        )
    return arr


def _as_2d_array(a, name: str = "array") -> np.ndarray:
    """Coerce input to a 2D ``np.ndarray`` and validate."""
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2D, got shape {arr.shape} (ndim={arr.ndim})."
        )
    return arr


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def stable_sort(
    a,
    axis: int = -1,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stable sort wrapper around ``np.sort`` / ``np.argsort``.

    A *stable* sort preserves the relative order of elements that compare
    equal. This matters whenever ties must be broken by original position.

    Parameters
    ----------
    a : array_like
        Input array. Must be sortable (numeric or string dtype).
    axis : int, default=-1
        Axis along which to sort.
    return_indices : bool, default=False
        If True, also return the indices that would sort the array
        (i.e. the result of a stable ``argsort``).

    Returns
    -------
    sorted_array : np.ndarray
        Sorted copy of ``a`` along ``axis``.
    indices : np.ndarray, optional
        Only returned if ``return_indices=True``. Stable argsort indices.

    Raises
    ------
    ValueError
        If ``a`` is empty along ``axis``-relevant dimension and cannot be
        coerced to an ndarray, or if the dtype is not orderable.

    Complexity
    ----------
    Time: O(n log n). Space: O(n) for the output copy.

    Examples
    --------
    >>> stable_sort([3, 1, 2])
    array([1, 2, 3])
    >>> stable_sort([3, 1, 2], return_indices=True)
    (array([1, 2, 3]), array([1, 2, 0]))
    """
    arr = np.asarray(a)
    if arr.size == 0:
        # Preserve shape; nothing to sort.
        if return_indices:
            return arr.copy(), np.empty(arr.shape, dtype=np.intp)
        return arr.copy()

    if return_indices:
        idx = np.argsort(arr, axis=axis, kind="stable")
        return np.take_along_axis(arr, idx, axis=axis), idx
    return np.sort(arr, axis=axis, kind="stable")


def multikey_sort(
    X,
    keys: Union[int, Tuple[int, ...]],
    ascending: Union[bool, Tuple[bool, ...]] = True,
) -> np.ndarray:
    """
    Sort rows of a 2D array by multiple columns (primary, secondary, ...).

    Uses ``np.lexsort``. Note ``np.lexsort`` treats the *last* key as primary,
    so we reverse the key order internally — callers pass keys in priority
    order (most significant first).

    Parameters
    ----------
    X : array_like, shape (n, d)
        2D array whose rows will be reordered.
    keys : int or tuple of int
        Column indices to sort by, in priority order (primary first).
    ascending : bool or tuple of bool, default=True
        Sort direction per key. A single bool applies to all keys.

    Returns
    -------
    X_sorted : np.ndarray, shape (n, d)
        Rows of ``X`` reordered.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, or if ``keys``/``ascending`` shapes mismatch,
        or if any key index is out of bounds.

    Complexity
    ----------
    Time: O(k * n log n) where k = len(keys). Space: O(n).

    Examples
    --------
    >>> X = np.array([[1, 20], [1, 10], [2, 5]])
    >>> multikey_sort(X, keys=(0, 1))            # sort by col 0 then col 1
    array([[ 1, 10],
           [ 1, 20],
           [ 2,  5]])
    >>> multikey_sort(X, keys=(0, 1), ascending=(True, False))
    array([[ 1, 20],
           [ 1, 10],
           [ 2,  5]])
    """
    arr = _as_2d_array(X, name="X")

    if isinstance(keys, int):
        keys = (keys,)
    keys = tuple(keys)
    if len(keys) == 0:
        raise ValueError("At least one key must be provided.")

    n_cols = arr.shape[1]
    for k in keys:
        if not -n_cols <= k < n_cols:
            raise ValueError(
                f"Key {k} out of bounds for array with {n_cols} columns."
            )

    if isinstance(ascending, bool):
        ascending = (ascending,) * len(keys)
    ascending = tuple(ascending)
    if len(ascending) != len(keys):
        raise ValueError(
            f"ascending has length {len(ascending)} but keys has length "
            f"{len(keys)}."
        )

    # Build column vectors; negate for descending (only works for numeric).
    cols = []
    for k, asc in zip(keys, ascending):
        col = arr[:, k]
        if not asc:
            if not np.issubdtype(col.dtype, np.number):
                raise ValueError(
                    f"Descending sort on non-numeric column {k} is not "
                    f"supported (dtype={col.dtype})."
                )
            col = -col
        cols.append(col)

    # lexsort: last key is primary → reverse our list.
    order = np.lexsort(cols[::-1])
    return arr[order]


# ---------------------------------------------------------------------------
# Top-k / partial sort
# ---------------------------------------------------------------------------

def topk(
    values,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
    sorted_output: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Return the k largest (or smallest) values from a 1D array.

    Uses ``np.argpartition`` for O(n) average selection, optionally followed
    by an O(k log k) sort of the selected elements. Much faster than a full
    sort when k << n.

    Parameters
    ----------
    values : array_like, 1D
        Input values.
    k : int
        Number of elements to select. Clipped to ``len(values)`` with a
        warning behaviour handled by raising if negative.
    largest : bool, default=True
        If True, return the k largest. If False, the k smallest.
    return_indices : bool, default=True
        If True, return ``(top_values, top_indices)``. Otherwise just
        ``top_values``.
    sorted_output : bool, default=True
        If True, the returned top-k are sorted (descending if ``largest``,
        ascending otherwise). If False, order is unspecified but selection
        is still correct (faster).

    Returns
    -------
    top_values : np.ndarray, shape (min(k, n),)
    top_indices : np.ndarray, shape (min(k, n),), optional

    Raises
    ------
    ValueError
        If ``values`` is not 1D, or ``k`` is negative.

    Complexity
    ----------
    Time: O(n) average for selection + O(k log k) if ``sorted_output``.
    Space: O(n) for argpartition working buffer.

    Examples
    --------
    >>> topk([5, 1, 4, 2, 3], k=3)
    (array([5, 4, 3]), array([0, 2, 4]))
    >>> topk([5, 1, 4, 2, 3], k=2, largest=False)
    (array([1, 2]), array([1, 3]))
    """
    arr = _as_1d_array(values, name="values")
    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {type(k).__name__}.")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")

    n = arr.shape[0]
    if k == 0 or n == 0:
        empty_vals = np.array([], dtype=arr.dtype)
        empty_idx = np.array([], dtype=np.intp)
        if return_indices:
            return empty_vals, empty_idx
        return empty_vals

    k_eff = min(k, n)

    # argpartition: index of k-th element in its sorted position, with
    # smaller-than on the left, larger-than on the right.
    if largest:
        # We want the top k largest → partition at position (n - k_eff).
        # Indices from (n - k_eff) onward are the k largest (unordered).
        part_idx = np.argpartition(arr, n - k_eff)[n - k_eff:]
    else:
        part_idx = np.argpartition(arr, k_eff - 1)[:k_eff]

    top_vals = arr[part_idx]

    if sorted_output:
        order = np.argsort(top_vals, kind="stable")
        if largest:
            order = order[::-1]
        part_idx = part_idx[order]
        top_vals = top_vals[order]

    if return_indices:
        return top_vals, part_idx
    return top_vals


def quickselect(
    values,
    k: int,
) -> float:
    """
    Find the k-th smallest element of a 1D array using Quickselect.

    This is an *educational* pure-Python implementation to demonstrate the
    selection algorithm. For production use, prefer ``topk`` which leverages
    ``np.argpartition``.

    Quickselect is a divide-and-conquer algorithm: pick a pivot, partition
    the array into elements <, ==, > pivot, then recurse into the side that
    contains the k-th element. Average O(n), worst-case O(n²). We use a
    randomised pivot to make the worst case vanishingly unlikely.

    Parameters
    ----------
    values : array_like, 1D
        Input array. Will be copied; not modified in place.
    k : int
        0-indexed rank of the element to find. k=0 returns the minimum,
        k=len(values)-1 returns the maximum.

    Returns
    -------
    kth_value : scalar
        The element that would occupy index k if the array were fully sorted.

    Raises
    ------
    ValueError
        If ``values`` is empty, not 1D, or ``k`` is out of range.

    Complexity
    ----------
    Time: O(n) average, O(n²) worst case. Space: O(1) extra (in-place on copy).

    Examples
    --------
    >>> quickselect([3, 1, 4, 1, 5, 9, 2, 6], k=0)  # minimum
    1
    >>> quickselect([3, 1, 4, 1, 5, 9, 2, 6], k=3)  # 4th smallest
    3
    """
    arr = _as_1d_array(values, name="values").copy()
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Cannot quickselect from an empty array.")
    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {type(k).__name__}.")
    if not 0 <= k < n:
        raise ValueError(f"k={k} out of range for array of length {n}.")

    rng = np.random.default_rng()
    lo, hi = 0, n - 1

    while lo < hi:
        # Randomised pivot → expected O(n) regardless of input ordering.
        pivot_idx = rng.integers(lo, hi + 1)
        pivot = arr[pivot_idx]

        # Lomuto-style three-way partition: [< pivot][== pivot][> pivot].
        # We track two pointers; i scans, lt is end of "<" region,
        # gt is start of ">" region (scanning from the right).
        lt, i, gt = lo, lo, hi
        while i <= gt:
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif arr[i] > pivot:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                i += 1

        # After partition: arr[lo..lt-1] < pivot, arr[lt..gt] == pivot,
        # arr[gt+1..hi] > pivot.
        if k < lt:
            hi = lt - 1
        elif k <= gt:
            return arr[k].item() if hasattr(arr[k], "item") else arr[k]
        else:
            lo = gt + 1

    return arr[k].item() if hasattr(arr[k], "item") else arr[k]


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def binary_search(
    sorted_array,
    x,
    side: str = "left",
) -> Tuple[Union[int, np.ndarray], Union[bool, np.ndarray]]:
    """
    Binary search on a sorted 1D array using ``np.searchsorted``.

    Returns both the insertion index (the position at which ``x`` would be
    inserted to keep ``sorted_array`` sorted) and a boolean indicating
    whether ``x`` is actually present.

    Supports scalar or array queries: if ``x`` is an array, both returns
    are arrays of matching shape.

    Parameters
    ----------
    sorted_array : array_like, 1D
        Array assumed to be sorted in ascending order. *Not* validated
        (validation would be O(n) and defeat the point); caller's
        responsibility.
    x : scalar or array_like
        Value(s) to search for.
    side : {'left', 'right'}, default='left'
        If 'left', returns the first index where ``sorted_array[i] >= x``.
        If 'right', returns the first index where ``sorted_array[i] > x``.
        Only affects the insertion index when ``x`` is present (ties).

    Returns
    -------
    index : int or np.ndarray
        Insertion index (or indices, if ``x`` is an array).
    found : bool or np.ndarray
        Whether ``x`` is present at that location.

    Raises
    ------
    ValueError
        If ``sorted_array`` is not 1D, or ``side`` is not 'left'/'right'.

    Complexity
    ----------
    Time: O(log n) per query, O(m log n) for m queries. Space: O(m).

    Examples
    --------
    >>> binary_search([1, 3, 5, 7, 9], 5)
    (2, True)
    >>> binary_search([1, 3, 5, 7, 9], 4)
    (2, False)
    >>> idx, found = binary_search([1, 3, 5, 7, 9], [3, 4, 11])
    >>> idx.tolist(), found.tolist()
    ([1, 2, 5], [True, False, False])
    """
    arr = _as_1d_array(sorted_array, name="sorted_array")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")

    # searchsorted handles scalars and arrays uniformly.
    idx = np.searchsorted(arr, x, side=side)

    # To check existence, always look at the 'left' insertion index
    # (regardless of `side`) and verify equality. We also guard against
    # out-of-bounds indices when x is larger than all elements.
    left_idx = np.searchsorted(arr, x, side="left")
    in_bounds = left_idx < arr.shape[0]
    # np.where avoids indexing errors for out-of-bounds positions.
    safe_idx = np.where(in_bounds, left_idx, 0)
    found = in_bounds & (arr[safe_idx] == x)

    # Preserve scalar-in → scalar-out contract.
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return int(idx), bool(found)
    return idx, found
