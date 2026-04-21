"""
optim.py
========

Finite-difference gradient and Jacobian estimation.

These routines approximate derivatives numerically by evaluating the function
at small perturbations of the input. They are useful for verifying analytic
gradients, working with black-box functions, or teaching the underlying
calculus. They are *not* a substitute for analytic derivatives in performance-
critical code — each gradient costs 2n function evaluations (central diff).

Numerical considerations
------------------------
- The step size ``h`` trades off two errors: *truncation error* (grows as h
  increases; O(h) for forward, O(h²) for central) and *round-off error*
  (grows as h decreases, because ``f(x+h) - f(x-h)`` loses precision when
  the two values are nearly equal). The optimal h for central differences
  on float64 is roughly ``eps**(1/3) ≈ 6e-6``; for forward differences,
  ``sqrt(eps) ≈ 1.5e-8``. We scale h by ``max(|x_i|, 1)`` per component
  to stay numerically sensible when x has mixed magnitudes.
- We use float64 internally regardless of input dtype.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_float_1d(x, name: str = "x") -> np.ndarray:
    """Coerce to a 1D float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 0D or 1D, got shape {arr.shape} (ndim={arr.ndim})."
        )
    return arr


def _step_sizes(x: np.ndarray, h: float) -> np.ndarray:
    """
    Per-component step sizes, scaled by magnitude of x.

    Returns h * max(|x_i|, 1) for each component, so components with large
    magnitude get proportionally larger steps (avoids losing precision) and
    small/zero components get the baseline h.
    """
    return h * np.maximum(np.abs(x), 1.0)


# ---------------------------------------------------------------------------
# Gradient of a scalar function
# ---------------------------------------------------------------------------

def grad(
    f: Callable[[np.ndarray], float],
    x,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """
    Finite-difference gradient of a scalar-valued function.

    Approximates ∇f(x) = [∂f/∂x_0, ∂f/∂x_1, ..., ∂f/∂x_{n-1}].

    Parameters
    ----------
    f : callable
        Function ``f(x) -> scalar`` where x is a 1D array of shape (n,).
        Must return a Python float or a 0D NumPy array.
    x : array_like, shape (n,)
        Point at which to evaluate the gradient.
    h : float, default=1e-5
        Base step size. Actual per-component step is scaled by
        ``max(|x_i|, 1)``.
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme. Central is O(h²) accurate and costs 2n
        evaluations; forward is O(h) accurate and costs n+1 evaluations.

    Returns
    -------
    g : np.ndarray, shape (n,)
        Estimated gradient ∇f(x).

    Raises
    ------
    ValueError
        If ``x`` is not 0D/1D, ``h <= 0``, or ``method`` is invalid.
    TypeError
        If ``f`` is not callable.

    Complexity
    ----------
    Time: O(n * cost(f)). Space: O(n).

    Examples
    --------
    >>> f = lambda x: (x ** 2).sum()
    >>> grad(f, [1.0, 2.0, 3.0])  # analytic: [2, 4, 6]
    array([2., 4., 6.])

    >>> f = lambda x: np.sin(x[0]) + np.cos(x[1])
    >>> g = grad(f, [0.0, 0.0])   # analytic: [cos(0), -sin(0)] = [1, 0]
    >>> np.allclose(g, [1.0, 0.0])
    True
    """
    if not callable(f):
        raise TypeError("f must be callable.")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}.")
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got {method!r}.")

    x = _as_float_1d(x, name="x")
    n = x.shape[0]
    steps = _step_sizes(x, h)
    g = np.empty(n, dtype=np.float64)

    if method == "central":
        # (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += steps[i]
            x_minus[i] -= steps[i]
            g[i] = (float(f(x_plus)) - float(f(x_minus))) / (2.0 * steps[i])
    else:  # forward
        # (f(x + h*e_i) - f(x)) / h
        f_x = float(f(x))
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += steps[i]
            g[i] = (float(f(x_plus)) - f_x) / steps[i]

    return g


# ---------------------------------------------------------------------------
# Jacobian of a vector function
# ---------------------------------------------------------------------------

def jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """
    Finite-difference Jacobian of a vector-valued function.

    For F: R^n -> R^m, returns the m-by-n matrix J where
    ``J[i, j] = ∂F_i / ∂x_j`` evaluated at x.

    Parameters
    ----------
    F : callable
        Function ``F(x) -> np.ndarray`` of shape (m,) where x has shape (n,).
        The output dimension m is inferred from ``F(x)``.
    x : array_like, shape (n,)
        Point at which to evaluate the Jacobian.
    h : float, default=1e-5
        Base step size (scaled per component as in ``grad``).
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme.

    Returns
    -------
    J : np.ndarray, shape (m, n)
        Estimated Jacobian. If F returns a scalar, J has shape (1, n).

    Raises
    ------
    ValueError
        If inputs or outputs have unexpected shape, or ``h <= 0``,
        or ``method`` is invalid.
    TypeError
        If ``F`` is not callable.

    Complexity
    ----------
    Time: O(n * cost(F)). Space: O(m * n).

    Examples
    --------
    >>> F = lambda x: np.array([x[0] ** 2, x[0] * x[1]])
    >>> J = jacobian(F, [2.0, 3.0])   # analytic: [[2*x0, 0], [x1, x0]]
    >>> np.allclose(J, [[4.0, 0.0], [3.0, 2.0]])
    True
    """
    if not callable(F):
        raise TypeError("F must be callable.")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}.")
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got {method!r}.")

    x = _as_float_1d(x, name="x")
    n = x.shape[0]

    # Probe F to determine output dimension m.
    f0 = np.atleast_1d(np.asarray(F(x), dtype=np.float64))
    if f0.ndim != 1:
        raise ValueError(
            f"F must return a 1D array (or scalar); got shape {f0.shape}."
        )
    m = f0.shape[0]

    steps = _step_sizes(x, h)
    J = np.empty((m, n), dtype=np.float64)

    if method == "central":
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += steps[j]
            x_minus[j] -= steps[j]
            f_plus = np.atleast_1d(np.asarray(F(x_plus), dtype=np.float64))
            f_minus = np.atleast_1d(np.asarray(F(x_minus), dtype=np.float64))
            if f_plus.shape != (m,) or f_minus.shape != (m,):
                raise ValueError(
                    f"F returned inconsistent output shape at column {j}."
                )
            J[:, j] = (f_plus - f_minus) / (2.0 * steps[j])
    else:  # forward
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += steps[j]
            f_plus = np.atleast_1d(np.asarray(F(x_plus), dtype=np.float64))
            if f_plus.shape != (m,):
                raise ValueError(
                    f"F returned inconsistent output shape at column {j}."
                )
            J[:, j] = (f_plus - f0) / steps[j]

    return J


# ---------------------------------------------------------------------------
# Optional: line search (spec says optional; short, clean implementation)
# ---------------------------------------------------------------------------

def line_search_backtracking(
    f: Callable[[np.ndarray], float],
    x,
    direction,
    grad_x=None,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 50,
) -> float:
    """
    Backtracking line search satisfying the Armijo (sufficient decrease) condition.

    Finds a step size α > 0 along ``direction`` such that
    ``f(x + α*d) <= f(x) + c1 * α * <grad_x, d>``. Starts at ``alpha0`` and
    multiplies by ``rho`` until the condition holds (or ``max_iter`` reached).

    Parameters
    ----------
    f : callable
        Scalar-valued function.
    x : array_like, shape (n,)
        Current point.
    direction : array_like, shape (n,)
        Search direction (typically -grad for gradient descent).
    grad_x : array_like, shape (n,), optional
        Gradient at x. If None, estimated via ``grad(f, x)``.
    alpha0 : float, default=1.0
        Initial step size.
    c1 : float, default=1e-4
        Armijo constant, 0 < c1 < 1 (usually small).
    rho : float, default=0.5
        Shrink factor, 0 < rho < 1.
    max_iter : int, default=50
        Maximum backtracking iterations.

    Returns
    -------
    alpha : float
        Step size satisfying the Armijo condition, or the final ``alpha`` if
        ``max_iter`` is exhausted.

    Raises
    ------
    ValueError
        If ``direction`` is not a descent direction (``<grad, d> >= 0``), or
        parameters are out of range.
    """
    if not (0 < c1 < 1):
        raise ValueError(f"c1 must be in (0, 1), got {c1}.")
    if not (0 < rho < 1):
        raise ValueError(f"rho must be in (0, 1), got {rho}.")
    if alpha0 <= 0:
        raise ValueError(f"alpha0 must be positive, got {alpha0}.")

    x = _as_float_1d(x, name="x")
    d = _as_float_1d(direction, name="direction")
    if d.shape != x.shape:
        raise ValueError(
            f"direction shape {d.shape} does not match x shape {x.shape}."
        )

    if grad_x is None:
        grad_x = grad(f, x)
    else:
        grad_x = _as_float_1d(grad_x, name="grad_x")

    slope = float(np.dot(grad_x, d))
    if slope >= 0:
        raise ValueError(
            f"direction is not a descent direction (<grad, d>={slope:.3e} >= 0)."
        )

    f_x = float(f(x))
    alpha = alpha0
    for _ in range(max_iter):
        if float(f(x + alpha * d)) <= f_x + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha
