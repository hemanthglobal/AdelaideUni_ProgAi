# Author: Musfir Ahmad Nazir (a1990098) — Member 2 (preprocessing.py, utils.py)

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Convert input to a 2-D float64 ndarray and validate shape.

    Parameters
    ----------
    X    : array-like of shape (n_samples, n_features)
    name : str — variable name used in error messages

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float64

    Raises
    ------
    TypeError  : if X cannot be converted to a numeric array
    ValueError : if X is not 1-D or 2-D
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to a numeric array. Got: {type(X)}"
        ) from exc
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {X.shape}.")
    return X


def _check_is_fitted(transformer, attr: str = "mean_") -> None:
    """Raise RuntimeError if transformer has not been fitted yet.

    Parameters
    ----------
    transformer : object
    attr        : str — attribute set during fit()
    """
    if not hasattr(transformer, attr):
        raise RuntimeError(
            f"{type(transformer).__name__} is not fitted. Call fit() first."
        )


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Formula per feature j:  z = (x - mean_j) / std_j

    Constant columns (std == 0) are set to 0 instead of dividing by zero.
    NaN values in training data are ignored via np.nanmean / np.nanstd.

    Parameters
    ----------
    with_mean : bool, default True  — subtract column mean
    with_std  : bool, default True  — divide by column std
    ddof      : int,  default 0     — degrees of freedom for std

    Attributes
    ----------
    mean_           : ndarray (n_features,) — per-feature mean
    scale_          : ndarray (n_features,) — per-feature std
    n_features_in_  : int — number of features seen during fit
    """

    def __init__(self, *, with_mean: bool = True, with_std: bool = True, ddof: int = 0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def fit(self, X: ArrayLike, y=None) -> "StandardScaler":
        """Compute per-feature mean and std from training data.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Time  : O(n_samples * n_features)
        Space : O(n_features)
        """
        X = _validate_array(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.nanmean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = np.nanstd(X, axis=0, ddof=self.ddof) if self.with_std else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply standardisation using fitted mean and std.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        Xt : ndarray (n_samples, n_features), float64

        Raises
        ------
        RuntimeError : if not fitted
        ValueError   : if feature count does not match fit

        Time  : O(n_samples * n_features)
        Space : O(n_samples * n_features)
        """
        _check_is_fitted(self)
        X = _validate_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        Xt : ndarray (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt: ArrayLike) -> np.ndarray:
        """Reverse the standardisation: X = Xt * std + mean.

        Parameters
        ----------
        Xt : array-like (n_samples, n_features)

        Returns
        -------
        X : ndarray (n_samples, n_features)
        """
        _check_is_fitted(self)
        Xt = _validate_array(Xt)
        return Xt * self.scale_ + self.mean_

    def __repr__(self) -> str:
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std}, ddof={self.ddof})"


class MinMaxScaler:
    """Scale features to a fixed range [lo, hi] (default [0, 1]).

    Formula per feature j:  z = (x - min_j) / (max_j - min_j) * (hi - lo) + lo

    Constant columns (max == min) are mapped to lo without dividing by zero.

    Parameters
    ----------
    feature_range : tuple (lo, hi), default (0, 1)

    Attributes
    ----------
    data_min_      : ndarray (n_features,) — per-feature minimum
    data_max_      : ndarray (n_features,) — per-feature maximum
    data_range_    : ndarray (n_features,) — max - min per feature
    scale_         : ndarray (n_features,) — scaling factor
    min_           : ndarray (n_features,) — offset term
    n_features_in_ : int
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}.")
        self.feature_range = feature_range

    def fit(self, X: ArrayLike, y=None) -> "MinMaxScaler":
        """Compute per-feature min and max from training data.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Time  : O(n_samples * n_features)
        Space : O(n_features)
        """
        X = _validate_array(X)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        lo, hi = self.feature_range
        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        full_scale = (hi - lo) / safe_range
        self.scale_ = np.where(self.data_range_ == 0, 0.0, full_scale)
        self.min_ = np.where(self.data_range_ == 0, lo, lo - self.data_min_ * full_scale)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply min-max scaling.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        Xt : ndarray (n_samples, n_features)

        Raises
        ------
        RuntimeError : if not fitted
        ValueError   : if feature count does not match fit

        Time  : O(n_samples * n_features)
        Space : O(n_samples * n_features)
        """
        _check_is_fitted(self, "data_min_")
        X = _validate_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X * self.scale_ + self.min_

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        Xt : ndarray (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt: ArrayLike) -> np.ndarray:
        """Reverse the min-max scaling.

        Parameters
        ----------
        Xt : array-like (n_samples, n_features)

        Returns
        -------
        X : ndarray (n_samples, n_features)
        """
        _check_is_fitted(self, "data_min_")
        Xt = _validate_array(Xt)
        safe_scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return (Xt - self.min_) / safe_scale

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range})"


class Imputer:
    """Replace NaN values with a per-column statistic.

    Strategies
    ----------
    'mean'          : replace NaN with column mean (default)
    'median'        : replace NaN with column median
    'most_frequent' : replace NaN with most common value in column
    'constant'      : replace NaN with a fixed fill_value

    Parameters
    ----------
    strategy   : str,   default 'mean'
    fill_value : float, default None  — required when strategy='constant'
    axis       : int,   default 0     — 0 = per column, 1 = per row

    Attributes
    ----------
    statistics_ : ndarray (n_features,) — fill value per feature learned during fit
    """

    _VALID_STRATEGIES = {"mean", "median", "most_frequent", "constant"}

    def __init__(self, strategy: str = "mean", fill_value: float | None = None, axis: int = 0):
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {sorted(self._VALID_STRATEGIES)}.")
        if strategy == "constant" and fill_value is None:
            raise ValueError("fill_value must be set when strategy='constant'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.axis = axis

    def fit(self, X: ArrayLike, y=None) -> "Imputer":
        """Compute the fill statistic for each feature.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Time  : O(n_samples * n_features)
        Space : O(n_features)
        """
        X = _validate_array(X)
        self._n_features = X.shape[1]
        ax = self.axis

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=ax)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=ax)
        elif self.strategy == "most_frequent":
            def _mode_col(col: np.ndarray) -> float:
                finite = col[~np.isnan(col)]
                if finite.size == 0:
                    return np.nan
                vals, counts = np.unique(finite, return_counts=True)
                return float(vals[np.argmax(counts)])
            self.statistics_ = np.apply_along_axis(_mode_col, ax, X)
        else:
            n = X.shape[1 - ax]
            self.statistics_ = np.full(n, float(self.fill_value))

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Fill NaN entries using learned statistics.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        Xt : ndarray (n_samples, n_features) with no NaN values

        Raises
        ------
        RuntimeError : if not fitted
        ValueError   : if feature count does not match fit

        Time  : O(n_samples * n_features)
        Space : O(n_samples * n_features)
        """
        _check_is_fitted(self, "statistics_")
        X = _validate_array(X).copy()

        if self.axis == 0:
            if X.shape[1] != self._n_features:
                raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}.")
            nan_mask = np.isnan(X)
            col_idx = np.where(nan_mask)[1]
            X[nan_mask] = self.statistics_[col_idx]
        else:
            nan_mask = np.isnan(X)
            row_idx = np.where(nan_mask)[0]
            X[nan_mask] = self.statistics_[row_idx]

        return X

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        Xt : ndarray (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return f"Imputer(strategy='{self.strategy}', fill_value={self.fill_value}, axis={self.axis})"


class OneHotEncoder:
    """Encode categorical columns as a one-hot binary matrix.

    Each unique category in column j becomes a separate binary output column.
    Output columns are the horizontal concatenation of one-hot blocks per input column.

    Parameters
    ----------
    sparse         : bool, default False       — reserved for future sparse support
    handle_unknown : str,  default 'error'     — 'error' raises on unseen categories;
                                                 'ignore' emits an all-zero row
    dtype          : numpy dtype, default float64

    Attributes
    ----------
    categories_         : list of ndarray — sorted unique categories per column
    n_features_in_      : int
    feature_names_out_  : list[str] — e.g. ['x0_cat', 'x0_dog', 'x1_red']
    """

    def __init__(self, *, sparse: bool = False, handle_unknown: str = "error", dtype=np.float64):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got '{handle_unknown}'.")
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def fit(self, X: ArrayLike, y=None) -> "OneHotEncoder":
        """Learn unique categories per column.

        Parameters
        ----------
        X : array-like (n_samples, n_features) — categorical data
        y : ignored

        Returns
        -------
        self

        Time  : O(n_samples * n_features * log n_samples)
        Space : O(sum of unique categories across all columns)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]

        self.categories_ = []
        for j in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, j]))

        self.feature_names_out_: list[str] = []
        for j, cats in enumerate(self.categories_):
            for c in cats:
                self.feature_names_out_.append(f"x{j}_{c}")

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode X as a one-hot binary array.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        Xt : ndarray (n_samples, total_categories), dtype float64

        Raises
        ------
        RuntimeError : if not fitted
        ValueError   : feature mismatch or unknown category with handle_unknown='error'

        Time  : O(n_samples * n_features * log n_categories)
        Space : O(n_samples * total_categories)
        """
        _check_is_fitted(self, "categories_")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        n_samples = X.shape[0]
        blocks: list[np.ndarray] = []

        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            n_cats = len(cats)
            block = np.zeros((n_samples, n_cats), dtype=self.dtype)

            idx = np.searchsorted(cats, col)
            idx_clipped = np.clip(idx, 0, n_cats - 1)
            match_mask = (cats[idx_clipped] == col)

            if not match_mask.all():
                if self.handle_unknown == "error":
                    bad = col[~match_mask]
                    raise ValueError(
                        f"Column {j}: unknown categories {bad}. "
                        "Pass handle_unknown='ignore' to suppress."
                    )

            row_idx = np.arange(n_samples)[match_mask]
            block[row_idx, idx_clipped[match_mask]] = 1.0
            blocks.append(block)

        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored

        Returns
        -------
        Xt : ndarray (n_samples, total_categories)
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Return output column names e.g. ['x0_cat', 'x0_dog', 'x1_red'].

        Returns
        -------
        names : list[str]
        """
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def __repr__(self) -> str:
        return f"OneHotEncoder(handle_unknown='{self.handle_unknown}', sparse={self.sparse})"
