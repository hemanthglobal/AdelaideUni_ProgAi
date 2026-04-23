# Author: Musfir Ahmad Nazir (a1990098) — Member 2 (preprocessing.py, utils.py)

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    # Converts any input (list, tuple, etc.) into a 2D NumPy float array.
    # Raises a clear error if the input can't be converted or has wrong shape.
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
    # Makes sure fit() was called before transform().
    # If not, raises a helpful error instead of a confusing crash.
    if not hasattr(transformer, attr):
        raise RuntimeError(
            f"{type(transformer).__name__} is not fitted. Call fit() first."
        )


class StandardScaler:
    # Rescales each column to have mean=0 and std=1.
    # Formula: z = (x - mean) / std
    # Useful when columns have very different scales (e.g. age vs salary).
    # Constant columns (std=0) are handled safely — output is set to 0.
    # NaN values in training data are ignored using np.nanmean / np.nanstd.

    def __init__(self, *, with_mean: bool = True, with_std: bool = True, ddof: int = 0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def fit(self, X: ArrayLike, y=None) -> "StandardScaler":
        # Calculates the mean and std of each column and stores them.
        # Nothing is changed in X yet — this just learns the statistics.
        X = _validate_array(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.nanmean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = np.nanstd(X, axis=0, ddof=self.ddof) if self.with_std else np.ones(X.shape[1])
        # If std is 0 (constant column), replace with 1 to avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        # Applies z = (x - mean) / std using the statistics from fit().
        # Raises an error if fit() hasn't been called or column count changed.
        _check_is_fitted(self)
        X = _validate_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        # Shortcut that runs fit() then transform() in one call.
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt: ArrayLike) -> np.ndarray:
        # Reverses the scaling: recovers original values from scaled ones.
        # Formula: x = z * std + mean
        _check_is_fitted(self)
        Xt = _validate_array(Xt)
        return Xt * self.scale_ + self.mean_

    def __repr__(self) -> str:
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std}, ddof={self.ddof})"


class MinMaxScaler:
    # Scales each column to fit within a target range, default [0, 1].
    # Formula: z = (x - min) / (max - min) * (hi - lo) + lo
    # Useful when a model expects inputs within a fixed range.
    # Constant columns (max == min) are mapped safely to the lower bound.

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}.")
        self.feature_range = feature_range

    def fit(self, X: ArrayLike, y=None) -> "MinMaxScaler":
        # Finds the min and max of each column from training data.
        # Also pre-computes the scale and offset to make transform() fast.
        X = _validate_array(X)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        lo, hi = self.feature_range
        # For constant columns use 1 temporarily to avoid division by zero
        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        full_scale = (hi - lo) / safe_range
        # Constant columns get scale=0 so they always output lo
        self.scale_ = np.where(self.data_range_ == 0, 0.0, full_scale)
        self.min_ = np.where(self.data_range_ == 0, lo, lo - self.data_min_ * full_scale)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        # Applies the min-max formula using values learned in fit().
        # Raises an error if fit() hasn't been called or column count changed.
        _check_is_fitted(self, "data_min_")
        X = _validate_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X * self.scale_ + self.min_

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        # Shortcut that runs fit() then transform() in one call.
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt: ArrayLike) -> np.ndarray:
        # Reverses the scaling to recover the original values.
        _check_is_fitted(self, "data_min_")
        Xt = _validate_array(Xt)
        safe_scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return (Xt - self.min_) / safe_scale

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range})"


class Imputer:
    # Fills in missing values (NaN) in a dataset using a chosen strategy.
    # Real datasets almost always have missing values — most ML models
    # crash or give wrong results when they encounter NaN, so this fixes that.
    #
    # Strategies:
    #   'mean'          — fill with column average (default)
    #   'median'        — fill with column middle value (better with outliers)
    #   'most_frequent' — fill with the most common value in the column
    #   'constant'      — fill with a fixed number you provide

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
        # Computes the fill value for each column based on the chosen strategy.
        # Stores results in self.statistics_ — one value per column.
        X = _validate_array(X)
        self._n_features = X.shape[1]
        ax = self.axis

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=ax)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=ax)
        elif self.strategy == "most_frequent":
            # Find the most common non-NaN value in each column
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
        # Replaces all NaN values using the statistics learned in fit().
        # Uses boolean masking and NumPy indexing — no loop over rows.
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
        # Shortcut that runs fit() then transform() in one call.
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return f"Imputer(strategy='{self.strategy}', fill_value={self.fill_value}, axis={self.axis})"


class OneHotEncoder:
    # Converts categorical text/label columns into binary (0/1) columns.
    # ML models only understand numbers, not strings like "cat" or "red".
    # Each unique category becomes its own column with a 1 where it appears.
    #
    # Example: colour column ["red","blue","red"] becomes:
    #   x0_blue  x0_red
    #     0        1
    #     1        0
    #     0        1
    #
    # handle_unknown controls what happens when transform() sees a new category:
    #   'error'  — raises a ValueError (default, safe choice)
    #   'ignore' — outputs an all-zero row for that sample

    def __init__(self, *, sparse: bool = False, handle_unknown: str = "error", dtype=np.float64):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got '{handle_unknown}'.")
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def fit(self, X: ArrayLike, y=None) -> "OneHotEncoder":
        # Scans each column and stores its sorted unique categories.
        # Also builds human-readable output column names like 'x0_cat', 'x1_red'.
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
        # Builds the one-hot binary matrix.
        # Uses np.searchsorted (binary search) to find each category's position
        # in the sorted list — fast and avoids looping over samples.
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

            # Binary search to find where each value sits in the sorted category list
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

            # Set the matching column to 1 for each row in one vectorised step
            row_idx = np.arange(n_samples)[match_mask]
            block[row_idx, idx_clipped[match_mask]] = 1.0
            blocks.append(block)

        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        # Shortcut that runs fit() then transform() in one call.
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        # Returns the output column names e.g. ['x0_cat', 'x0_dog', 'x1_red'].
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def __repr__(self) -> str:
        return f"OneHotEncoder(handle_unknown='{self.handle_unknown}', sparse={self.sparse})"
