import numpy as np

def rank(X: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Args:
        X      : shape (N,) input array
        method : tie-handling method
                 'average' = tied values get mean rank
                 'dense'   = tied values get same rank, no gaps
                 'ordinal' = tied values get different ranks
    Returns:
        ranks  : shape (N,) rank of each element
    """
    if method == 'ordinal':
        return np.argsort(np.argsort(X)) + 1
    elif method == 'dense':
        # np.unique method returns sorted unique values
        _, inverse = np.unique(X, return_inverse=True)
        return inverse + 1
    elif method == 'average':
        ordinal = np.argsort(np.argsort(X)) + 1.0

        # find ties and avg their ranks
        unique, counts = np.unique(X, return_counts=True)
        for val, count in zip(unique, counts):
            if count > 1:
                mask = X == val
                ordinal[mask] = ordinal[mask].mean()
        return ordinal.astype(float)
    
def percentile(X: np.ndarray, q: float | list, interpolation: str = 'linear') -> float | np.ndarray:
    """
    Args:
        X             : shape (N,) input array
        q             : percentile(s) between 0 and 100
        interpolation : method when percentile falls between values
                        'linear', 'lower', 'higher', 'midpoint', 'nearest'
    Returns:
        scalar float if q is float
        array shape (len(q),) if q is list
    """
    return np.percentile(X, q, method=interpolation)

X = np.array([40, 10, 10, 20])

print(rank(X, method='average'))  # → [4.0, 1.5, 1.5, 3.0]
print(rank(X, method='dense'))    # → [3, 1, 1, 2]
print(rank(X, method='ordinal'))  # → [4, 1, 2, 3]

print(percentile(X, 25))          # → 10.0
print(percentile(X, 50))          # → 15.0
print(percentile(X, 75))          # → 25.0
print(percentile(X, [25, 50, 75]))# → [10.0, 15.0, 25.0]