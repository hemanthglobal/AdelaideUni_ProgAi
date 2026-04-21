import numpy as np

# confusion matrix
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Args:
        y_true: shape (N,) actual values
        y_pred: shape (N,) predicted values
    Returns:
        matrix: shape(2,2)
        [[TN, FP],
         [FN, TP]]
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP],
                     [FN, TP]])

# accuracy
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true: shape (N,) actual values
        y_pred: shape (N,) predicted values
    Returns:
        scalar float - accuracy score between 0 and 1
    """
    return np.sum(y_true == y_pred) / len(y_true)

# precision
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true: shape (N,) actual values
        y_pred: shape (N,) predicted values
    Returns:
        scalar float - precision score between 0 and 1
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

# recall
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true: shape (N,) actual values
        y_pred: shape (N,) predicted values
    Returns:
        scalar float - recall score between 0 and 1
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

# f1 score
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true: shape (N,)
        y_pred: shape (N,)
    Returns:
        scalar float - f1 score between 0 and 1
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

# mse
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Args:
        y_true: shape (N,) actual values
        y_pred: shape (N,) predicted values
    Returns:
        scalar float - mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


# testing
# classification
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 1, 1, 0, 1])

print(confusion_matrix(y_true, y_pred))  # → [[TN=2, FP=2], [FN=1, TP=3]]
print(accuracy(y_true, y_pred))          # → 0.625
print(precision(y_true, y_pred))         # → 0.6
print(recall(y_true, y_pred))            # → 0.75
print(f1_score(y_true, y_pred))          # → 0.666

# regression
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.1, 2.9, 6.5])

print(mse(y_true, y_pred))               # → 0.115