import numpy as np

def mean(X: np.ndarray, axis: int = None):
    # compute the mean of array along the specified axis
    # X = input array
    # axis along which the mean is computed 
    # 0=column wise, 1=row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """

    return np.nanmean(X, axis=axis)

def median(X: np.ndarray, axis: int = None):
    # compute the median of array along the specified axis
    # X = input array
    # axis along which the median is computed
    #  0 = column wise, 1 = row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """
    
    return np.nanmedian(X, axis=axis)

def standard_deviation(X: np.ndarray, axis: int = None):
    # compute the standard deviation of array along the specified axis
    # X = input array
    # axis along which the standard deviation is computed
    #  0 = colums wise, 1 = row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """

    return np.nanstd(X, axis=axis)

def minimum(X: np.ndarray, axis: int = None):
    # compute the min of array along the specified axis
    #  X= input array
    # axis along which the min is computed
    # 0 = column wise, 1 = row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """

    return np.nanmin(X, axis=axis)

def maximum(X: np.ndarray, axis: int = None):
    # compute the max of array along the specified axis
    #  X= input array
    # axis along which the max is computed
    #  0 = column wise, 1 = row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """

    return np.nanmax(X, axis=axis)

def histogram(X: np.ndarray, bins: int | str = 'auto'):
    #  compute bins and counts for histogram
    # x = input array
    # bins are intercval used to group data

    # counts are array of each bin count
    # edges are array of bin boundaries

    counts, edges = np.histogram(X, bins=bins)
    return counts, edges

def quantile(X: np.ndarray,q, axis: int = None):
    # compute the q-th quantile of array along the specified axis
    # X = input array
    # q = quantile to compute, which must be between 0 and 1 inclusive
    # axis along which the quantiles are computed
    # 0 = column wise, 1 = row wise computation

    """
        Args:
                X    : shape (N,) or (N, M)
                axis : 0 = column-wise, 1 = row-wise, None = global
            Returns:
                shape () if axis=None
                shape (M,) if axis=0
                shape (N,) if axis=1    
    """


    return np.nanpercentile(X, q=q*100, axis=axis)

# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 100]])
# print(f"Counts and edges for histogram : {histogram(X.flatten())}")
# print(mean(X , axis=1))
# print(quantile(X, q=0.25, axis=1))