import os       # used to check if a file exists on disk
import numpy as np 

def load_csv(filepath, delimiter=",", fill_value=0.0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find the file: '{filepath}'\n"
            f"Please check the path is correct."
        )
    try:
        data = np.genfromtxt(
            fname=filepath,
            delimiter=delimiter,
            filling_values=fill_value,
            dtype=float,
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse the file '{filepath}'.\n"
            f"Make sure it contains only numbers (and optional empty cells).\n"
            f"Original error: {e}"
        )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def load_csv_with_header(filepath, delimiter=",", fill_value=0.0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find the file: '{filepath}'\n"
            f"Please check the path is correct."
        )
    
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    headers = first_line.split(delimiter)

    try:
        data = np.genfromtxt(
            fname=filepath,
            delimiter=delimiter,
            filling_values=fill_value,
            dtype=float,
            skip_header=1,      # skip the first row (the header)
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse the file '{filepath}'.\n"
            f"Original error: {e}"
        )

    # Same single-row fix as in load_csv
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        raise ValueError(f"The file '{filepath}' has no data rows after the header.")

    return headers, data

