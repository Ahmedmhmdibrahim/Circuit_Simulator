import numpy as np
from typing import List


def Solve_real_Linear_Matrix(Y: np.ndarray, J: np.ndarray) -> List:

    return np.linalg.solve(Y, J)

def Solve_complex_Linear_Matrix(Y: np.ndarray, J: np.ndarray) -> List:

    return np.abs(np.linalg.solve(Y, J))
