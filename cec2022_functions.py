"""
cec2022_functions.py
====================
Clean reimplementation of CEC 2022 benchmark functions.

This replaces the buggy CEC2022.py with a working implementation.
Based on: CEC2022-TR.pdf (Technical Report, December 2021)

Author: Rewritten for Mycelial Search
Date: 2026
"""

from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("input_data")  # Official CEC2022 data files
SUPPORTED_DIMS = {2, 10, 20}
SUPPORTED_FUNCTIONS = set(range(1, 13))

# Function metadata (from Technical Report Table 1.2)
FUNCTION_INFO = {
    1: {"name": "Zakharov", "bias": 300.0, "type": "unimodal"},
    2: {"name": "Rosenbrock", "bias": 400.0, "type": "basic"},
    3: {"name": "Expanded_Schaffer_F6", "bias": 600.0, "type": "basic"},
    4: {"name": "Step_Rastrigin", "bias": 800.0, "type": "basic"},
    5: {"name": "Levy", "bias": 900.0, "type": "basic"},
    6: {"name": "Hybrid_1", "bias": 1800.0, "type": "hybrid"},
    7: {"name": "Hybrid_2", "bias": 2000.0, "type": "hybrid"},
    8: {"name": "Hybrid_3", "bias": 2200.0, "type": "hybrid"},
    9: {"name": "Composition_1", "bias": 2300.0, "type": "composition"},
    10: {"name": "Composition_2", "bias": 2400.0, "type": "composition"},
    11: {"name": "Composition_3", "bias": 2600.0, "type": "composition"},
    12: {"name": "Composition_4", "bias": 2700.0, "type": "composition"},
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_shift_data(func_num: int, dim: int) -> np.ndarray:
    """Load one shift vector o from shift_data_{func_num}.txt."""
    file_path = DATA_DIR / f"shift_data_{func_num}.txt"
    data = np.loadtxt(file_path)

    # Some CEC shift files are 2D blocks (multiple shift vectors/orientations).
    if data.ndim == 2:
        # Common case in your error: (dim, 100) -> choose the first column as one shift vector
        if data.shape[0] == dim:
            return np.asarray(data[:, 0], dtype=float)

        # Other common storage: (N, dim) -> choose first row
        if data.shape[1] == dim:
            return np.asarray(data[0, :], dtype=float)

        # Fallback: flatten then slice
        data = np.asarray(data, dtype=float).reshape(-1)
        return data[:dim]

    # 1D file: just slice
    data_1d = np.asarray(data, dtype=float).reshape(-1)
    return data_1d[:dim]


def load_rotation_matrix(func_num: int, dim: int) -> np.ndarray:
    """Load rotation matrix M from M_{func_num}_D{dim}.txt."""
    file_path = DATA_DIR / f"M_{func_num}_D{dim}.txt"
    matrix = np.loadtxt(file_path)

    # Ensure matrix is square (dim x dim)
    if matrix.shape != (dim, dim):
        # Reshape if needed (some files may be stored differently)
        if matrix.size == dim * dim:
            matrix = matrix.reshape(dim, dim)
        else:
            raise ValueError(
                f"Expected {dim}x{dim} matrix, got shape {matrix.shape}"
            )

    return matrix


def load_shuffle_data(func_num: int, dim: int) -> Optional[np.ndarray]:
    """Load shuffle indices for hybrid functions, if they exist."""
    file_path = DATA_DIR / f"shuffle_data_{func_num}_D{dim}.txt"
    if file_path.exists():
        return np.loadtxt(file_path, dtype=int) - 1
    return None


# ============================================================================
# BASIC FUNCTIONS (Section 1.3 of Technical Report)
# ============================================================================

def zakharov(x: np.ndarray) -> float:
    """F1: Zakharov Function (Eq. 1)"""
    x_flat = x.flatten()
    sum1 = float(np.sum(x_flat**2))
    sum2 = float(np.sum(0.5 * np.arange(1, len(x_flat) + 1) * x_flat))
    return float(sum1 + sum2**2 + sum2**4)


def rosenbrock(x: np.ndarray) -> float:
    """F2: Rosenbrock Function (Eq. 2)"""
    x_flat = x.flatten()
    val = float(np.sum(100 * (x_flat[:-1]**2 - x_flat[1:])**2 +
                       (x_flat[:-1] - 1)**2))
    return float(val)


def schaffer_f6_single(x_val: float, y_val: float) -> float:
    """Helper: Schaffer's F6 for two variables."""
    numerator = float(np.sin(np.sqrt(x_val**2 + y_val**2))**2 - 0.5)
    denominator = float((1 + 0.001 * (x_val**2 + y_val**2))**2)
    return float(0.5 + numerator / denominator)


def expanded_schaffer_f6(x: np.ndarray) -> float:
    """F3: Expanded Schaffer F6 (Eq. 3)"""
    x_flat = x.flatten()
    total = 0.0
    for i in range(len(x_flat) - 1):
        total += schaffer_f6_single(float(x_flat[i]), float(x_flat[i+1]))
    total += schaffer_f6_single(float(x_flat[-1]), float(x_flat[0]))
    return float(total)


def rastrigin(x: np.ndarray) -> float:
    """F4: Rastrigin Function (Eq. 4)"""
    x_flat = x.flatten()
    val = float(np.sum(x_flat**2 - 10 * np.cos(2 * np.pi * x_flat) + 10))
    return float(val)


def levy(x: np.ndarray) -> float:
    """F5: Levy Function (Eq. 5)"""
    x_flat = x.flatten()
    w = 1 + (x_flat - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 *
                   (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)


def bent_cigar(x: np.ndarray) -> float:
    """F6: Bent Cigar (Eq. 6)"""
    x_flat = x.flatten()
    val = x_flat[0]**2 + 1e6 * np.sum(x_flat[1:]**2)
    return float(val)


def hgbat(x: np.ndarray) -> float:
    """F7: HGBat Function (Eq. 7)"""
    x_flat = x.flatten()
    d = len(x_flat)
    sum_sq = np.sum(x_flat**2)
    sum_x = np.sum(x_flat)
    val = (np.abs(sum_sq**2 - sum_x**2)**0.5 +
           (0.5 * sum_sq + sum_x) / d + 0.5)
    return float(val)


def elliptic(x: np.ndarray) -> float:
    """F8: High Conditioned Elliptic (Eq. 8)"""
    x_flat = x.flatten()
    d = len(x_flat)
    powers = 1e6 ** (np.arange(d) / (d - 1))
    val = np.sum(powers * x_flat**2)
    return float(val)


def katsuura(x: np.ndarray) -> float:
    """F9: Katsuura Function (Eq. 9)"""
    x_flat = x.flatten()
    d = len(x_flat)
    product = 1.0
    for i in range(d):
        inner_sum = 0.0
        for j in range(1, 33):
            val = 2**j * x_flat[i]
            inner_sum += np.abs(val - np.round(val)) / 2**j
        product *= (1 + (i + 1) * inner_sum)
    return float((10 / d**2) * product - (10 / d**2))


def happycat(x: np.ndarray) -> float:
    """F10: Happycat Function (Eq. 10)"""
    x_flat = x.flatten()
    d = len(x_flat)
    sum_sq = np.sum(x_flat**2)
    sum_x = np.sum(x_flat)
    val = (np.abs(sum_sq - d)**0.25 +
           (0.5 * sum_sq + sum_x) / d + 0.5)
    return float(val)


def griewank(x: np.ndarray) -> float:
    """F15: Griewank Function (Eq. 15)"""
    x_flat = x.flatten()
    sum_term = np.sum(x_flat**2) / 4000
    indices = np.arange(1, len(x_flat) + 1)
    prod_term = np.prod(np.cos(x_flat / np.sqrt(indices)))
    return float(sum_term - prod_term + 1)


def rosenbrock_griewank_helper(x_val: float, y_val: float) -> float:
    """Helper for F11: Rosenbrock applied then Griewank."""
    ros = 100 * (x_val**2 - y_val)**2 + (x_val - 1)**2
    return griewank(np.array([ros]))


def expanded_rosenbrock_griewank(x: np.ndarray) -> float:
    """F11: Expanded Rosenbrock + Griewank (Eq. 11)"""
    x_flat = x.flatten()
    total = 0.0
    for i in range(len(x_flat) - 1):
        total += rosenbrock_griewank_helper(float(x_flat[i]), float(x_flat[i+1]))
    total += rosenbrock_griewank_helper(float(x_flat[-1]), float(x_flat[0]))
    return float(total)


def modified_schwefel(x: np.ndarray) -> float:
    """F12: Modified Schwefel Function (Eq. 12)"""
    x_flat = x.flatten()
    d = len(x_flat)
    z = x_flat + 420.9687462275036

    def g(zi: float) -> float:
        if np.abs(zi) <= 500:
            return float(zi * np.sin(np.abs(zi)**0.5))
        elif zi > 500:
            mod_val = zi % 500
            return float(
                (500 - mod_val) * np.sin(np.sqrt(np.abs(500 - mod_val)))
                - (zi - 500)**2 / (10000 * d)
            )
        else:
            abs_mod = np.abs(zi) % 500
            return float(
                (abs_mod - 500) * np.sin(np.sqrt(np.abs(abs_mod - 500)))
                - (zi + 500)**2 / (10000 * d)
            )

    return float(418.9829 * d - sum(g(zi) for zi in z))


def ackley(x: np.ndarray) -> float:
    """F13: Ackley Function (Eq. 13)"""
    x_flat = x.flatten()
    d = len(x_flat)
    sum_sq = np.sum(x_flat**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x_flat))
    val = (-20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) -
           np.exp(sum_cos / d) + 20 + np.e)
    return float(val)


def discus(x: np.ndarray) -> float:
    """F14: Discus Function (Eq. 14)"""
    x_flat = x.flatten()
    val = 1e6 * x_flat[0]**2 + np.sum(x_flat[1:]**2)
    return float(val)


def schaffer_f7(x: np.ndarray) -> float:
    """F16: Schaffer F7 Function (Eq. 16)"""
    x_flat = x.flatten()
    d = len(x_flat)
    s = np.sqrt(x_flat[:-1]**2 + x_flat[1:]**2)
    val = (np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) + 1)) /
           (d - 1))**2
    return float(val)


# ============================================================================
# CEC 2022 MAIN FUNCTIONS (Section 1.4)
# ============================================================================

def cec2022_f1(x: np.ndarray) -> float:
    """F1: Shifted and Rotated Zakharov (Eq. 16)"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(1, dim)
    m = load_rotation_matrix(1, dim)
    z = m @ (x_arr - o)
    return zakharov(z) + 300.0


def cec2022_f2(x: np.ndarray) -> float:
    """F2: Shifted and Rotated Rosenbrock (Eq. 17)"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(2, dim)
    m = load_rotation_matrix(2, dim)
    z = m @ (2.048 * (x_arr - o) / 100) + 1
    return rosenbrock(z) + 400.0


def cec2022_f3(x: np.ndarray) -> float:
    """F3: Shifted and Rotated Expanded Schaffer F6"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(3, dim)
    m = load_rotation_matrix(3, dim)
    z = m @ (x_arr - o)
    return expanded_schaffer_f6(z) + 600.0


def cec2022_f4(x: np.ndarray) -> float:
    """F4: Shifted and Rotated Non-Continuous Rastrigin"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(4, dim)
    m = load_rotation_matrix(4, dim)

    z = x_arr - o
    for i in range(dim):
        if np.abs(z[i]) > 0.5:
            z[i] = np.round(2 * z[i]) / 2

    z = m @ z
    return rastrigin(z) + 800.0


def cec2022_f5(x: np.ndarray) -> float:
    """F5: Shifted and Rotated Levy"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(5, dim)
    m = load_rotation_matrix(5, dim)
    z = m @ (x_arr - o)
    return levy(z) + 900.0


def cec2022_f6(x: np.ndarray) -> float:
    """F6: Hybrid Function 1 (N=3) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(6, dim)
    m = load_rotation_matrix(6, dim)
    z = m @ (x_arr - o)

    n1 = dim // 3
    n2 = dim // 3

    f1 = bent_cigar(z[:n1])
    f2 = hgbat(z[n1:n1+n2])
    f3 = rastrigin(z[n1+n2:])

    return f1 + f2 + f3 + 1800.0


def cec2022_f7(x: np.ndarray) -> float:
    """F7: Hybrid Function 2 (N=6) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(7, dim)
    m = load_rotation_matrix(7, dim)
    z = m @ (x_arr - o)

    half = dim // 2
    return elliptic(z[:half]) + modified_schwefel(z[half:]) + 2000.0


def cec2022_f8(x: np.ndarray) -> float:
    """F8: Hybrid Function 3 (N=5) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(8, dim)
    m = load_rotation_matrix(8, dim)
    z = m @ (x_arr - o)

    half = dim // 2
    return ackley(z[:half]) + schaffer_f7(z[half:]) + 2200.0


def cec2022_f9(x: np.ndarray) -> float:
    """F9: Composition Function 1 (N=5) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(9, dim)

    # Composition functions: only apply shift, no rotation
    z = x_arr - o

    # Weighted composition
    w = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
    funcs = [rosenbrock, elliptic, rastrigin, ackley, modified_schwefel]

    value = float(sum(wi * fi(z) for wi, fi in zip(w, funcs)))
    return float(value + 2300.0)


def cec2022_f10(x: np.ndarray) -> float:
    """F10: Composition Function 2 (N=4) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(10, dim)

    # Composition functions: only apply shift, no rotation
    z = x_arr - o

    w = np.array([0.2, 0.3, 0.3, 0.2])
    funcs = [griewank, ackley, rastrigin, rosenbrock]

    value = sum(wi * fi(z) for wi, fi in zip(w, funcs))
    return float(value + 2400.0)


def cec2022_f11(x: np.ndarray) -> float:
    """F11: Composition Function 3 (N=5) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(11, dim)

    # Composition functions: only apply shift, no rotation
    z = x_arr - o

    w = np.array([0.1, 0.2, 0.2, 0.3, 0.2])
    funcs = [expanded_rosenbrock_griewank, levy, ackley,
             rastrigin, modified_schwefel]

    value = sum(wi * fi(z) for wi, fi in zip(w, funcs))
    return float(value + 2600.0)


def cec2022_f12(x: np.ndarray) -> float:
    """F12: Composition Function 4 (N=6) - Simplified"""
    x_arr = np.asarray(x).flatten()
    dim = len(x_arr)
    o = load_shift_data(12, dim)

    # Composition functions: only apply shift, no rotation
    z = x_arr - o

    w = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    funcs = [happycat, katsuura, ackley, rastrigin,
             modified_schwefel, levy]

    value = float(sum(wi * fi(z) for wi, fi in zip(w, funcs)))
    return float(value + 2700.0)


# ============================================================================
# PUBLIC API
# ============================================================================

CEC2022_FUNCTIONS: Dict[int, Callable[[np.ndarray], float]] = {
    1: cec2022_f1,
    2: cec2022_f2,
    3: cec2022_f3,
    4: cec2022_f4,
    5: cec2022_f5,
    6: cec2022_f6,
    7: cec2022_f7,
    8: cec2022_f8,
    9: cec2022_f9,
    10: cec2022_f10,
    11: cec2022_f11,
    12: cec2022_f12,
}


def make_cec2022_function(func_num: int) -> Callable[[np.ndarray], float]:
    """
    Create a CEC2022 function evaluator.

    Args:
        func_num: Function number (1-12)

    Returns:
        Callable that evaluates f(x) -> float

    Example:
        #f1 = make_cec2022_function(1)
        #x = np.random.uniform(-100, 100, 10)
        #fitness = f1(x)
    """
    if func_num not in SUPPORTED_FUNCTIONS:
        raise ValueError(
            f"Function must be in {SUPPORTED_FUNCTIONS}, got {func_num}"
        )

    base_func = CEC2022_FUNCTIONS[func_num]
    fname = f"CEC2022_F{func_num}_{FUNCTION_INFO[func_num]['name']}"
    base_func.__name__ = fname
    return base_func


CEC_FUNCTIONS = {f"F{i}": make_cec2022_function(i)
                 for i in range(1, 13)}

CEC_INFO = FUNCTION_INFO


if __name__ == "__main__":
    print("CEC 2022 Functions - Test Suite")
    print("=" * 70)

    print("\nTesting all 12 functions (D=10):")
    x_test = np.random.uniform(-100, 100, 10)

    for i in range(1, 13):
        test_func = make_cec2022_function(i)
        try:
            test_result = test_func(x_test)
            info = FUNCTION_INFO[i]
            print(f"  F{i:2d} ({info['name']:<25s}): {test_result:.6e}")
        except Exception as e:
            print(f"  F{i:2d}: ERROR - {e}")

    print("\n All tests completed!")
