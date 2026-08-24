"""
cec_functions.py
================
Wrapper for CEC 2022 benchmark functions.

It uses a clean rewrite of cec2022_functions.py

This provides a clean API for the Myco benchmarking.
"""

from typing import Callable, Dict
import numpy as np

from cec2022_functions import (
    make_cec2022_function
)


# ============================================================================
# PUBLIC API
# ============================================================================

CEC_METADATA = {
    1: {"name": "Zakharov", "bias": 300.0},
    2: {"name": "Rosenbrock", "bias": 400.0},
    3: {"name": "Schaffer's F7", "bias": 600.0},
    4: {"name": "Non-Continuous Rastrigin", "bias": 800.0},
    5: {"name": "Levy", "bias": 900.0},
    6: {"name": "Hybrid F1", "bias": 1800.0},
    7: {"name": "Hybrid F2", "bias": 2000.0},
    8: {"name": "Hybrid F3", "bias": 2200.0},
    9: {"name": "Composition F1", "bias": 2300.0},
    10: {"name": "Composition F2", "bias": 2400.0},
    11: {"name": "Composition F3", "bias": 2600.0},
    12: {"name": "Composition F4", "bias": 2700.0},
}

SUPPORTED_FUNCNUMS = set(range(1, 13))


def make_cec_function(func_num: int) -> Callable[[np.ndarray], float]:
    """
    Create a callable f(x) -> float for the CEC 2022 function func_num.

    Args:
        func_num: Function ID (1-12)

    Returns:
        Function that takes a 1D array x and returns a scalar fitness

    Example:
        # f5 = make_cec_function(5)  # Levy function
        # x = np.random.uniform(-100, 100, 10)
        # fitness = f5(x)
    """
    if func_num not in SUPPORTED_FUNCNUMS:
        raise ValueError(
            f"func_num must be in {SUPPORTED_FUNCNUMS}, got {func_num}"
        )

    return make_cec2022_function(func_num)


CEC_FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
    f"F{i}": make_cec_function(i) for i in range(1, 13)
}

CEC_CORE: Dict[str, Callable[[np.ndarray], float]] = {
    "F1": CEC_FUNCTIONS["F1"],
    "F2": CEC_FUNCTIONS["F2"],
    "F4": CEC_FUNCTIONS["F4"],
    "F6": CEC_FUNCTIONS["F6"],
    "F9": CEC_FUNCTIONS["F9"],
}

CEC_EXTRA: Dict[str, Callable[[np.ndarray], float]] = {
    "F3": CEC_FUNCTIONS["F3"],   # Expanded Schaffer F6
    "F5": CEC_FUNCTIONS["F5"],   # Levy
    "F7": CEC_FUNCTIONS["F7"],   # Hybrid Function 2
    "F8": CEC_FUNCTIONS["F8"],   # Hybrid Function 3
    "F10": CEC_FUNCTIONS["F10"], # Composition Function 2
    "F11": CEC_FUNCTIONS["F11"], # Composition Function 3
    "F12": CEC_FUNCTIONS["F12"], # Composition Function 4
}

CEC_INFO: Dict[str, Dict] = {
    f"F{i}": meta for i, meta in CEC_METADATA.items()
}


def wrap_for_myco(
        cec_func: Callable[[np.ndarray], float]
) -> Callable[[np.ndarray], float]:
    """
    Ensure the CEC function works with the Myco optimiser.

    The MycelialSearch optimiser expects: objfunc(x) -> scalar
    This wrapper ensures correct shape handling.
    """
    def wrapped(position: np.ndarray) -> float:
        position_flat = np.asarray(position, dtype=float).flatten()
        return cec_func(position_flat)

    return wrapped


CEC_FOR_Myco: Dict[str, Callable[[np.ndarray], float]] = {
    name: wrap_for_myco(func) for name, func in CEC_FUNCTIONS.items()
}


if __name__ == "__main__":
    print("CEC 2022 Benchmark Wrapper Test")
    print("=" * 60)

    print("\n1. Single-Point Evaluation Test:")
    x_test = np.random.uniform(-100, 100, 10)

    for name in ["F1", "F2", "F4", "F6", "F9"]:
        func = CEC_CORE[name]
        val = func(x_test)
        func_id = int(name[1:])
        info = CEC_METADATA[func_id]
        print(f"   {name} ({info['name']:<20s}): {val:.6e}")

    print("\n2. Multi-Dimension Support Test:")
    for dim in [2, 10, 20]:
        pos = np.random.uniform(-100, 100, dim)
        f1 = CEC_FUNCTIONS["F1"]
        result = f1(pos)
        print(f"   D={dim:2d}: F1(x) = {result:.6e}")

    print("\n3. All Functions Test (D=10):")
    x_test = np.random.uniform(-100, 100, 10)
    for i in range(1, 13):
        func = CEC_FUNCTIONS[f"F{i}"]
        val = func(x_test)
        info = CEC_METADATA[i]
        print(f"   F{i:2d} ({info['name']:<20s}): {val:.6e}")

    print("\n All tests passed!")
