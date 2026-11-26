"""Select grid sizes before sampling a geometry; never resize existing labels."""

from numbers import Integral

GRID_POLICIES = ("auto", "even", "odd", "even-13", "odd-13")
FFT_PRIMES = (2, 3, 5, 7, 11, 13)


def is_fft_size(n):
    """Return whether a positive integer has no prime factor larger than 13."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        return False
    for prime in FFT_PRIMES:
        while n % prime == 0:
            n //= prime
    return n == 1


def select_grid(n, policy="auto", *, discretization="gani"):
    """Return the smallest allowed grid edge greater than or equal to ``n``.

    ``auto`` chooses ``odd-13`` for GaNi and ``even-13`` for HEX8. ``odd`` and
    ``even`` constrain parity only; their ``-13`` variants also limit prime
    factors to 13, as required by the generated CUDA FFTs. Device memory can
    still require the solver's library FFT path. This function does not
    initialize CUDA, allocate a geometry or modify an existing discretization.

    Examples
    --------
    >>> select_grid(32)
    33
    >>> select_grid(17, "odd")
    17
    >>> select_grid(17, "odd-13")
    21
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 4:
        raise ValueError("Grid edge must be an integer >= 4")
    if policy not in GRID_POLICIES:
        raise ValueError(f"Unknown grid policy {policy!r}; choose from {GRID_POLICIES}")
    if discretization not in ("gani", "hex8"):
        raise ValueError("discretization must be 'gani' or 'hex8'")
    if policy == "auto":
        policy = "odd-13" if discretization == "gani" else "even-13"
    parity = int(policy.startswith("odd"))
    n = int(n) + (int(n) % 2 != parity)
    while policy.endswith("-13") and not is_fft_size(n):
        n += 2
    return n
