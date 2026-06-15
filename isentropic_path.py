#
# https://github.com/BaseMax/research-fuzzy-graph-path
#

import math
import sys
import os
from typing import List, Tuple, Optional


LN2 = math.log(2.0)


def pause_if_windows():
    if os.name == "nt":
        try:
            if sys.stdin.isatty():
                input("\nExecution complete. Press Enter to exit...")
        except Exception:
            pass

def f(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0

    return -(x * math.log(x) + (1.0 - x) * math.log(1.0 - x))


def f_inverse_lower(y: float,
                    tol: float = 1e-14,
                    max_iter: int = 200) -> Optional[float]:
    if y < 0.0:
        return None
    if y > LN2 + 1e-12:
        return None

    if y >= LN2:
        return 0.5
    if y == 0.0:
        return 0.0

    lo, hi = 1e-18, 0.5

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)

        if abs(fm - y) < tol:
            return mid

        if fm < y:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def construct_isentropic_path(
    A: List[Tuple[str, float]],
    branch: str = "lower",
    verbose: bool = True,
) -> List[Tuple[str, float]]:
    n = len(A)
    if n < 2:
        raise ValueError("Need at least 2 vertices to form a path.")
    for v, a in A:
        if not (0.0 < a < 1.0):
            raise ValueError(f"a_{v} = {a} must lie strictly in (0, 1).")
    if branch not in ("lower", "upper"):
        raise ValueError("branch must be either 'lower' or 'upper'.")

    if verbose:
        print("=" * 64)
        print("  Algorithm: Isentropic Fuzzy Graph Path  P_n = (A, B)")
        print("=" * 64)
        print(f"  n = {n} vertices,  branch = '{branch}'")
        print(f"  ln 2 = {LN2:.15f}   (maximum value of f)")
        print()
        print("  Step 0: vertex entropies  f(a_i)")
        print("  " + "-" * 60)

    fa: List[float] = []
    for v, a in A:
        val = f(a)
        fa.append(val)
        if verbose:
            print(f"    f(a_{v}) = -[{a}*ln({a}) + {1-a:.6f}*ln({1-a:.6f})] = {val:.12f}")

    En_A = sum(fa)
    if verbose:
        print()
        print(f"  En(A) = sum_i f(a_i) = {En_A:.12f}")
        print()

    B: List[Tuple[str, float]] = []
    En_B = 0.0

    for i in range(n - 1):
        (v1, a1), (v2, a2) = A[i], A[i + 1]

        total = fa[i] + fa[i + 1]

        if verbose:
            print("  " + "-" * 60)
            print(f"  Step {i+1}: edge  e_{i+1} = {v1}{v2}")
            print(f"    s_i target =  f(a_{v1}) + f(a_{v2})")
            print(f"               =  {fa[i]:.12f} + {fa[i+1]:.12f}")
            print(f"               =  {total:.12f}")

        s_low = f_inverse_lower(total)

        if s_low is None:
            if verbose:
                print(f"    !! f(a_{v1}) + f(a_{v2}) = {total:.6f} > ln 2"
                      f" = {LN2:.6f}")
                print(f"    !! No real s_i in (0,1) exists.")
                print(f"    !! (No isentropically-valid b_{i+1} for this pair.)")
            B.append((f"{v1}{v2}", float("nan")))
            continue

        s_up = 1.0 - s_low

        b_i = s_low if branch == "lower" else s_up

        bound = min(a1, a2)
        valid_fuzzy = b_i <= bound + 1e-12

        if verbose:
            print(f"    Bisection in (0, 0.5]  ->  s_i        = {s_low:.12f}")
            print(f"    Symmetric partner       1 - s_i     = {s_up:.12f}")
            print(f"    Check  f(s_i)        = {f(s_low):.12f}  (target {total:.12f})")
            print(f"    Check  f(1 - s_i)    = {f(s_up):.12f}")
            print(f"    Chosen (branch={branch}):  b_{i+1} = {b_i:.12f}")
            print(f"    Fuzzy-graph bound  min(a_{v1}, a_{v2}) = {bound}")
            ok = "OK" if valid_fuzzy else "VIOLATES  b_i <= min(a_i, a_{i+1})"
            print(f"    b_{i+1} <= min(a_i, a_{{i+1}})  ?  {ok}")

        if not valid_fuzzy:
            if verbose:
                print(f"    !! b_{i+1} = {b_i:.12f} > min(a_{v1}, a_{v2}) = {bound}")
                print(f"    !! No valid b_{i+1} exists for this pair (fuzzy-graph bound violated).")
            B.append((f"{v1}{v2}", float("nan")))
            continue

        En_B += f(b_i)
        B.append((f"{v1}{v2}", b_i))

    if verbose:
        print()
        print("  " + "=" * 60)
        print("  Result")
        print("  " + "-" * 60)
        print(f"    En(A) = {En_A:.12f}")
        print(f"    En(B) = {En_B:.12f}")
        print(f"    |En(A) - En(B)|  =  {abs(En_A - En_B):.3e}")
        print()
        print("    B = {")
        for e, b in B:
            if math.isnan(b):
                print(f"          ( {e},  b = NO SOLUTION )")
            else:
                print(f"          ( {e},  b = {b:.12f} )")
        print("        }")
        print("  " + "=" * 60)

    return B


def read_A_interactive() -> List[Tuple[str, float]]:
    while True:
        try:
            n = int(input("Number of vertices  n (>= 2): ").strip())
            if n >= 2:
                break
        except ValueError:
            pass
        print("  Please enter an integer >= 2.")

    A: List[Tuple[str, float]] = []
    print(f"Enter the {n} membership values  a_i  in (0,1):")
    for i in range(1, n + 1):
        while True:
            try:
                raw = input(f"  a_{i} = ").strip()
                a = float(raw)
                if 0.0 < a < 1.0:
                    A.append((f"v{i}", a))
                    break
            except ValueError:
                pass
            print("    Value must be a real number strictly between 0 and 1.")
    return A


def ask_branch() -> str:
    raw = input(
        "Choose branch for b_i  -- 'l' = lower (in (0,0.5]),  "
        "'u' = upper (in [0.5,1)) [default l]: "
    ).strip().lower()
    return "upper" if raw.startswith("u") else "lower"


def main() -> None:
    print("Isentropic fuzzy graph path  --  Algorithm implementation")

    A = read_A_interactive()
    branch = ask_branch()
    construct_isentropic_path(A, branch=branch, verbose=True)
    
    pause_if_windows()

if __name__ == "__main__":
    main()
