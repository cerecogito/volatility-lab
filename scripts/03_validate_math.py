"""Step 3: numerical validation of the pricing engine (Track A, simulated / analytic).

Runs the mathematical test-suite first. If any test fails we stop, because no
model may be used for empirical work until its basic tests pass. Then it
measures:
  * CRR binomial convergence to Black-Scholes,
  * Monte Carlo: estimate, SE, CI, runtime and variance-reduction efficiency,
  * Monte Carlo 95% CI coverage over many independent repeats,
  * implied-vol recovery error on a grid of known-vol prices,
  * analytic vs finite-difference Greeks.

Usage:  uv run python scripts/03_validate_math.py
Output: reports/experiments/math_validation.json
"""

import json
import subprocess
import sys
import time

import numpy as np

from volatility_lab.binomial import crr_price
from volatility_lab.black_scholes import bs_delta, bs_gamma, bs_price, bs_theta, bs_vega
from volatility_lab.config import EXPERIMENTS_DIR, PROJECT_ROOT, load_config
from volatility_lab.implied_vol import implied_volatility
from volatility_lab.monte_carlo import mc_price

MATH_TESTS = ["tests/test_black_scholes.py", "tests/test_binomial.py", "tests/test_implied_vol.py",
              "tests/test_monte_carlo.py", "tests/test_bayesian.py", "tests/test_hedging.py"]


def run_math_tests() -> dict:
    # (-q is already set in pyproject.toml, so the last line is the "N passed" summary)
    completed = subprocess.run([sys.executable, "-m", "pytest", *MATH_TESTS],
                               cwd=PROJECT_ROOT, capture_output=True, text=True)
    last_line = completed.stdout.strip().splitlines()[-1]
    print("math tests:", last_line)
    if completed.returncode != 0 or "passed" not in last_line:
        print(completed.stdout)
        raise SystemExit("mathematical tests failed - not continuing")
    return {"command": "pytest " + " ".join(MATH_TESTS), "summary": last_line}


def binomial_convergence() -> list[dict]:
    spot, strike, time_to_expiry, rate, vol = 100.0, 100.0, 0.5, 0.03, 0.3
    exact = {kind: float(bs_price(spot, strike, time_to_expiry, rate, vol, kind)) for kind in ["call", "put"]}
    rows = []
    for steps in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]:
        for kind in ["call", "put"]:
            tree = crr_price(spot, strike, time_to_expiry, rate, vol, steps, kind)
            rows.append({"steps": steps, "option_type": kind, "crr_price": tree, "bs_price": exact[kind],
                         "abs_error": abs(tree - exact[kind])})
    return rows


def monte_carlo_comparison(num_paths: int, seed: int) -> list[dict]:
    args = dict(spot=100.0, strike=105.0, time_to_expiry=0.5, rate=0.03, volatility=0.3)
    exact = float(bs_price(args["spot"], args["strike"], args["time_to_expiry"], args["rate"],
                           args["volatility"], "call"))
    rows = []
    for method in ["plain", "antithetic", "control_variate"]:
        result = mc_price(**args, option_type="call", num_paths=num_paths,
                          rng=np.random.default_rng(seed), method=method)
        rows.append({"method": method, "normal_draws": num_paths, "independent_samples": result.sample_size,
                     "estimate": result.estimate, "standard_error": result.standard_error,
                     "ci_low": result.ci_low, "ci_high": result.ci_high, "bs_price": exact,
                     "error": result.estimate - exact, "runtime_seconds": result.runtime_seconds})
    plain_se = rows[0]["standard_error"]
    for row in rows:
        # variance reduction factor at equal number of normal draws
        row["variance_reduction_factor"] = (plain_se / row["standard_error"]) ** 2
    return rows


def monte_carlo_coverage(repeats: int, num_paths: int, seed: int) -> list[dict]:
    args = dict(spot=100.0, strike=105.0, time_to_expiry=0.5, rate=0.03, volatility=0.3)
    exact = float(bs_price(args["spot"], args["strike"], args["time_to_expiry"], args["rate"],
                           args["volatility"], "call"))
    rows = []
    for method in ["plain", "antithetic", "control_variate"]:
        rng = np.random.default_rng(seed)
        covered, errors, ses = 0, [], []
        for _ in range(repeats):
            result = mc_price(**args, option_type="call", num_paths=num_paths, rng=rng, method=method)
            covered += result.ci_low <= exact <= result.ci_high
            errors.append(result.estimate - exact)
            ses.append(result.standard_error)
        coverage = covered / repeats
        rows.append({"method": method, "repeats": repeats, "normal_draws_per_estimate": num_paths,
                     "coverage_95": coverage,
                     "coverage_binomial_se": float(np.sqrt(0.95 * 0.05 / repeats)),
                     "empirical_sd_of_estimates": float(np.std(errors, ddof=1)),
                     "mean_reported_se": float(np.mean(ses)),
                     "mean_error": float(np.mean(errors))})
    return rows


def implied_vol_recovery() -> dict:
    strikes = np.linspace(50, 150, 21)
    times = [2 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 2.0]
    vols = [0.05, 0.2, 0.5, 1.0, 2.0]
    errors, statuses = [], {}
    started = time.perf_counter()
    for strike in strikes:
        for tau in times:
            for vol in vols:
                for kind in ["call", "put"]:
                    price = float(bs_price(100.0, strike, tau, 0.01, vol, kind))
                    found, status = implied_volatility(price, 100.0, strike, tau, 0.01, kind)
                    statuses[status] = statuses.get(status, 0) + 1
                    if status == "ok":
                        errors.append(abs(found - vol))
    return {"cases": int(sum(statuses.values())), "status_counts": statuses,
            "max_abs_vol_error_when_ok": float(np.max(errors)),
            "median_abs_vol_error_when_ok": float(np.median(errors)),
            "runtime_seconds": time.perf_counter() - started}


def greek_check() -> dict:
    rng = np.random.default_rng(5)
    worst = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for _ in range(500):
        spot, strike = rng.uniform(50, 150), rng.uniform(50, 150)
        tau, rate, vol = rng.uniform(0.05, 2), rng.uniform(0, 0.05), rng.uniform(0.1, 1.0)
        kind = rng.choice(["call", "put"])
        price = lambda s=spot, t=tau, v=vol: float(bs_price(s, strike, t, rate, v, kind))  # noqa: E731
        h = 1e-3 * spot
        numeric = {
            "delta": (price(s=spot + h) - price(s=spot - h)) / (2 * h),
            "gamma": (price(s=spot + h) - 2 * price() + price(s=spot - h)) / h**2,
            "vega": (price(v=vol + 1e-5) - price(v=vol - 1e-5)) / 2e-5,
            "theta": -(price(t=tau + 1e-6) - price(t=tau - 1e-6)) / 2e-6,
        }
        analytic = {
            "delta": float(bs_delta(spot, strike, tau, rate, vol, kind)),
            "gamma": float(bs_gamma(spot, strike, tau, rate, vol)),
            "vega": float(bs_vega(spot, strike, tau, rate, vol)),
            "theta": float(bs_theta(spot, strike, tau, rate, vol, kind)),
        }
        for greek in worst:
            worst[greek] = max(worst[greek], abs(numeric[greek] - analytic[greek]))
    return {"random_cases": 500, "max_abs_difference": worst}


def main() -> None:
    config = load_config()
    simulation = config["simulation"]
    seed = simulation["master_seed"]
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {"experiment_id": config["experiment_id"], "evidence_type": "simulated/analytic",
              "tests": run_math_tests()}
    output["binomial_convergence"] = binomial_convergence()
    output["monte_carlo_comparison"] = monte_carlo_comparison(simulation["mc_validation_paths"], seed)
    output["monte_carlo_coverage"] = monte_carlo_coverage(simulation["mc_coverage_repeats"],
                                                          simulation["mc_coverage_paths"], seed + 1)
    output["implied_vol_recovery"] = implied_vol_recovery()
    output["greek_check"] = greek_check()
    output["seeds"] = {"monte_carlo_comparison": seed, "monte_carlo_coverage": seed + 1, "greek_check": 5}

    (EXPERIMENTS_DIR / "math_validation.json").write_text(json.dumps(output, indent=1))
    for row in output["monte_carlo_comparison"]:
        print(f"{row['method']:16s} est={row['estimate']:.4f} se={row['standard_error']:.5f} "
              f"VRF={row['variance_reduction_factor']:.2f}")
    for row in output["monte_carlo_coverage"]:
        print(f"coverage {row['method']:16s} {row['coverage_95']:.3f}")
    print("IV recovery:", output["implied_vol_recovery"]["status_counts"],
          "max err", output["implied_vol_recovery"]["max_abs_vol_error_when_ok"])
    print("Greeks:", output["greek_check"]["max_abs_difference"])


if __name__ == "__main__":
    main()
