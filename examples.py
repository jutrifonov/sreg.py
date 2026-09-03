"""Runnable examples for sreg 2.1.0.

From the repository directory, run:

    python examples.py

To run every example and then the complete test suite:

    python examples.py --run-tests

To run only the tests yourself:

    python -m pytest -q
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sreg import AEJapp, sreg, sreg_rgen


REPOSITORY = Path(__file__).resolve().parent


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show_result(name: str, fit) -> None:
    print(f"\n{name}")
    print(fit)


def large_strata_examples() -> list:
    """Large-strata examples with every cluster/covariate combination."""
    heading("1. Large-strata simulated examples")
    fits = []

    individual = sreg_rgen(
        n=600,
        tau_vec=(0.2, 0.8),
        n_strata=5,
        cluster=False,
        is_cov=True,
        random_state=101,
    )
    fit = sreg(individual.Y, individual.S, individual.D)
    show_result("Individual assignment, without covariates", fit)
    fits.append(fit)

    fit = sreg(
        individual.Y,
        individual.S,
        individual.D,
        X=individual[["x_1", "x_2"]],
    )
    show_result("Individual assignment, with covariates", fit)
    fits.append(fit)

    clustered = sreg_rgen(
        n=150,
        tau_vec=(0.2, 0.8),
        n_strata=5,
        cluster=True,
        is_cov=True,
        random_state=102,
    )
    fit = sreg(
        clustered.Y,
        clustered.S,
        clustered.D,
        G_id=clustered.G_id,
        Ng=clustered.Ng,
    )
    show_result("Cluster assignment, without covariates", fit)
    fits.append(fit)

    fit = sreg(
        clustered.Y,
        clustered.S,
        clustered.D,
        G_id=clustered.G_id,
        Ng=clustered.Ng,
        X=clustered[["x_1", "x_2"]],
    )
    show_result("Cluster assignment, with covariates", fit)
    fits.append(fit)
    return fits


def small_strata_examples() -> list:
    """Triplet-strata examples with every cluster/covariate combination."""
    heading("2. Small-strata simulated examples")
    fits = []

    individual = sreg_rgen(
        n=300,
        tau_vec=(0.2, 0.8),
        cluster=False,
        is_cov=True,
        small_strata=True,
        k=3,
        treat_sizes=(1, 1, 1),
        random_state=201,
    )
    fit = sreg(
        individual.Y,
        individual.S,
        individual.D,
        small_strata=True,
        k=3,
    )
    show_result("Individual assignment, without covariates", fit)
    fits.append(fit)

    fit = sreg(
        individual.Y,
        individual.S,
        individual.D,
        X=individual[["x_1", "x_2"]],
        small_strata=True,
        k=3,
    )
    show_result("Individual assignment, with covariates", fit)
    fits.append(fit)

    clustered = sreg_rgen(
        n=150,
        tau_vec=(0.2, 0.8),
        cluster=True,
        is_cov=True,
        small_strata=True,
        k=3,
        treat_sizes=(1, 1, 1),
        random_state=202,
    )
    fit = sreg(
        clustered.Y,
        clustered.S,
        clustered.D,
        G_id=clustered.G_id,
        Ng=clustered.Ng,
        small_strata=True,
        k=3,
    )
    show_result("Cluster assignment, without covariates", fit)
    fits.append(fit)

    fit = sreg(
        clustered.Y,
        clustered.S,
        clustered.D,
        G_id=clustered.G_id,
        Ng=clustered.Ng,
        X=clustered[["x_1", "x_2"]],
        small_strata=True,
        k=3,
    )
    show_result("Cluster assignment, with covariates", fit)
    fits.append(fit)
    return fits


def aejapp_examples() -> list:
    """Reproduce the package's AEJ application (individual, large strata)."""
    heading("3. AEJapp empirical application")
    data = AEJapp().copy()
    treatment = data.treatment.replace(3, 0)

    unadjusted = sreg(data.gradesq34, data.class_level, treatment)
    show_result("AEJapp without covariates", unadjusted)

    adjusted = sreg(
        data.gradesq34,
        data.class_level,
        treatment,
        X=data[["pills_taken", "age_months"]],
    )
    show_result("AEJapp with covariates", adjusted)
    return [unadjusted, adjusted]


def plot_examples(fits: list, *, show: bool) -> None:
    """Use the result object's plot method and save an example image."""
    heading("4. Plotting")
    fit = fits[-1]
    ax = fit.plot(
        treatment_labels=["Treatment 1", "Treatment 2"],
        title="AEJapp treatment effects",
        x_axis_title="ATE relative to control",
        bar_fill=("#3B82F6", "#14B8A6"),
        point_fill="white",
    )
    output = REPOSITORY / "sreg_example_plot.png"
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output}")
    if show:
        plt.show()
    else:
        plt.close(ax.figure)


def run_tests() -> None:
    heading("5. Complete pytest suite")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the complete pytest suite after the examples.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the example plot without opening an interactive window.",
    )
    args = parser.parse_args()

    fits = large_strata_examples()
    fits.extend(small_strata_examples())
    fits.extend(aejapp_examples())
    plot_examples(fits, show=not args.no_show)

    print("\nTo run all tests manually:")
    print(f'  cd "{REPOSITORY}"')
    print("  python -m pytest -q")
    if args.run_tests:
        run_tests()


if __name__ == "__main__":
    main()
