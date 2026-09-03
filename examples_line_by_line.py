"""Console-friendly sreg examples.

This file deliberately contains no function definitions. In VS Code, put the
cursor on a statement (or select one complete statement) and press Shift+Enter.
Run the file from top to bottom so variables exist before they are used.
"""

# 0. Imports -- run these first
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sreg import AEJapp, sreg, sreg_rgen


# 1A. Large strata: individual assignment, without covariates
large_individual = sreg_rgen(n=600, tau_vec=(0.2, 0.8), n_strata=5, cluster=False, is_cov=True, random_state=101)
large_individual_unadjusted = sreg(large_individual.Y, large_individual.S, large_individual.D)
print(large_individual_unadjusted)


# 1B. Large strata: individual assignment, with covariates
large_individual_adjusted = sreg(large_individual.Y, large_individual.S, large_individual.D, X=large_individual[["x_1", "x_2"]])
print(large_individual_adjusted)


# 1C. Large strata: cluster assignment, without covariates
large_cluster = sreg_rgen(n=150, tau_vec=(0.2, 0.8), n_strata=5, cluster=True, is_cov=True, random_state=102)
large_cluster_unadjusted = sreg(large_cluster.Y, large_cluster.S, large_cluster.D, G_id=large_cluster.G_id, Ng=large_cluster.Ng)
print(large_cluster_unadjusted)


# 1D. Large strata: cluster assignment, with covariates
large_cluster_adjusted = sreg(large_cluster.Y, large_cluster.S, large_cluster.D, G_id=large_cluster.G_id, Ng=large_cluster.Ng, X=large_cluster[["x_1", "x_2"]])
print(large_cluster_adjusted)


# 2A. Small strata: individual assignment, without covariates
small_individual = sreg_rgen(n=300, tau_vec=(0.2, 0.8), cluster=False, is_cov=True, small_strata=True, k=3, treat_sizes=(1, 1, 1), random_state=201)
small_individual_unadjusted = sreg(small_individual.Y, small_individual.S, small_individual.D, small_strata=True, k=3)
print(small_individual_unadjusted)


# 2B. Small strata: individual assignment, with covariates
small_individual_adjusted = sreg(small_individual.Y, small_individual.S, small_individual.D, X=small_individual[["x_1", "x_2"]], small_strata=True, k=3)
print(small_individual_adjusted)


# 2C. Small strata: cluster assignment, without covariates
small_cluster = sreg_rgen(n=150, tau_vec=(0.2, 0.8), cluster=True, is_cov=True, small_strata=True, k=3, treat_sizes=(1, 1, 1), random_state=202)
small_cluster_unadjusted = sreg(small_cluster.Y, small_cluster.S, small_cluster.D, G_id=small_cluster.G_id, Ng=small_cluster.Ng, small_strata=True, k=3)
print(small_cluster_unadjusted)


# 2D. Small strata: cluster assignment, with covariates
small_cluster_adjusted = sreg(small_cluster.Y, small_cluster.S, small_cluster.D, G_id=small_cluster.G_id, Ng=small_cluster.Ng, X=small_cluster[["x_1", "x_2"]], small_strata=True, k=3)
print(small_cluster_adjusted)


# 3A. AEJapp: reproduce its individual-assignment, large-strata design
aej = AEJapp().copy()
aej_treatment = aej.treatment.replace(3, 0)
aej_unadjusted = sreg(aej.gradesq34, aej.class_level, aej_treatment)
print(aej_unadjusted)


# 3B. AEJapp with covariate adjustment
aej_adjusted = sreg(aej.gradesq34, aej.class_level, aej_treatment, X=aej[["pills_taken", "age_months"]])
print(aej_adjusted)


# 4. Plot the AEJapp estimates
ax = aej_adjusted.plot(treatment_labels=["Treatment 1", "Treatment 2"], title="AEJapp treatment effects", x_axis_title="ATE relative to control", bar_fill=("#3B82F6", "#14B8A6"), point_fill="white")
ax.figure.tight_layout()
plt.show()


# 5. Run all tests (145 should pass)
test_result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=Path.cwd(), check=False)
print("pytest exit code:", test_result.returncode)
