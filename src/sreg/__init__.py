# sreg.py/__init__.py
"""
Sreg:Stratified Randomized Experiments
============
The sreg package offers a toolkit for estimating average treatment effects (ATEs) in stratified randomized experiments. 
The package is designed to accommodate scenarios with multiple treatments and cluster-level treatment assignments, 
and accomodates optimal linear covariate adjustment based on baseline observable characteristics. The package 
computes estimators and standard errors based on Bugni, Canay, Shaikh (2018); Bugni, Canay, Shaikh, Tabord-Meehan (2023); 
and Jiang, Linton, Tang, Zhang (2023).
Authors:
- Juri Trifonov <jutrifonov@uchicago.edu>
- Yuehao Bai <yuehao.bai@usc.edu>
- Azeem Shaikh <amshaikh@uchicago.edu>
- Max Tabord-Meehan <maxtm@uchicago.edu>

Maintainer:
- Juri Trifonov <jutrifonov@uchicago.edu>

References:
- Bugni, F. A., Canay, I. A., and Shaikh, A. M. (2018). Inference Under Covariate-Adaptive Randomization. \emph{Journal of the American Statistical Association}, 113(524), 1784–1796, \doi{10.1080/01621459.2017.1375934}.
- Bugni, F., Canay, I., Shaikh, A., and Tabord-Meehan, M. (2024+). Inference for Cluster Randomized Experiments with Non-ignorable Cluster Sizes. \emph{Forthcoming in the Journal of Political Economy: Microeconomics}, \doi{10.48550/arXiv.2204.08356}.
- Jiang, L., Linton, O. B., Tang, H., and Zhang, Y. (2023+). Improving Estimation Efficiency via Regression-Adjustment in Covariate-Adaptive Randomizations with Imperfect Compliance. \emph{Forthcoming in Review of Economics and Statistics}, \doi{10.48550/arXiv.2204.08356}.
"""

print("sreg package is being imported")

# Import only the public functions and classes
from .core import sreg, sreg_rgen
from .output import Sreg

__all__ = ["sreg", "sreg_rgen", "Sreg"]

# Expose sreg_rgen and sreg directly in the sreg namespace
sreg_rgen = sreg_rgen
sreg = sreg
Sreg = Sreg
