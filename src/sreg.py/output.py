#' Print \code{sreg} Objects
import numpy as np
import pandas as pd
from scipy.stats import norm

class Sreg:
    def __init__(self, result):
        self.result = result
    
    def __repr__(self):
        return self.print_sreg()
    
    def print_sreg(self):
        if 'G_id' not in self.result['data'].columns:
            n = len(self.result['data']['Y'])
            tau_hat = self.result['tau_hat']
            se_rob = self.result['se_rob']
            t_stat = self.result['t_stat']
            p_value = self.result['p_value']
            CI_left = self.result['CI_left']
            CI_right = self.result['CI_right']
            lin_adj = self.result['lin_adj']

            if lin_adj is not None:
                print("Saturated Model Estimation Results under CAR with linear adjustments")
            else:
                print("Saturated Model Estimation Results under CAR")

            print(f"Observations: {n}")
            print(f"Number of treatments: {self.result['data']['D'].max()}")
            print(f"Number of strata: {self.result['data']['S'].max()}")
            if lin_adj is not None:
                print(f"Covariates used in linear adjustments: {', '.join(lin_adj.columns)}")
            print("---")
            print("Coefficients:")

            stars = [''] * len(tau_hat)
            for i, p in enumerate(p_value):
                if p <= 0.001:
                    stars[i] = "***"
                elif p <= 0.01:
                    stars[i] = "**"
                elif p <= 0.05:
                    stars[i] = "*"
                elif p <= 0.1:
                    stars[i] = "."

            df = pd.DataFrame({
                "Tau": tau_hat,
                "As.se": se_rob,
                "T-stat": t_stat,
                "P-value": p_value,
                "CI.left(95%)": CI_left,
                "CI.right(95%)": CI_right,
                "Significance": stars
            })

            df = df.round(5)
            print(df.to_string(index=False))
            print("---")
            print("Signif. codes:  0 `***` 0.001 `**` 0.01 `*` 0.05 `.` 0.1 ` ` 1")
        
        else:
            n = len(self.result['data']['Y'])
            G = len(self.result['data']['G_id'].unique())
            tau_hat = self.result['tau_hat']
            se_rob = self.result['se_rob']
            t_stat = self.result['t_stat']
            p_value = self.result['p_value']
            CI_left = self.result['CI_left']
            CI_right = self.result['CI_right']
            lin_adj = self.result['lin_adj']

            if lin_adj is not None:
                print("Saturated Model Estimation Results under CAR with clusters and linear adjustments")
            else:
                print("Saturated Model Estimation Results under CAR with clusters")
            
            print(f"Observations: {n}")
            print(f"Clusters: {G}")
            print(f"Number of treatments: {self.result['data']['D'].max()}")
            print(f"Number of strata: {self.result['data']['S'].max()}")
            if lin_adj is not None:
                print(f"Covariates used in linear adjustments: {', '.join(lin_adj.columns)}")
            print("---")
            print("Coefficients:")

            stars = [''] * len(tau_hat)
            for i, p in enumerate(p_value):
                if p <= 0.001:
                    stars[i] = "***"
                elif p <= 0.01:
                    stars[i] = "**"
                elif p <= 0.05:
                    stars[i] = "*"
                elif p <= 0.1:
                    stars[i] = "."

            df = pd.DataFrame({
                "Tau": tau_hat,
                "As.se": se_rob,
                "T-stat": t_stat,
                "P-value": p_value,
                "CI.left(95%)": CI_left,
                "CI.right(95%)": CI_right,
                "Significance": stars
            })

            df = df.round(5)
            print(df.to_string(index=False))
            print("---")
            print("Signif. codes:  0 `***` 0.001 `**` 0.01 `*` 0.05 `.` 0.1 ` ` 1")
        
        return ""
    
    def __getitem__(self, key):
        return self.result[key]
