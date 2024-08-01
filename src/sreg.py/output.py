#' Print \code{sreg} Objects
import numpy as np
import pandas as pd
from scipy.stats import norm

class Sreg:
    def __init__(self, res_dict):
        self.__dict__.update(res_dict)

    def __repr__(self):
        return self.print_sreg()

    def print_sreg(self):
        output = []
        if 'G_id' not in self.data.columns:
            n = len(self.data['Y'])
            tau_hat = np.array(self.tau_hat)
            se_rob = np.array(self.se_rob)
            t_stat = np.array(self.t_stat)
            p_value = np.array(self.p_value)
            CI_left = np.array(self.CI_left)
            CI_right = np.array(self.CI_right)

            if self.lin_adj is not None:
                output.append("Saturated Model Estimation Results under CAR with linear adjustments\n")
                covariates_str = ', '.join(self.lin_adj.columns)
            else:
                output.append("Saturated Model Estimation Results under CAR\n")
                covariates_str = 'None'

            output.append(f"Observations: {n}\n")
            output.append(f"Number of treatments: {self.data['D'].max()}\n")
            output.append(f"Number of strata: {self.data['S'].max()}\n")
            if self.lin_adj is not None:
                output.append(f"Covariates used in linear adjustments: {', '.join(self.lin_adj.columns)}\n")
            else:
                output.append("Covariates used in linear adjustments:\n")
            output.append("---\n")
            output.append("Coefficients:\n")

            m = len(tau_hat)
            stars = np.array([""] * m, dtype=object)
            stars[p_value <= 0.001] = "***"
            stars[(p_value > 0.001) & (p_value <= 0.01)] = "**"
            stars[(p_value > 0.01) & (p_value <= 0.05)] = "*"
            stars[(p_value > 0.05) & (p_value <= 0.1)] = "."

            df = pd.DataFrame({
                "Tau": tau_hat,
                "As.se": se_rob,
                "T-stat": t_stat,
                "P-value": p_value,
                "CI.left(95%)": CI_left,
                "CI.right(95%)": CI_right,
                "Significance": stars
            })

            is_df_num_col = df.apply(lambda col: pd.api.types.is_numeric_dtype(col))
            df.loc[:, is_df_num_col] = df.loc[:, is_df_num_col].round(5)
            output.append(df.to_string(index=False))
            output.append("\n---\n")
            output.append("Signif. codes:  0 `***` 0.001 `**` 0.01 `*` 0.05 `.` 0.1 ` ` 1\n")

            if self.lin_adj is not None:
                if any([np.isnan(x).any() for x in self.ols_iter]):
                    raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
                
        else:
            n = len(self.data['Y'])
            G = len(self.data['G_id'].unique())
            tau_hat = np.array(self.tau_hat)
            se_rob = np.array(self.se_rob)
            t_stat = np.array(self.t_stat)
            p_value = np.array(self.p_value)
            CI_left = np.array(self.CI_left)
            CI_right = np.array(self.CI_right)

            if self.lin_adj is not None:
                output.append("Saturated Model Estimation Results under CAR with clusters and linear adjustments\n")
                covariates_str = ', '.join(self.lin_adj.columns)
            else:
                output.append("Saturated Model Estimation Results under CAR with clusters\n")
                covariates_str = 'None'


            output.append(f"Observations: {n}\n")
            output.append(f"Clusters: {G}\n")
            output.append(f"Number of treatments: {self.data['D'].max()}\n")
            output.append(f"Number of strata: {self.data['S'].max()}\n")
            if self.lin_adj is not None:
                output.append(f"Covariates used in linear adjustments: {', '.join(self.lin_adj.columns)}\n")
            else:
                output.append("Covariates used in linear adjustments:\n")
            output.append("---\n")
            output.append("Coefficients:\n")

            m = len(tau_hat)
            stars = np.array([""] * m, dtype=object)
            stars[p_value <= 0.001] = "***"
            stars[(p_value > 0.001) & (p_value <= 0.01)] = "**"
            stars[(p_value > 0.01) & (p_value <= 0.05)] = "*"
            stars[(p_value > 0.05) & (p_value <= 0.1)] = "."

            df = pd.DataFrame({
                "Tau": tau_hat,
                "As.se": se_rob,
                "T-stat": t_stat,
                "P-value": p_value,
                "CI.left(95%)": CI_left,
                "CI.right(95%)": CI_right,
                "Significance": stars
            })

            is_df_num_col = df.apply(lambda col: pd.api.types.is_numeric_dtype(col))
            df.loc[:, is_df_num_col] = df.loc[:, is_df_num_col].round(5)
            output.append(df.to_string(index=False))
            output.append("\n---\n")
            output.append("Signif. codes:  0 `***` 0.001 `**` 0.01 `*` 0.05 `.` 0.1 ` ` 1\n")

            if 'Ng' not in self.data.columns or self.data['Ng'].isnull().all():
                output.append("Warning: Cluster sizes have not been provided (Ng = None). Ng is assumed to be equal to the number of available observations in every cluster g.\n")
            if self.lin_adj is not None:
                if any([np.isnan(x).any() for x in self.ols_iter]):
                    raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
            if self.lin_adj is not None:
                if not check_cluster(pd.DataFrame({"G_id": self.data['G_id'], **self.lin_adj.to_dict()})):
                    output.append("Warning: sreg cannot use individual-level covariates for covariate adjustment in cluster-randomized experiments. Any individual-level covariates have been aggregated to their cluster-level averages.\n")
        return ''.join(output)
