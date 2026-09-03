"""Formatted output and plotting for :class:`sreg.Sreg` results."""
import pandas as pd
import numpy as np

class Sreg:
    """Mapping-like result returned by :func:`sreg.sreg`.

    Use square brackets to access estimator components, ``print(fit)`` for the
    formatted table, and :meth:`plot` for confidence-interval visualization.
    """
    def __init__(self, result):
        self.result = result
    
    def __repr__(self):
        return self.print_sreg()
    
    def print_sreg(self):
        """Print the R-style estimation summary and return an empty string."""
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
            mixed=self.result.get('mixed_design',False); small=self.result.get('small_strata',False)
            print("Setup: " + ("mixed design (includes both small and large strata)" if mixed else "small strata" if small else "large strata"))
            if small:
                source=self.result['res_small']['data'] if mixed else self.result['data']
                k=len(source)/source['S'].nunique()
                print(("Strata size (k, small strata only): " if mixed else "Strata size (k): ")+f"{k:g}")
            print("Standard errors: " + ("adjusted (HC1)" if self.result.get('HC1',False) else "unadjusted"))
            print("Treatment assignment: individual level")
            print(f"Covariates used in linear adjustments: {', '.join(map(str,lin_adj.columns)) if lin_adj is not None else ''}")
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
            n = len(self.result['data'])
            G = len(self.result['data']['G_id'].unique())
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
            print(f"Clusters: {G}")
            print(f"Number of treatments: {self.result['data']['D'].max()}")
            print(f"Number of strata: {self.result['data']['S'].max()}")
            mixed=self.result.get('mixed_design',False); small=self.result.get('small_strata',False)
            print("Setup: " + ("mixed design (includes both small and large strata)" if mixed else "small strata" if small else "large strata"))
            if small:
                source=self.result['res_small']['data'] if mixed else self.result['data']
                sizes=source[['S','G_id']].drop_duplicates().groupby('S').size()
                k=(str(int(sizes.iloc[0])) if sizes.nunique()==1 else
                   f"varying (min={int(sizes.min())}, max={int(sizes.max())})")
                print(("Strata size (k, small strata only): " if mixed else "Strata size (k): ")+k)
            print("Standard errors: " + ("adjusted (HC1)" if self.result.get('HC1',False) else "unadjusted"))
            print("Treatment assignment: cluster level")
            print(f"Covariates used in linear adjustments: {', '.join(map(str,lin_adj.columns)) if lin_adj is not None else ''}")
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

    def __setitem__(self, key, value):
        self.result[key] = value

    def get(self, key, default=None):
        return self.result.get(key, default)

    def keys(self):
        return self.result.keys()

    def plot(self, level=0.95, ax=None, treatment_labels=None,
             title="Estimated ATEs with Confidence Intervals", bar_fill=None,
             point_shape='D', point_size=3, point_fill="white", point_stroke=1.2,
             point_color="black", label_color="black", label_size=4,
             bg_color=None, grid=True, zero_line=True,
             y_axis_title=None, x_axis_title=None, **kwargs):
        """Plot estimates and normal confidence intervals.

        Parameters
        ----------
        level : float, default=0.95
            Confidence level strictly between zero and one.
        ax : matplotlib.axes.Axes or None, default=None
            Existing axes; a new figure is created when omitted.
        treatment_labels : sequence of str or None, default=None
            One y-axis label per active treatment.
        title : str or None
            Plot title.
        bar_fill : color, two-color sequence, or None
            Confidence-bar color or gradient endpoints. ``None`` uses viridis.
        point_shape : str or int, default="D"
            Matplotlib marker or compatible R shape code 21--25.
        point_size, point_stroke : float
            Marker size and border width.
        point_fill, point_color : color
            Marker fill and outline colors.
        label_color, label_size : color, float
            Estimate-label styling.
        bg_color : color or None
            Axes background color.
        grid, zero_line : bool
            Display grid lines and the zero reference line.
        y_axis_title, x_axis_title : str or None
            Axis titles.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot. Access its figure through ``ax.figure``.

        Examples
        --------
        >>> data = __import__("sreg").sreg_rgen(60, tau_vec=(0.5,),
        ...     n_strata=3, cluster=False, random_state=4)
        >>> fit = __import__("sreg").sreg(data.Y, data.S, data.D)
        >>> ax = fit.plot()
        >>> ax.get_title()
        'Estimated ATEs with Confidence Intervals'
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        if not 0 < level < 1:
            raise ValueError("level must be between zero and one.")
        if ax is None:
            _, ax = plt.subplots()
        tau = np.asarray(self.result['tau_hat'])
        half = norm.ppf((1 + level) / 2) * np.asarray(self.result['se_rob'])
        y = np.arange(1, len(tau) + 1)
        import warnings
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        if bar_fill is None:
            colors=plt.get_cmap('viridis')(Normalize()(tau))
        elif isinstance(bar_fill,str):
            colors=[bar_fill]*len(tau)
        elif hasattr(bar_fill,'__len__') and len(bar_fill)==2:
            colors=LinearSegmentedColormap.from_list('sreg_bar_fill',bar_fill)(Normalize()(tau))
        else:
            warnings.warn("bar_fill must be None, a single color, or a sequence of two colors. Ignoring custom fill.",UserWarning,stacklevel=2)
            colors=plt.get_cmap('viridis')(Normalize()(tau))
        marker_map={23:'D',21:'o',22:'s',24:'^',25:'v'}
        marker=marker_map.get(point_shape,point_shape)
        for estimate,position,error,color,se in zip(tau,y,half,colors,np.asarray(self.result['se_rob'])):
            ax.errorbar(estimate,position,xerr=error,fmt=marker,capsize=4,
                        color=color,markersize=point_size*2,
                        markerfacecolor=point_fill,markeredgecolor=point_color,
                        markeredgewidth=point_stroke)
            ax.annotate(f"{estimate:.2f} ({se:.2f})",(estimate,position),
                        xytext=(0,8),textcoords='offset points',ha='center',
                        color=label_color,fontsize=label_size)
        if zero_line: ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        labels=treatment_labels or [f"Treatment {i}" for i in y]
        if len(labels)!=len(tau): raise ValueError("treatment_labels must have one label per treatment.")
        ax.set(yticks=y, yticklabels=labels, xlabel=x_axis_title, ylabel=y_axis_title,title=title)
        ax.grid(grid)
        if bg_color is not None: ax.set_facecolor(bg_color)
        return ax
