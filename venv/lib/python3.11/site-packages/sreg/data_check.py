# check the data
import numpy as np
import pandas as pd

def check_data_types(Y, S, D, G_id, Ng, X):
    non_null_vars = [var for var in [Y, S, D, G_id, Ng, X] if var is not None]
    var_names = ['Y', 'S', 'D', 'G_id', 'Ng', 'X']

    for var, name in zip([Y, S, D, G_id, Ng, X], var_names):
        if var is not None:
            if not (isinstance(var, (np.ndarray, pd.DataFrame, pd.Series, list)) or np.issubdtype(type(var), np.number)):
                raise ValueError(f"Error: Variable {name} has a different type than matrix, numeric series, or data frame.")

def check_integers(S, D, G_id, Ng):
    non_null_vars = {'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng}

    for var_name, var in non_null_vars.items():
        if var is not None:
            if isinstance(var, pd.DataFrame):
                if not all(var.apply(lambda col: col.dropna().apply(float.is_integer).all())):
                    raise ValueError(f"Error: Variable {var_name} must contain only integer values.")
            else:
                if not np.all(np.isnan(var) | (var.astype(int) == var)):
                    raise ValueError(f"Error: Variable {var_name} must contain only integer values.")
                
# def check_range(var, range_min=None, range_max=None):
#     def find_missing_values(data, current_range_min, current_range_max):
#         return set(range(current_range_min, current_range_max + 1)) - set(data.dropna().astype(int))
                
def check_range(var, range_min=None, range_max=None):
    def find_missing_values(data, current_range_min, current_range_max):
        # Convert current_range_min and current_range_max to integers
        current_range_min = int(current_range_min)
        current_range_max = int(current_range_max)
        return set(range(current_range_min, current_range_max + 1)) - set(data.dropna().astype(int))

    if isinstance(var, pd.DataFrame):
        for col in var.columns:
            data = var[col]
            current_range_min = int(range_min) if range_min is not None else int(np.nanmin(data))
            current_range_max = int(range_max) if range_max is not None else int(np.nanmax(data))

            missing_values = find_missing_values(pd.Series(data), current_range_min, current_range_max)
            if len(missing_values) > 0:
                raise ValueError(
                    f"Error: There are skipped values in the range of {col}: {missing_values}. "
                    "Variables S and D must not contain any skipped values within the range. "
                    "For example, if min(S) = 1 and max(S) = 3, then S should encompass values 1, 2, and 3."
                )
    else:
        current_range_min = int(range_min) if range_min is not None else int(np.nanmin(var))
        current_range_max = int(range_max) if range_max is not None else int(np.nanmax(var))

        missing_values = find_missing_values(pd.Series(var), current_range_min, current_range_max)
        if len(missing_values) > 0:
            raise ValueError(
                f"Error: There are skipped values in the range of {pd.Series(var).name}: {missing_values}. "
                "Variables S and D must not contain any skipped values within the range. "
                "For example, if min(S) = 1 and max(S) = 3, then S should encompass values 1, 2, and 3."
            )

    if isinstance(var, pd.DataFrame):
        for col in var.columns:
            data = var[col]
            current_range_min = range_min if range_min is not None else data.min()
            current_range_max = range_max if range_max is not None else data.max()
            
            missing_values = find_missing_values(data, current_range_min, current_range_max)
            if missing_values:
                raise ValueError(
                    f"Error: There are skipped values in the range of {col}: {', '.join(map(str, missing_values))}. "
                    "Variables S and D must not contain any skipped values within the range. For example, if min(S) = 1 and max(S) = 3, "
                    "then S should encompass values 1, 2, and 3."
                )
    else:
        current_range_min = range_min if range_min is not None else np.nanmin(var)
        current_range_max = range_max if range_max is not None else np.nanmax(var)
        
        missing_values = find_missing_values(pd.Series(var), current_range_min, current_range_max)
        if missing_values:
            var_name = "variable"
            raise ValueError(
                f"Error: There are skipped values in the range of D and/or S: {', '.join(map(str, missing_values))}. "
                "Variables S and D must not contain any skipped values within the range. For example, if min(S) = 1 and max(S) = 3, "
                "then S should encompass values 1, 2, and 3."
            )
        
def is_boolean(x):
    return isinstance(x, bool) and not pd.isna(x)

def boolean_check(var):
    if not is_boolean(var):
        raise ValueError("Error: The value of HC must be either True or False. A non-boolean value was provided.")