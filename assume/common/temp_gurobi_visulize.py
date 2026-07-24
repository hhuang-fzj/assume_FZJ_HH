import re
import math
import csv
from collections.abc import Mapping
from datetime import date, datetime
from numbers import Number
from pathlib import Path
import gurobipy as gp
import matplotlib
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _parse_var_name(var_name):
    """
    Parses Gurobi variable names like:
        x[0]
        x[i,0]
        flow[A,B,12]

    Returns:
        base_name: str
        indices: tuple[str, ...]
    """
    match = re.match(r"^([^\[]+)\[(.*)\]$", var_name)
    if not match:
        return var_name, tuple()

    base = match.group(1)
    idx_raw = match.group(2)

    if idx_raw == "":
        return base, tuple()

    indices = tuple(part.strip() for part in idx_raw.split(","))
    return base, indices


def _looks_like_time_series(var_names):
    """
    Heuristic:
    A group is treated as a time series if the last index is numeric
    for at least one variable.
    """
    for name in var_names:
        _, indices = _parse_var_name(name)
        if indices:
            try:
                int(indices[-1])
                return True
            except ValueError:
                pass
    return False


def _time_index_from_name(var_name):
    """
    Extracts the last index as time index.
    Example:
        AC_Qdot_out[3]      -> 3
        flow[A,B,3]         -> 3
    """
    _, indices = _parse_var_name(var_name)

    if not indices:
        return None

    try:
        return int(indices[-1])
    except ValueError:
        return None


def _series_key_from_name(var_name):
    """
    Removes the final numeric time index to identify one time series.

    Examples:
        AC_Qdot_out[0]      -> AC_Qdot_out
        flow[A,B,0]         -> flow[A,B]
        y[unit1,3]          -> y[unit1]

    Non-indexed variables:
        z                   -> z
    """
    base, indices = _parse_var_name(var_name)

    if not indices:
        return base

    try:
        int(indices[-1])
        time_is_last_index = True
    except ValueError:
        time_is_last_index = False

    if not time_is_last_index:
        return var_name

    non_time_indices = indices[:-1]

    if len(non_time_indices) == 0:
        return base

    return f"{base}[{','.join(non_time_indices)}]"


def _parse_iso_datetime(value):
    """
    Parses ISO-like datetime strings used for time-index input and zoom bounds.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _normalize_time_value(value):
    """
    Converts common timestamp objects to values matplotlib can format cleanly.
    """
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()

    if type(value).__name__ == "datetime64":
        try:
            return value.astype("datetime64[us]").astype(datetime)
        except (AttributeError, TypeError, ValueError):
            return value

    if isinstance(value, str):
        try:
            return _parse_iso_datetime(value)
        except ValueError:
            return value

    return value


def _time_value_for_step(time_index, step):
    """
    Maps an optimization timestep integer to the provided external time index.

    For sequences and pandas indexes, timestep 0 maps to time_index[0].
    For mappings, the timestep itself is used as key; stringified integer keys
    are also accepted for convenience.
    """
    if isinstance(time_index, Mapping):
        if step in time_index:
            return _normalize_time_value(time_index[step])

        step_as_text = str(step)
        if step_as_text in time_index:
            return _normalize_time_value(time_index[step_as_text])

        raise KeyError(step)

    if hasattr(time_index, "iloc"):
        return _normalize_time_value(time_index.iloc[step])

    return _normalize_time_value(time_index[step])


def _validate_time_index_for_steps(time_index, timesteps):
    """
    Returns the provided time index only if it covers all selected timesteps.
    """
    if time_index is None or not timesteps:
        return None

    missing_steps = []

    for step in timesteps:
        try:
            _time_value_for_step(time_index, step)
        except (IndexError, KeyError, TypeError):
            missing_steps.append(step)

    if missing_steps:
        preview = ", ".join(str(step) for step in missing_steps[:5])
        suffix = "..." if len(missing_steps) > 5 else ""
        print(
            "\nWarning: the provided time_index does not cover timestep(s) "
            f"{preview}{suffix}. Falling back to numeric timestep labels."
        )
        return None

    first_value = _time_value_for_step(time_index, timesteps[0])

    if not (_is_datetime_like(first_value) or isinstance(first_value, Number)):
        print(
            "\nWarning: the provided time_index is not datetime-like or numeric. "
            "Falling back to numeric timestep labels."
        )
        return None

    return time_index


def _is_datetime_like(value):
    """
    Detects values that should use a datetime-aware matplotlib x-axis.
    """
    return isinstance(value, (datetime, date))


def _selected_timesteps(series_data, selected_series):
    """
    Returns sorted timesteps that occur in the selected series.
    """
    timesteps = set()

    for key in selected_series:
        if key in series_data:
            timesteps.update(series_data[key].keys())

    return sorted(timesteps)


def _plot_axis_values(times, time_index):
    """
    Builds x-axis values for a single series without filling missing timesteps.
    """
    if time_index is None:
        return list(times)

    return [_time_value_for_step(time_index, step) for step in times]


def _uses_datetime_axis(time_index, timesteps):
    """
    Checks whether the selected plot should use calendar/time formatting.
    """
    if time_index is None or not timesteps:
        return False

    first_value = _time_value_for_step(time_index, timesteps[0])
    return _is_datetime_like(first_value)


def _format_axis_value(value):
    """
    Formats timestamps and numeric timesteps for user prompts.
    """
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")

    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def _parse_zoom_range(raw_range, datetime_axis, reference_value=None):
    """
    Parses a user-entered x-axis zoom range.
    """
    parts = [part.strip() for part in raw_range.split(",") if part.strip()]

    if len(parts) != 2:
        raise ValueError("Please enter exactly two values separated by a comma.")

    if datetime_axis:
        start = _parse_iso_datetime(parts[0])
        end = _parse_iso_datetime(parts[1])

        if (
            isinstance(reference_value, datetime)
            and reference_value.tzinfo is not None
            and start.tzinfo is None
            and end.tzinfo is None
        ):
            start = start.replace(tzinfo=reference_value.tzinfo)
            end = end.replace(tzinfo=reference_value.tzinfo)
    else:
        start = float(parts[0])
        end = float(parts[1])

    if end < start:
        print("Zoom range end is before start; swapping the bounds.")
        start, end = end, start

    return start, end


def ask_zoom_range(time_index, timesteps):
    """
    Asks the user whether the x-axis should be restricted to a time range.
    """
    if not timesteps:
        return None

    answer = input("\nZoom to a designated time range? [y/N]: ").strip().lower()

    if answer not in ("y", "yes"):
        return None

    datetime_axis = _uses_datetime_axis(time_index, timesteps)
    axis_values = _plot_axis_values(timesteps, time_index)

    print(
        "Available plot range: "
        f"{_format_axis_value(axis_values[0])} to {_format_axis_value(axis_values[-1])}"
    )

    if datetime_axis:
        print("Enter start and end as ISO-like dates/timestamps separated by a comma.")
        if isinstance(axis_values[0], datetime):
            print("Example: 2024-01-01 00:00, 2024-01-02 12:00")
        else:
            print("Example: 2024-01-01, 2024-01-07")
        prompt = "Start time, end time: "
    else:
        print("Enter start and end timesteps separated by a comma.")
        print("Example: 24, 72")
        prompt = "Start timestep, end timestep: "

    raw_range = input(prompt).strip()

    if not raw_range:
        print("No zoom range provided. Plotting the full range.")
        return None

    try:
        return _parse_zoom_range(
            raw_range,
            datetime_axis,
            reference_value=axis_values[0],
        )
    except ValueError as exc:
        print(f"Invalid zoom range: {exc}")
        print("Plotting the full range.")
        return None


def collect_solution_timeseries(model, include_zero_series=True, zero_tol=1e-9):
    """
    Collects solution values from a solved Gurobi model.

    Returns:
        series_data:
            dict[str, dict[int, float]]
            Example:
                {
                    "AC_Qdot_out": {0: 10.0, 1: 12.0},
                    "flow[A,B]": {0: 5.0, 1: 7.0}
                }

        scalar_data:
            dict[str, float]

        binary_series:
            set[str]

        binary_scalars:
            set[str]
    """
    if model.SolCount == 0:
        raise RuntimeError(
            f"No solution available. Model status = {model.Status}. "
            "You can only plot variable .X values after Gurobi has found a feasible solution."
        )

    vars_ = model.getVars()

    series_data = {}
    scalar_data = {}
    binary_series = set()
    binary_scalars = set()

    for v in vars_:
        name = v.VarName
        value = v.X
        vtype = v.VType

        t = _time_index_from_name(name)

        if t is None:
            scalar_data[name] = value
            if vtype == GRB.BINARY:
                binary_scalars.add(name)
            continue

        key = _series_key_from_name(name)

        if key not in series_data:
            series_data[key] = {}

        series_data[key][t] = value

        if vtype == GRB.BINARY:
            binary_series.add(key)

    if not include_zero_series:
        filtered = {}
        for key, values_by_t in series_data.items():
            if any(abs(val) > zero_tol for val in values_by_t.values()):
                filtered[key] = values_by_t
        series_data = filtered

    return series_data, scalar_data, binary_series, binary_scalars


def write_available_timeseries_csv(series_data, time_index=None):
    """
    Writes all available time series as a wide CSV table.

    Each time series is one column. Missing timestep values are left blank, so
    gaps in an optimization result are visible instead of silently filled.
    """
    output_path = Path(__file__).with_name("available_time_series.csv")
    series_keys = sorted(series_data.keys())
    timesteps = _selected_timesteps(series_data, series_keys)
    csv_time_index = _validate_time_index_for_steps(time_index, timesteps)
    has_time_column = csv_time_index is not None

    header = ["time_step"]
    if has_time_column:
        header.append("time")
    header.extend(series_keys)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for step in timesteps:
            row = [step]

            if has_time_column:
                row.append(_format_axis_value(_time_value_for_step(csv_time_index, step)))

            for key in series_keys:
                row.append(series_data[key].get(step, ""))

            writer.writerow(row)

    print(f"\nAvailable time series values CSV written to: {output_path}")


def print_variable_overview(model):
    """
    Prints a compact overview of variables in the model.
    Repeated time-indexed variables are shown once.
    Binary variables are listed separately.
    """
    vars_ = model.getVars()

    grouped = {}
    binaries_grouped = {}

    for v in vars_:
        name = v.VarName
        key = _series_key_from_name(name)

        if key not in grouped:
            grouped[key] = {
                "representative": name,
                "count": 0,
                "vtype": v.VType,
                "lb": v.LB,
                "ub": v.UB,
            }

        grouped[key]["count"] += 1

        if v.VType == GRB.BINARY:
            if key not in binaries_grouped:
                binaries_grouped[key] = {
                    "representative": name,
                    "count": 0,
                }
            binaries_grouped[key]["count"] += 1

    print("\n==============================")
    print("VARIABLE OVERVIEW")
    print("==============================")

    for i, (key, info) in enumerate(sorted(grouped.items()), start=1):
        rep = info["representative"]
        count = info["count"]
        vtype = info["vtype"]
        lb = info["lb"]
        ub = info["ub"]

        repeated_text = f"repeated {count} times" if count > 1 else "scalar/single variable"

        print(
            f"{i:>4}. {key:<40} "
            f"first name: {rep:<45} "
            f"type: {vtype:<2} "
            f"bounds: [{lb}, {ub}] "
            f"({repeated_text})"
        )

    print("\n==============================")
    print("BINARY VARIABLES")
    print("==============================")

    if not binaries_grouped:
        print("No binary variables found.")
    else:
        for i, (key, info) in enumerate(sorted(binaries_grouped.items()), start=1):
            print(
                f"{i:>4}. {key:<40} "
                f"first name: {info['representative']:<45} "
                f"count: {info['count']}"
            )


def plot_selected_timeseries(series_data, selected_series, time_index=None, zoom_range=None):
    """
    Plots up to three selected time series in one figure.
    """
    if len(selected_series) == 0:
        print("No series selected.")
        return

    if len(selected_series) > 3:
        raise ValueError("Please select at most three time series for one plot.")

    all_timesteps = _selected_timesteps(series_data, selected_series)
    datetime_axis = _uses_datetime_axis(time_index, all_timesteps)

    fig, ax = plt.subplots(figsize=(11, 5))

    for key in selected_series:
        if key not in series_data:
            print(f"Warning: '{key}' was not found and will be skipped.")
            continue

        values_by_t = series_data[key]

        if not values_by_t:
            print(f"Warning: '{key}' has no values and will be skipped.")
            continue

        times = sorted(values_by_t.keys())
        x_values = _plot_axis_values(times, time_index)
        values = [values_by_t[t] for t in times]

        ax.plot(x_values, values, marker="o", label=key)

    if zoom_range is not None:
        ax.set_xlim(*zoom_range)

    if datetime_axis:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()

    ax.set_xlabel("Time" if time_index is not None else "Time step")
    ax.set_ylabel("Solution value")
    ax.set_title("Selected Gurobi Optimization Time Series")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def interactive_timeseries_plot(model, time_index=None, include_zero_series=True):
    """
    Main function:
        1. Checks that the model has a feasible solution.
        2. Prints compact variable overview.
        3. Lists binary variables separately.
        4. Asks user which time series to plot.
        5. Optionally asks for an x-axis zoom range.
        6. Plots up to three selected time series.

    Args:
        model:
            Solved Gurobi model.
        time_index:
            Optional external time index for plotting. For sequences and pandas
            indexes, optimization timestep 0 maps to time_index[0]. Use a
            mapping when model timestep numbers are not zero-based.
        include_zero_series:
            Whether all-zero time series should be listed.
    """
    # Backwards compatibility for older positional calls:
    # interactive_timeseries_plot(model, False)
    print(matplotlib.get_backend())
    if isinstance(time_index, bool) and include_zero_series is True:
        include_zero_series = time_index
        time_index = None

    if model.SolCount == 0:
        print("\nNo feasible solution is available.")
        print(f"Model status: {model.Status}")
        print("Do not access variable .X values until SolCount > 0.")
        return

    print_variable_overview(model)

    series_data, scalar_data, binary_series, binary_scalars = collect_solution_timeseries(
        model,
        include_zero_series=include_zero_series,
    )

    print("\n==============================")
    print("AVAILABLE TIME SERIES")
    print("==============================")

    if not series_data:
        print("No time-indexed variables detected.")
        return

    sorted_keys = sorted(series_data.keys())

    write_available_timeseries_csv(series_data, time_index=time_index)

    for i, key in enumerate(sorted_keys, start=1):
        n_points = len(series_data[key])
        is_binary = "binary" if key in binary_series else "continuous/integer"
        print(f"{i:>4}. {key:<45} points: {n_points:<5} type: {is_binary}")

    print("\nEnter up to three time series to plot.")
    print("You may enter either numbers or names.")
    print("Examples:")
    print("  1")
    print("  1, 2")
    print("  AC_Qdot_out, CC_Qdot_out")
    print("  1, AC_Qdot_out, 5")

    user_input = input("\nTime series to plot: ").strip()

    if not user_input:
        print("No input provided. Nothing plotted.")
        return

    raw_choices = [item.strip() for item in user_input.split(",") if item.strip()]

    if len(raw_choices) > 3:
        print("You selected more than three series. Only the first three will be plotted.")
        raw_choices = raw_choices[:3]

    selected = []

    for choice in raw_choices:
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sorted_keys):
                selected.append(sorted_keys[idx - 1])
            else:
                print(f"Invalid number ignored: {choice}")
        else:
            if choice in series_data:
                selected.append(choice)
            else:
                # Try partial matching as a convenience
                matches = [key for key in sorted_keys if choice.lower() in key.lower()]

                if len(matches) == 1:
                    selected.append(matches[0])
                    print(f"Matched '{choice}' to '{matches[0]}'.")
                elif len(matches) > 1:
                    print(f"Ambiguous input '{choice}'. Matches:")
                    for m in matches:
                        print(f"   - {m}")
                    print("Please use the exact name or list number next time.")
                else:
                    print(f"No match found for '{choice}'.")

    # Remove duplicates while preserving order
    selected_unique = []
    for key in selected:
        if key not in selected_unique:
            selected_unique.append(key)

    if not selected_unique:
        print("No valid time series selected.")
        return

    selected_timesteps = _selected_timesteps(series_data, selected_unique)
    plot_time_index = _validate_time_index_for_steps(time_index, selected_timesteps)
    zoom_range = ask_zoom_range(plot_time_index, selected_timesteps)

    plot_selected_timeseries(
        series_data,
        selected_unique,
        time_index=plot_time_index,
        zoom_range=zoom_range,
    )


# ============================================================
# Usage after solving your model:
# ============================================================
#
# model.optimize()
#
# interactive_timeseries_plot(model)
#
# To plot against a datetime-aware index:
#
# interactive_timeseries_plot(model, time_index=my_datetime_index)
#
# If you do not want to list all-zero time series:
#
# interactive_timeseries_plot(model, include_zero_series=False)
