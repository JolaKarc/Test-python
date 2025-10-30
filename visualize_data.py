"""Utility module for loading tabular data and creating simple visualizations."""
from __future__ import annotations

import os
from typing import Iterable, Mapping, MutableMapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


FilterValue = Union[Tuple[Optional[float], Optional[float]], Iterable[Union[str, int, float]], str, int, float]


def load_dataset(path: str, **read_kwargs: MutableMapping[str, object]) -> pd.DataFrame:
    """Load a dataset from ``path`` using :func:`pandas.read_csv`.

    Parameters
    ----------
    path:
        Path to the CSV file that should be read.
    **read_kwargs:
        Optional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    dataset = pd.read_csv(path, **read_kwargs)
    return dataset


def _apply_filters(data: pd.DataFrame, filters: Optional[Mapping[str, FilterValue]] = None) -> pd.DataFrame:
    """Filter ``data`` according to ``filters`` description."""
    if not filters:
        return data

    filtered = data
    for column, condition in filters.items():
        if column not in filtered.columns:
            raise KeyError(f"Column '{column}' was not found in the dataset.")

        if isinstance(condition, tuple) and len(condition) == 2:
            lower, upper = condition
            if lower is not None:
                filtered = filtered[filtered[column] >= lower]
            if upper is not None:
                filtered = filtered[filtered[column] <= upper]
        elif isinstance(condition, (set, list, tuple)) and not (isinstance(condition, tuple) and len(condition) == 2):
            filtered = filtered[filtered[column].isin(condition)]
        else:
            filtered = filtered[filtered[column] == condition]
    return filtered


def create_plot(
    data: pd.DataFrame,
    *,
    x: str,
    y: Optional[str] = None,
    kind: str = "line",
    filters: Optional[Mapping[str, FilterValue]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dir: str = "plots",
) -> plt.Axes:
    """Generate a plot using ``matplotlib``.

    Parameters
    ----------
    data:
        Dataset to visualize.
    x, y:
        Column names used for the axes. ``y`` is optional for ``hist`` plots.
    kind:
        Type of the plot (``"line"``, ``"hist"`` or ``"scatter"``).
    filters:
        Mapping that defines filtering conditions before plotting. A tuple ``(min, max)``
        restricts the numeric column to that range, an iterable performs an ``isin`` check,
        while any other value is matched directly.
    xlabel, ylabel, title:
        Optional axis labels and title overrides.
    save_name:
        When provided, the figure is stored under ``plots/save_name``.
    save_dir:
        Directory where plots are saved (created automatically).

    Returns
    -------
    matplotlib.axes.Axes
        The axis that contains the generated plot.
    """
    filtered = _apply_filters(data, filters)

    if kind not in {"line", "hist", "scatter"}:
        raise ValueError(f"Unsupported plot kind '{kind}'. Expected 'line', 'hist' or 'scatter'.")

    if kind in {"line", "scatter"} and y is None:
        raise ValueError("Parameter 'y' must be provided for line and scatter plots.")

    if kind == "line":
        ax = filtered.plot(x=x, y=y, kind="line")
    elif kind == "scatter":
        ax = filtered.plot.scatter(x=x, y=y)
    else:  # hist
        ax = filtered[x].plot(kind="hist")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        ax.figure.savefig(save_path)

    return ax


def visualize_dataset(
    csv_path: str,
    *,
    x: str,
    y: Optional[str] = None,
    kind: str = "line",
    filters: Optional[Mapping[str, FilterValue]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dir: str = "plots",
    **read_kwargs,
) -> plt.Axes:
    """Load data from ``csv_path`` and generate a visualization."""
    dataset = load_dataset(csv_path, **read_kwargs)
    return create_plot(
        dataset,
        x=x,
        y=y,
        kind=kind,
        filters=filters,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        save_name=save_name,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    demo_path = os.path.join("data", "sample_data.csv")
    axes = visualize_dataset(
        demo_path,
        x="date",
        y="value",
        kind="line",
        filters={"category": {"A", "B"}},
        xlabel="Data",
        ylabel="Reikšmė",
        title="Mėginio duomenų dinamika",
        save_name="sample_line_plot.png",
    )
    print("Sugeneruota vizualizacija ir išsaugota aplanke 'plots'.")
    plt.show(block=False)
    plt.pause(1.5)
    plt.close(axes.figure)
