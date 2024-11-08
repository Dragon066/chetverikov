from __future__ import annotations

import warnings
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Chetverikov:
    """
    This class implements the Chetverikov method for time series analysis.

    The Chetverikov class provides several methods for fitting and analyzing time series data:

    - `fit`: fits the Chetverikov model to the provided time series data
    - `summary`: returns a summary of the fitted Chetverikov model, including the trend, seasonality, and residual components
    - `plot`: generates plots of the trend, seasonality, and residual components of the Chetverikov model

    The Chetverikov class also provides several attributes that store the fitted data and results of the analysis:

    - `y`: the original time series data
    - `L`: the number of periods in the time series
    - `rolling`: the size of the rolling window used for second trend assessment
    - `f0`: the preliminary trend assessment
    - `f1`: the first trend assessment
    - `f2`: the second trend assessment
    - `s1`: the first assessment of seasonality
    - `s2`: the second assessment of seasonality
    - `resid`: the residual component
    - `k`: the intensity coefficient of the seasonal wave
    """

    def __init__(self) -> None:
        self.y = None
        self.L = None
        self.rolling = None
        self.f0 = None
        self.f1 = None
        self.f2 = None
        self.s1 = None
        self.s2 = None
        self.resid = None
        self.k = None

    def _calculate_seasonality(self, y: np.array, trend: np.array):
        deviation = y - trend

        sigma = np.sqrt(
            (np.sum(deviation**2, axis=1) - np.sum(deviation, axis=1) ** 2 / self.L)
            / (self.L - 1)
        )

        deviation_norm = deviation / sigma[:, None]

        seasonality = np.mean(deviation_norm, axis=0)

        return seasonality, sigma[:, None], deviation

    def fit(self, y: Iterable, L: int = 12, rolling: int = None) -> Chetverikov:
        """Fit the Chetverikov model.

        Args:
            y (Iterable): any iterable object with target values.
            L (int, optional): Number of periods (months, quarters, etc.). Defaults to 12.
            rolling (int, optional): Size of rolling window for second trend assessment. Defaults to (L // 2 - 1).

        Returns:
            self: Chetverikov instance with fitted data.
        """
        y = np.array(y).reshape(-1)

        assert len(y) >= L, f"Data contains less periods than {L=}."

        if rolling:
            assert len(y) >= rolling, "Rolling window is too big for presented data."

        if len(y) % L:
            warnings.warn(
                f"Data can't be divided into {L} periods. The last {len(y) % L} periods are truncated."
            )
            y = y[: -(len(y) % L)]

        if rolling and rolling >= L:
            warnings.warn(
                f"Rolling window ({rolling}) is bigger than amount of periods ({L}). It can distort the results."
            )

        self.y = y
        self.L = L

        if rolling is None:
            rolling = int(0.5 * L)
        self.rolling = rolling

        y_reshaped = y.reshape(-1, L)

        chronological_mean = (
            0.5 * y_reshaped[:, 0]
            + np.sum(y_reshaped[:, 1:-1], axis=1)
            + 0.5 * y_reshaped[:, -1]
        ) / L

        s1, sigma1, _ = self._calculate_seasonality(
            y_reshaped, chronological_mean[:, None]
        )

        f1 = y_reshaped - s1 * sigma1
        f1 = f1.reshape(-1)

        f2 = [np.nan] * (int(0.5 * L) - 1)

        f2 = pd.DataFrame(f1).rolling(window=rolling).mean()

        for i in range(rolling - 1):
            f2.iloc[i] = np.average(f1[: i + 1], weights=np.arange(i + 1, 0, -1))

        f2 = np.array(f2).reshape(-1, L)

        s2, sigma2, l2 = self._calculate_seasonality(y_reshaped, f2)

        resid = l2 - s2[[list(range(0, L)) for i in range(len(y_reshaped))]] * sigma2

        k = np.sum(l2**2 * resid, axis=1) / np.sum(resid**2, axis=1)

        self.f0 = np.stack([chronological_mean] * L, axis=1).reshape(-1)
        self.f1 = f1
        self.f2 = f2.reshape(-1)
        self.s1 = s1
        self.s2 = s2
        self.resid = resid.reshape(-1)
        self.k = k

        return self

    def summary(self) -> pd.DataFrame:
        """This function generates a summary of the fitted Chetverikov model.

        The summary includes the following columns:

        - Initial series: The original time series data.
        - Preliminary trend assessment: The preliminary trend assessment.
        - First trend assessment: The first trend assessment.
        - Second trend assessment: The second trend assessment.
        - First assessment of seasonality: The first assessment of seasonality.
        - Second assessment of seasonality: The second assessment of seasonality.
        - Residual component: The residual component.
        - The intensity coefficient of the seasonal wave: The intensity coefficient of the seasonal wave.

        Returns:
            DataFrame: A summary of the fitted Chetverikov model.
        """
        assert self.y is not None, "Data isn't fitted yet. Use fit() method first."

        df = pd.DataFrame(index=pd.Series(np.arange(1, len(self.y) + 1), name="t"))

        df["Initial series"] = self.y

        df["Preliminary trend assessment"] = self.f0

        df["First trend assessment"] = self.f1

        df["Second trend assessment"] = self.f2

        df["First assessment of seasonality"] = np.stack(
            [self.s1] * (len(self.y) // self.L), axis=0
        ).reshape(-1)

        df["Second assessment of seasonality"] = np.stack(
            [self.s2] * (len(self.y) // self.L), axis=0
        ).reshape(-1)

        df["Residual component"] = self.resid

        df["The intensity coefficient of the seasonal wave"] = np.stack(
            [self.k] * self.L, axis=1
        ).reshape(-1)

        return df

    def plot(
        self, chart: Literal["trend_all", "trend", "season", "resid", "k"] = "trend_all"
    ) -> None:
        """This function generates plots of the trend, seasonality, and residual components of the Chetverikov model.

        Args:
            chart (str, optional): The type of chart to generate. Options are "trend_all", "trend", "season", "resid", and "k". Defaults to "trend_all".

        Returns:
            None: This function generates plots using Matplotlib and does not return any additional values.
        """
        available_charts = ["trend_all", "trend", "season", "resid", "k"]

        assert (
            chart in available_charts
        ), f"Chart {chart} is not available. Available charts are {available_charts}"

        if chart == "trend_all":
            self._plot_trend_all()
            return

        if chart == "trend":
            self._plot_trend()
            return

        if chart == "season":
            self._plot_season()
            return

        if chart == "resid":
            self._plot_resid()
            return

        if chart == "k":
            self._plot_k()
            return

    def _plot_trend_all(self) -> None:
        plt.title("Trend diagram")
        plt.grid(alpha=0.5, linestyle="--")

        titles = [
            "Initial series",
            "Preliminary trend assessment",
            "First trend assessment",
            "Second Trend assessment",
        ]

        plt.plot(self.y, label=titles[0])
        plt.plot(self.f0, label=titles[1])
        plt.plot(self.f1, label=titles[2])
        plt.plot(self.f2, label=titles[3])

        plt.legend()

        plt.show()

    def _plot_trend(self) -> None:
        plt.figure(figsize=(8, 8))
        plt.suptitle("Trend diagrams")
        plt.subplots_adjust(hspace=0.5)

        titles = [
            "Preliminary trend assessment",
            "First trend assessment",
            "Second Trend assessment",
        ]

        for i, value in enumerate([self.f0, self.f1, self.f2]):
            plt.subplot(3, 1, i + 1)
            plt.title(titles[i])
            plt.grid(alpha=0.5)
            if i == 2:
                plt.xlabel("t, time period")

            plt.plot(np.arange(len(value)), value, c="orange")

        plt.show()

    def _plot_season(self) -> None:
        plt.figure(figsize=(10, 7))
        plt.suptitle(f"Seasonal wave diagram (L = {self.L})")
        plt.subplots_adjust(hspace=0.5)

        titles = [
            "First assessment of seasonality",
            "Second assessment of seasonality",
            "Comparison",
        ]

        for i, col in enumerate([self.s1, self.s2, [self.s1, self.s2]]):
            plt.subplot(3, 1, i + 1)
            plt.title(titles[i])
            plt.grid(alpha=0.5, linestyle="--")
            if i == 2:
                plt.xlabel("L, period")

            if i < 2:
                plt.plot(
                    np.arange(1, self.L + 1),
                    col,
                    c=("red" if i else "green"),
                    linewidth=1,
                )
            else:
                plt.plot(np.arange(1, self.L + 1), col[0], c="red", linewidth=1)
                plt.plot(np.arange(1, self.L + 1), col[1], c="green", linewidth=1)

        plt.show()

    def _plot_resid(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.title("Residual component")
        plt.grid(alpha=0.5, linestyle="--")
        plt.xlabel("t, time period")
        plt.ylabel("error")
        plt.xlim(-1, len(self.y) + 1)
        plt.hlines(0, -1, len(self.y) + 1, color="black", linewidth=0.4)

        plt.plot(
            np.arange(len(self.y)),
            self.resid,
            c="lightblue",
            label="Residual component",
            linewidth=3,
        )

        plt.legend()

        plt.show()

    def _plot_k(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.title("The intensity coefficient of the seasonal wave (k)")
        plt.grid(alpha=0.5, linestyle="--")
        plt.xlabel("t * L, time period")
        plt.ylabel("k")

        plt.plot(
            np.arange(1, (len(self.y) // self.L) + 1),
            self.k,
            c="brown",
            label="k",
            linewidth=3,
        )

        plt.legend()

        plt.show()
