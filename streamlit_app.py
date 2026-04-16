from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

def find_data_csv():
    """Look for combined CSV next to this script or in ../data/ (same layout as README)."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "boston_crime_combined.csv",
        here.parent / "data" / "boston_crime_combined.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner = "Loading and aggregating data ...")
def load_and_aggregate_monthly(filepath):
    """
    Same idea as time_series.ipynb: monthly counts, drop partial months, MS frequency.
    """
    df = pd.read_csv(filepath, low_memory = False)
    df["OCCURRED_ON_DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"], format = "mixed", utc = True)

    # shooting flag (same cleaning as data_prep / mlp notebook)
    df["SHOOTING"] = pd.to_numeric(df["SHOOTING"], errors = "coerce").fillna(0).astype(int)

    monthly = df.groupby(["YEAR", "MONTH"], as_index = False).agg(
        crime_count = ("OCCURRED_ON_DATE", "size"),
        shootings = ("SHOOTING", "sum"),
    )

    monthly["date"] = pd.to_datetime(monthly[["YEAR", "MONTH"]].assign(DAY = 1))
    monthly = monthly.sort_values("date").reset_index(drop = True)

    # drop partial months (same as notebook)
    partial = (
        ((monthly.YEAR == 2015) & (monthly.MONTH == 6)) |
        ((monthly.YEAR == 2026) & (monthly.MONTH == 2))
    )
    monthly = monthly[~partial].reset_index(drop = True)

    monthly = monthly.set_index("date")
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp()
    monthly.index.freq = "MS"

    return monthly[["crime_count", "shootings"]]


def create_covid_indicator(index):
    """
    Binary indicator for COVID period, same window as time_series.ipynb.
    March 2020 - December 2021.
    """
    dates = pd.to_datetime(index)
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2021-12-31")
    return ((dates >= covid_start) & (dates <= covid_end)).astype(int)


@st.cache_data(show_spinner = "Fitting SARIMAX and generating forecast ...")
def fit_and_forecast(train_values, train_dates, test_dates):
    """
    Fit SARIMAX(1,1,2)(0,1,1,12) + COVID then forecast the test period.
    Same best params from time_series.ipynb. Returns arrays that are easy to cache.
    """
    order = (1, 1, 2)
    seasonal_order = (0, 1, 1, 12)

    train_index = pd.DatetimeIndex(train_dates, freq = "MS")
    test_index = pd.DatetimeIndex(test_dates, freq = "MS")

    covid_train = create_covid_indicator(train_index)
    covid_test = create_covid_indicator(test_index)

    model = SARIMAX(
        train_values,
        order = order,
        seasonal_order = seasonal_order,
        exog = covid_train,
        enforce_stationarity = False,
        enforce_invertibility = False,
    )
    fitted = model.fit(disp = False)

    fc = fitted.get_forecast(steps = len(test_index), exog = covid_test)

    preds = np.asarray(fc.predicted_mean)

    ci_raw = fc.conf_int()
    ci = np.asarray(ci_raw)  # plain numpy so it serializes fine

    return preds, ci


def plot_series(monthly, n_test, show_shootings, year_lo, year_hi):
    """
    Train (blue) + actual test (red) + SARIMAX forecast (orange dashed) + CI band.
    """
    train = monthly.iloc[:-n_test]
    test = monthly.iloc[-n_test:]

    # fit model on train, get forecast for test period
    preds, ci_arr = fit_and_forecast(
        train["crime_count"].values,
        list(train.index.strftime("%Y-%m-%d")),
        list(test.index.strftime("%Y-%m-%d")),
    )
    # wrap CI back into a DataFrame aligned with test index for easy slicing
    ci = pd.DataFrame(ci_arr, index = test.index, columns = ["lower", "upper"])

    # compute error metrics on test set
    actual = test["crime_count"].values
    errors = actual - preds
    mae  = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual)) * 100

    # year filter
    def in_year(idx):
        y = idx.year
        return (y >= year_lo) & (y <= year_hi)

    m  = monthly.loc[in_year(monthly.index)]
    tr = train.loc[in_year(train.index)]
    te = test.loc[in_year(test.index)]

    # mask forecast arrays to match filtered test index
    test_mask = np.asarray(in_year(test.index))
    preds_f = preds[test_mask]
    ci_f = ci.iloc[test_mask]

    fig, ax1 = plt.subplots(figsize = (12, 4.5))

    # train
    ax1.plot(tr.index, tr["crime_count"], color = "steelblue", label = "Train (actual)")

    # actual test values
    ax1.plot(te.index, te["crime_count"], color = "red", marker = "o",
             markersize = 4, label = "Test (actual)")

    # SARIMAX forecast
    if len(preds_f) > 0:
        ax1.plot(te.index, preds_f, color = "orange", linestyle = "--", linewidth = 2,
                 marker = "s", markersize = 4, label = "SARIMAX forecast")
        ax1.fill_between(te.index, ci_f["lower"].values, ci_f["upper"].values,
                         color = "orange", alpha = 0.15, label = "95% CI")

    if len(test) > 0:
        ax1.axvline(test.index[0], color = "gray", linestyle = "--", alpha = 0.6)

    ax1.set_ylabel("Crime count")
    ax1.set_title("Monthly Boston Crime Volume — SARIMAX Forecast vs Actual")
    ax1.legend(loc = "upper left", fontsize = 8)

    if show_shootings and m["shootings"].sum() > 0:
        ax2 = ax1.twinx()
        ax2.plot(m.index, m["shootings"], color = "darkgreen", alpha = 0.6,
                 linewidth = 1.2, label = "Shootings / month")
        ax2.set_ylabel("Shootings")
        ax2.legend(loc = "upper right", fontsize = 8)

    plt.tight_layout()
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return fig, len(train), len(test), metrics


def build_analysis(n_train, n_test, metrics, year_lo, year_hi):
    """
    Return a markdown paragraph that changes based on current filter settings.
    """
    mae  = metrics["MAE"]
    rmse = metrics["RMSE"]
    mape = metrics["MAPE"]

    text = (
        f"With **{n_train}** training months and **{n_test}** test months, "
        f"the SARIMAX(1,1,2)(0,1,1,12) model with a COVID indicator achieves "
        f"a MAE of **{mae:.0f}**, RMSE of **{rmse:.0f}**, and MAPE of **{mape:.1f}%** "
        f"on the held-out period. "
    )

    if mape < 5:
        text += (
            "A MAPE under 5% suggests the model captures the overall trend and "
            "seasonality quite well. "
        )
    elif mape < 10:
        text += (
            "A MAPE between 5% and 10% indicates reasonable performance, though "
            "the model may struggle with unusual months or structural shifts. "
        )
    else:
        text += (
            "A MAPE above 10% suggests the forecast is not tracking the actual "
            "counts closely — this can happen when the test window includes a "
            "sharp regime change (e.g. COVID) that the training data did not cover. "
        )

    if n_test > 18:
        text += (
            "Holding out more than 18 months makes long-horizon forecasts harder, "
            "so higher errors are expected compared to the 12-month window used in the notebook. "
        )
    elif n_test < 6:
        text += (
            "With fewer than 6 test months the metrics are based on very few points, "
            "so they may not generalize well. "
        )

    # note about year filter
    text += (
        f"The plot currently shows years **{year_lo}–{year_hi}**; the forecast is "
        f"always over the last {n_test} months of the full series regardless of the year range."
    )

    return text



def render_landing_page():
    st.markdown("## Boston Crime Analysis — DS 4420 Final Project")
    st.markdown("**SioWa Luo, Li Zou, Shijie Lin**")
    st.markdown("---")

    st.markdown("### Background")
    st.markdown(
        "Crime prediction is an important public safety problem. Being able to forecast "
        "when and where crime is likely to occur can help law enforcement allocate resources "
        "more effectively. In this project, we apply three machine learning methods to "
        "Boston Police Department crime incident data to explore two questions: can we "
        "forecast monthly crime volume, and can we predict whether a crime incident involves "
        "a shooting?"
    )

    st.markdown("### Data")
    st.markdown(
        "We use crime incident reports from [Analyze Boston]"
        "(https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system), "
        "covering 2015 to 2026 with around 905,000 records. Each record includes information "
        "such as district, date and time, day of week, and geographic coordinates. Shooting "
        "incidents make up only about 0.7% of all records, so we apply undersampling to "
        "balance the classes for the classification models."
    )

    st.markdown("### Models")

    # three models in separate blocks
    st.markdown("#### 1 · SARIMA / SARIMAX")
    st.markdown(
        "We fit a SARIMA model on monthly crime counts. A grid search selects the best "
        "(p, d, q)(P, D, Q, s) parameters, and we add a COVID binary indicator to account "
        "for the sharp drop in 2020. The model is evaluated with both a 12-month holdout "
        "and a rolling one-step-ahead scheme."
    )

    st.markdown("#### 2 · MLP Neural Network")
    st.markdown(
        "An MLP built from scratch in NumPy classifies whether an incident involves a "
        "shooting. Features include cyclical-encoded hour and month, one-hot district and "
        "day of week, and geographic coordinates. We balance the dataset by undersampling "
        "non-shooting records."
    )

    st.markdown("#### 3 · Bayesian Logistic Regression")
    st.markdown(
        "A Bayesian logistic regression in R using brms gives us full posterior distributions "
        "over the coefficients, so we can quantify uncertainty. The same balanced dataset and "
        "features are used. Trace plots and posterior predictive checks confirm convergence."
    )

    st.markdown("### Summary")
    st.markdown(
        "The SARIMA model achieves a MAPE of 3.2% and reduces RMSE by 38.6% over a naive "
        "baseline. For the classification task, the MLP achieves an F1 of 0.756 and the "
        "Bayesian model achieves an F1 of 0.714. Both models show that district location "
        "is the strongest predictor of shooting risk, with B2 and B3 showing the highest "
        "risk overall."
    )


def main():
    st.set_page_config(page_title = "Boston Crime — DS 4420", layout = "wide")

    tab_home, tab_viz = st.tabs(["About the Project", "Time Series Explorer"])

    # tab 1: landing page
    with tab_home:
        render_landing_page()

    # tab 2: interactive viz 
    with tab_viz:
        path = find_data_csv()
        if path is None:
            st.warning(
                "Could not find `boston_crime_combined.csv`. "
                "Place it next to `streamlit_app.py` or under `../data/`."
            )
            return

        st.caption(f"Loaded: `{path}`")

        monthly = load_and_aggregate_monthly(str(path))

        years = sorted(monthly.index.year.unique())
        y0, y1 = int(years[0]), int(years[-1])

        c1, c2, c3 = st.columns(3)
        with c1:
            n_test = st.slider(
                "Months held out (test period)",
                min_value = 1,
                max_value = min(36, len(monthly) - 24),
                value = min(12, len(monthly) // 4),
                help = "Matches the train/test idea in time_series.ipynb",
            )
        with c2:
            year_lo = st.slider("Plot from year", min_value = y0, max_value = y1, value = y0)
        with c3:
            year_hi = st.slider("Plot through year", min_value = y0, max_value = y1, value = y1)

        if year_lo > year_hi:
            st.error("Start year must be <= end year.")
            return

        show_shoot = st.checkbox("Overlay monthly shooting counts (right axis)", value = False)

        fig, n_train, n_test_len, metrics = plot_series(monthly, n_test, show_shoot, year_lo, year_hi)
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            f"Full series: **{len(monthly)}** months. "
            f"Current split: **{n_train}** train, **{n_test_len}** test."
        )

        # dynamic analysis section
        st.markdown("---")
        st.subheader("Analysis")
        st.markdown(build_analysis(n_train, n_test_len, metrics, year_lo, year_hi))

        # collapsible monthly table
        with st.expander("Monthly data table (filtered years)"):
            view = monthly.loc[(monthly.index.year >= year_lo) & (monthly.index.year <= year_hi)].copy()
            view.index = view.index.strftime("%Y-%m")
            view = view.rename(columns = {"crime_count": "incidents", "shootings": "shootings"})
            st.dataframe(view, use_container_width = True)


if __name__ == "__main__":
    main()
