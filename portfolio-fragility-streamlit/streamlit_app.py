"""Streamlit entry point for Portfolio Fragility Lab."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import apply_institutional_theme, render_dashboard
from src.engine import (
    AnalysisConfig,
    PortfolioError,
    download_market_data,
    run_analysis,
    validate_portfolio,
)
from src.instruments import (
    asset_option_label,
    custom_horizon,
    parse_localized_number,
    portfolio_weight_total,
    yahoo_asset_search,
)


st.set_page_config(
    page_title="Portfolio Fragility Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_institutional_theme()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], benchmark: str, start: date):
    """Cache market downloads so presentation-only reruns remain fast."""
    return download_market_data(tickers, benchmark, start)


@st.cache_data(ttl=86_400, max_entries=512, show_spinner=False)
def cached_asset_search(query: str) -> list[dict[str, str]]:
    """Cache global Yahoo Finance searches to reduce latency and rate limits."""
    return yahoo_asset_search(query)


DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["AAPL", "MSFT", "GLD", "TLT"],
        "Weight (%)": ["30.0", "30.0", "20.0", "20.0"],
    }
)
if "portfolio_editor_data" not in st.session_state:
    st.session_state.portfolio_editor_data = DEFAULT_PORTFOLIO.copy()
if "portfolio_editor_version" not in st.session_state:
    st.session_state.portfolio_editor_version = 0
if "asset_search_results" not in st.session_state:
    st.session_state.asset_search_results = []


with st.sidebar:
    st.markdown("### Portfolio setup")
    st.caption("Add or remove assets and set target weights.")

    with st.form("global_asset_search", clear_on_submit=False):
        search_query = st.text_input(
            "Find an investment",
            placeholder="Apple, gold, Nikkei 225, ISCTR...",
            help=(
                "Search Yahoo Finance by company, fund, commodity, index, "
                "currency, cryptocurrency, or ticker name."
            ),
        )
        search_submitted = st.form_submit_button("Search Yahoo Finance", width="stretch")
    if search_submitted:
        try:
            if len(search_query.strip()) < 2:
                st.session_state.asset_search_results = []
                st.warning("Enter at least two characters.")
            else:
                st.session_state.asset_search_results = cached_asset_search(search_query.strip())
                if not st.session_state.asset_search_results:
                    st.info("No Yahoo Finance matches were found. You can still enter a ticker manually.")
        except Exception as error:
            st.session_state.asset_search_results = []
            st.warning("Yahoo Finance search is temporarily unavailable. You can still enter a ticker manually.")
            with st.expander("Search details"):
                st.code(str(error))

    if st.session_state.asset_search_results:
        search_results = st.session_state.asset_search_results
        selected_result = st.selectbox(
            "Search results",
            options=range(len(search_results)),
            format_func=lambda index: asset_option_label(search_results[index]),
            help="Results may include exchanges and instruments from multiple countries.",
        )
        add_weight = st.text_input(
            "Weight to add (%)",
            value="0",
            help="Both decimal point and decimal comma are accepted, for example 20.3 or 20,3.",
        )
        if st.button("Add selected asset", width="stretch"):
            asset = search_results[selected_result]
            current = st.session_state.portfolio_editor_data.copy()
            current_symbols = (
                current.get("Ticker", pd.Series(dtype=str))
                .astype(str)
                .str.strip()
                .str.upper()
            )
            parsed_add_weight = parse_localized_number(add_weight)
            if asset["symbol"] in set(current_symbols):
                st.info(f"{asset['symbol']} is already in the portfolio.")
            elif not np.isfinite(parsed_add_weight) or parsed_add_weight < 0:
                st.warning("Enter a valid non-negative weight before adding the asset.")
            else:
                new_row = pd.DataFrame(
                    {"Ticker": [asset["symbol"]], "Weight (%)": [str(add_weight).strip()]}
                )
                st.session_state.portfolio_editor_data = pd.concat(
                    [current, new_row], ignore_index=True
                )
                st.session_state.portfolio_editor_version += 1
                st.rerun()

    portfolio_table = st.data_editor(
        st.session_state.portfolio_editor_data,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"portfolio_editor_{st.session_state.portfolio_editor_version}",
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Yahoo Finance symbol. Use the search above or enter a valid symbol directly.",
            ),
            "Weight (%)": st.column_config.TextColumn(
                "Weight (%)",
                help="Accepts decimal point or comma, for example 20.3 or 20,3.",
            ),
        },
    )
    st.session_state.portfolio_editor_data = portfolio_table.copy()
    total_weight = portfolio_weight_total(portfolio_table)
    st.caption(f"Total weight: {total_weight:.1f}%")
    st.divider()
    initial_investment = st.number_input("Initial investment ($)", min_value=1, value=100_000, step=100)
    target_mode = st.selectbox(
        "Target type",
        ["Portfolio value ($)", "Growth target (%)"],
        help="Choose an absolute terminal value or a percentage increase over the initial investment.",
    )
    if target_mode == "Portfolio value ($)":
        target_input = st.number_input(
            "Target portfolio value ($)", min_value=1, value=115_000, step=100
        )
        target_value = float(target_input)
    else:
        target_input = st.number_input(
            "Target growth (%)",
            min_value=0.1,
            value=15.0,
            step=0.5,
            format="%.1f",
        )
        target_value = float(initial_investment) * (1 + float(target_input) / 100)
        st.caption(f"Resolved target value: ${target_value:,.0f}")
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    start_date = st.date_input("Historical data start", value=date(2018, 1, 1), max_value=date.today())
    st.markdown("##### Simulation horizon")
    horizon_amount_column, horizon_unit_column = st.columns([0.85, 1.15])
    with horizon_amount_column:
        horizon_amount = st.number_input(
            "Amount",
            min_value=1,
            max_value=10_000,
            value=1,
            step=1,
            label_visibility="collapsed",
        )
    with horizon_unit_column:
        horizon_unit = st.selectbox(
            "Unit",
            ["Hours", "Days", "Weeks", "Months", "Years"],
            index=4,
            label_visibility="collapsed",
        )
    simulation_days, horizon_label, horizon_calendar_days = custom_horizon(
        int(horizon_amount), horizon_unit
    )
    st.caption(
        f"{horizon_label} resolves to {simulation_days:,} model steps on a "
        "252-step annual basis. Short horizons use fractional-day return scaling."
    )
    simulations = st.select_slider("Number of simulations", [1_000, 5_000, 10_000, 25_000], value=10_000)
    method = st.selectbox(
        "Simulation model",
        ["Regime Switching", "Geometric Brownian Motion", "Historical Bootstrap", "Student-t"],
        help="Regime Switching explicitly models bull, normal, crisis, and recovery states.",
    )
    with st.expander("Liquidity cascade assumptions"):
        redemption_probability = st.slider("Redemption probability after a breach", 0.0, 1.0, 0.35, 0.05)
        redemption_pct = st.slider("Investor redemption (%)", 0, 40, 12, 1) / 100
        gross_leverage = st.number_input(
            "Collateral-assumption gross leverage (x)",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help=(
                "Used only for the synthetic collateral-call and liquidity overlay. "
                "It does not lever historical or simulated asset returns. At 1.0×, "
                "modeled collateral-call probability is zero."
            ),
        )
        margin_call_pct = st.slider("Margin call (%)", 0, 25, 5, 1) / 100
        margin_trigger = -st.slider("Drawdown trigger (%)", 5, 40, 15, 1) / 100
        cash_buffer_pct = st.slider("Cash buffer (%)", 0, 30, 3, 1) / 100
    with st.expander("Advanced model settings"):
        confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
        risk_free_rate = st.number_input("Annual risk-free rate (%)", 0.0, 30.0, 4.0, 0.25) / 100
        seed = st.number_input("Random seed", 0, 1_000_000, 42)
        student_df = st.slider("Student-t degrees of freedom", 3, 20, 5)
    run_button = st.button("Run Fragility Analysis", type="primary", width="stretch")
    st.caption("Educational research only. Not investment advice.")


def _portfolio_input_signature(table: pd.DataFrame) -> tuple[tuple[str, float | None], ...]:
    """Create a stable signature so results cannot outlive changed sidebar inputs."""
    rows: list[tuple[str, float | None]] = []
    for _, row in table.fillna("").iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        raw_weight = parse_localized_number(row.get("Weight (%)"))
        weight = None if pd.isna(raw_weight) else round(float(raw_weight), 8)
        rows.append((ticker, weight))
    return tuple(rows)


input_signature = (
    _portfolio_input_signature(portfolio_table),
    float(initial_investment),
    float(target_value),
    target_mode,
    float(target_input),
    benchmark,
    start_date.isoformat(),
    int(horizon_amount),
    horizon_unit,
    simulation_days,
    int(simulations),
    method,
    float(redemption_probability),
    float(redemption_pct),
    float(gross_leverage),
    float(margin_call_pct),
    float(margin_trigger),
    float(cash_buffer_pct),
    float(confidence),
    float(risk_free_rate),
    int(seed),
    int(student_df),
)

inputs_changed = (
    "analysis" in st.session_state
    and st.session_state.get("analysis_input_signature") != input_signature
)
if inputs_changed:
    # Never display results calculated from an earlier investment, target, model,
    # portfolio, or assumption set beside newly edited controls.
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_config", None)
    st.session_state.pop("analysis_input_signature", None)


if not run_button and "analysis" not in st.session_state:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Institutional portfolio risk intelligence</div>
          <h1>Portfolio Fragility Lab</h1>
          <p>Regime-aware Monte Carlo analysis, liquidity cascades, historical risk diagnostics,
          stress testing and crisis-resilient allocation in one decision-oriented workspace.</p>
        </div>
        <div class="method-note">Build the portfolio in the left panel, select the model assumptions,
        and click <b>Run Fragility Analysis</b>. Results are cleared whenever an input changes so an
        earlier investment amount can never be shown as the current analysis.</div>
        """,
        unsafe_allow_html=True,
    )
    if inputs_changed:
        st.info("Portfolio inputs changed. Run the analysis again to calculate results using the new values.")
    st.stop()


if run_button:
    try:
        tickers, weights = validate_portfolio(portfolio_table)
        if float(target_value) <= float(initial_investment):
            raise PortfolioError("Target portfolio value must be greater than the initial investment.")
        if benchmark in tickers:
            st.info("The benchmark is also held in the portfolio; relative analytics will still be calculated.")
        config = AnalysisConfig(
            tickers=tickers,
            weights=weights,
            benchmark=benchmark,
            start_date=start_date,
            initial_investment=float(initial_investment),
            simulation_days=simulation_days,
            simulations=int(simulations),
            confidence=float(confidence),
            target_value=float(target_value),
            risk_free_rate=float(risk_free_rate),
            method=method,
            seed=int(seed),
            student_df=int(student_df),
            redemption_probability=float(redemption_probability),
            redemption_pct=float(redemption_pct),
            gross_leverage=float(gross_leverage),
            margin_call_pct=float(margin_call_pct),
            margin_trigger=float(margin_trigger),
            cash_buffer_pct=float(cash_buffer_pct),
            horizon_label=horizon_label,
            calendar_days=float(horizon_calendar_days),
            base_currency="USD",
        )
        simulation_cells = config.simulation_days * config.simulations * len(config.tickers)
        if simulation_cells > 80_000_000:
            raise PortfolioError(
                "This horizon, path count, and asset count would exceed the cloud memory guard. "
                "Reduce the horizon, simulations, or number of assets."
            )
        with st.spinner("Downloading market data and running regime, liquidity, and stress models..."):
            prices, benchmark_prices, volumes = cached_market_data(tickers, benchmark, start_date)
            result = run_analysis(config, prices, benchmark_prices, volumes)
        st.session_state.analysis = result
        st.session_state.analysis_config = config
        st.session_state.analysis_input_signature = input_signature
    except PortfolioError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error("The analysis could not be completed. Check the tickers and try again.")
        with st.expander("Technical details"):
            st.code(str(error))
        st.stop()


result = st.session_state.analysis
config = st.session_state.analysis_config
st.success(
    f"Completed {config.simulations:,} paths for {', '.join(config.tickers)} "
    f"using {config.method}. Initial investment: ${config.initial_investment:,.0f}; "
    f"target: ${config.target_value:,.0f}."
)
render_dashboard(result, config)
