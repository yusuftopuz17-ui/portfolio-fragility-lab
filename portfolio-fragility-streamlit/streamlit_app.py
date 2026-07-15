"""Streamlit entry point for Portfolio Fragility Lab."""

from datetime import date

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


# Keep the Portfolio Setup panel stable: all inputs and defaults intentionally
# remain identical to the existing application.
with st.sidebar:
    st.markdown("### Portfolio setup")
    st.caption("Add or remove assets and set target weights.")
    default_portfolio = pd.DataFrame(
        {"Ticker": ["AAPL", "MSFT", "GLD", "TLT"], "Weight (%)": [30.0, 30.0, 20.0, 20.0]}
    )
    portfolio_table = st.data_editor(
        default_portfolio,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance ticker symbol"),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
    )
    total_weight = pd.to_numeric(portfolio_table.get("Weight (%)"), errors="coerce").sum()
    st.caption(f"Total weight: {total_weight:.1f}%")
    st.divider()
    initial_investment = st.number_input("Initial investment ($)", min_value=1_000, value=100_000, step=5_000)
    target_value = st.number_input("Target portfolio value ($)", min_value=1_000, value=115_000, step=5_000)
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    start_date = st.date_input("Historical data start", value=date(2018, 1, 1), max_value=date.today())
    horizon_label = st.selectbox("Simulation horizon", ["3 months", "6 months", "1 year", "2 years", "3 years"], index=2)
    horizon_map = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504, "3 years": 756}
    simulations = st.select_slider("Number of simulations", [1_000, 5_000, 10_000, 25_000], value=10_000)
    method = st.selectbox(
        "Simulation model",
        ["Regime Switching", "Geometric Brownian Motion", "Historical Bootstrap", "Student-t"],
        help="Regime Switching explicitly models bull, normal, crisis, and recovery states.",
    )
    with st.expander("Liquidity cascade assumptions"):
        redemption_probability = st.slider("Redemption probability after a breach", 0.0, 1.0, 0.35, 0.05)
        redemption_pct = st.slider("Investor redemption (%)", 0, 40, 12, 1) / 100
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
        and click <b>Run Fragility Analysis</b>.</div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


if run_button:
    try:
        tickers, weights = validate_portfolio(portfolio_table)
        if benchmark in tickers:
            st.info("The benchmark is also held in the portfolio; relative analytics will still be calculated.")
        config = AnalysisConfig(
            tickers=tickers,
            weights=weights,
            benchmark=benchmark,
            start_date=start_date,
            initial_investment=float(initial_investment),
            simulation_days=horizon_map[horizon_label],
            simulations=int(simulations),
            confidence=float(confidence),
            target_value=float(target_value),
            risk_free_rate=float(risk_free_rate),
            method=method,
            seed=int(seed),
            student_df=int(student_df),
            redemption_probability=float(redemption_probability),
            redemption_pct=float(redemption_pct),
            margin_call_pct=float(margin_call_pct),
            margin_trigger=float(margin_trigger),
            cash_buffer_pct=float(cash_buffer_pct),
        )
        with st.spinner("Downloading market data and running regime, liquidity, and stress models..."):
            prices, benchmark_prices, volumes = cached_market_data(tickers, benchmark, start_date)
            result = run_analysis(config, prices, benchmark_prices, volumes)
        st.session_state.analysis = result
        st.session_state.analysis_config = config
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
    f"using {config.method}."
)
render_dashboard(result, config)
