"""English, results-first Streamlit interface for Portfolio Fragility Lab."""

from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

st.markdown(
    """
    <style>
    :root { --ink:#e9eef6; --muted:#91a0b6; --cyan:#43d5c7; --navy:#07111f; --orange:#f2a65a; }
    .stApp { background:linear-gradient(135deg,#07111f 0%,#0b1727 55%,#101c2c 100%); }
    [data-testid="stSidebar"] { background:#081321; border-right:1px solid #1e3046; }
    .block-container { padding-top:2rem; padding-bottom:4rem; max-width:1540px; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-0.025em; }
    p,.stCaption { color:var(--muted); }
    [data-testid="stMetric"] { background:rgba(15,31,50,.88); border:1px solid #20354d;
      border-radius:14px; padding:17px 19px; box-shadow:0 12px 30px rgba(0,0,0,.18); }
    [data-testid="stMetricLabel"] { color:#91a0b6; }
    [data-testid="stMetricValue"] { color:#f5f8fc; }
    .hero { padding:16px 0 22px; }
    .eyebrow { color:#43d5c7; font-size:.78rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2.2rem,4vw,4rem); line-height:1; margin:.5rem 0 1rem; }
    .hero p { max-width:850px; font-size:1.05rem; }
    .notice,.method-note { padding:14px 16px; border:1px solid #24435a; border-radius:12px; background:#0c1c2c; color:#a9b8cb; }
    .score-low { color:#43d5c7; } .score-mid { color:#f2a65a; } .score-high { color:#ff6b6b; }
    .stButton>button { border-radius:10px; font-weight:700; min-height:46px; }
    [data-testid="stDataFrame"] { border:1px solid #20354d; border-radius:12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], benchmark: str, start: date):
    return download_market_data(tickers, benchmark, start)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def base_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,.55)",
        font={"color": "#b9c5d6"},
        margin={"l": 35, "r": 20, "t": 58, "b": 40},
        hovermode="x unified",
    )
    return figure


def style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,.55)",
        font={"color": "#b9c5d6"},
        margin={"l": 35, "r": 20, "t": 58, "b": 40},
    )
    return figure


def fan_chart(percentiles: pd.DataFrame) -> go.Figure:
    x = np.arange(len(percentiles))
    figure = base_figure("Simulated Portfolio Value Range")
    figure.add_trace(go.Scatter(x=x, y=percentiles["P95"], line={"width": 0}, showlegend=False))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P5"], fill="tonexty", fillcolor="rgba(67,213,199,.12)", line={"width": 0}, name="5th–95th percentile"))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P75"], line={"width": 0}, showlegend=False))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P25"], fill="tonexty", fillcolor="rgba(67,213,199,.25)", line={"width": 0}, name="25th–75th percentile"))
    figure.add_trace(go.Scatter(x=x, y=percentiles["Median"], line={"color": "#43d5c7", "width": 2.4}, name="Median"))
    figure.update_yaxes(tickprefix="$", tickformat=",")
    figure.update_xaxes(title="Trading day")
    return figure


def terminal_chart(result, config) -> go.Figure:
    figure = px.histogram(
        x=result.terminal_values,
        nbins=60,
        labels={"x": "Terminal portfolio value", "y": "Simulation paths"},
        color_discrete_sequence=["#43d5c7"],
        opacity=0.78,
        title="Terminal Wealth Distribution After Liquidity Events",
    )
    style_figure(figure)
    for value, label, color in [
        (config.initial_investment, "Initial", "#f2a65a"),
        (result.metrics["median_terminal"], "Median", "#ffffff"),
        (config.target_value, "Target", "#ad7cff"),
    ]:
        figure.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=label)
    figure.update_xaxes(tickprefix="$", tickformat=",")
    figure.update_layout(showlegend=False)
    return figure


def normalized_chart(result, benchmark: str) -> go.Figure:
    normalized = result.prices / result.prices.iloc[0] * 100
    benchmark_series = (1 + result.benchmark_returns).cumprod() * 100
    figure = base_figure("Historical Growth of 100")
    for column in normalized:
        figure.add_trace(go.Scatter(x=normalized.index, y=normalized[column], name=column, line={"width": 1.5}))
    figure.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, name=benchmark, line={"color": "#ffffff", "width": 2.5, "dash": "dot"}))
    return figure


def portfolio_benchmark_chart(result, benchmark: str) -> go.Figure:
    aligned = pd.concat(
        [result.portfolio_returns.rename("Portfolio"), result.benchmark_returns.rename(benchmark)],
        axis=1,
    ).dropna()
    growth = (1 + aligned).cumprod() * 100
    figure = base_figure("Portfolio vs Benchmark — Growth of 100")
    figure.add_trace(
        go.Scatter(
            x=growth.index,
            y=growth["Portfolio"],
            name="Portfolio",
            line={"color": "#43d5c7", "width": 2.6},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=growth.index,
            y=growth[benchmark],
            name=benchmark,
            line={"color": "#ffffff", "width": 2.1, "dash": "dot"},
        )
    )
    return figure


def score_label(score: float) -> tuple[str, str]:
    if score < 35:
        return "Resilient", "score-low"
    if score < 65:
        return "Watch", "score-mid"
    return "Fragile", "score-high"


def download_workbook(result, config) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([config.__dict__]).to_excel(writer, "Configuration", index=False)
        result.prices.to_excel(writer, "Prices")
        result.asset_returns.to_excel(writer, "Returns")
        pd.Series(result.metrics, name="Value").to_excel(writer, "Risk Metrics")
        result.risk_contributions.to_excel(writer, "Risk Contributions", index=False)
        result.liquidity.to_excel(writer, "Liquidity", index=False)
        result.stress_tests.to_excel(writer, "Stress Tests", index=False)
        result.regime_summary.to_excel(writer, "Regimes", index=False)
        result.fragility_components.to_excel(writer, "Fragility Score", index=False)
        result.allocation_comparison.to_excel(writer, "Allocation Comparison", index=False)
        pd.Series(result.terminal_values, name="Terminal Value").to_excel(writer, "Terminal Values", index=False)
    return output.getvalue()


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


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Regime-aware portfolio stress testing</div>
      <h1>Portfolio Fragility Lab</h1>
      <p>Test how a portfolio behaves when volatility jumps, correlations converge, investors redeem, margin calls arrive, and liquid assets must be sold into stressed markets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run_button and "analysis" not in st.session_state:
    st.markdown('<div class="notice">Build your portfolio in the left panel, choose a model, and click <b>Run Fragility Analysis</b>.</div>', unsafe_allow_html=True)
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
metrics = result.metrics
label, score_class = score_label(metrics["fragility_score"])

st.success(f"Completed {config.simulations:,} paths for {', '.join(config.tickers)} using {config.method}.")

metric_columns = st.columns(6)
metric_columns[0].metric("Fragility Score", f"{metrics['fragility_score']:.0f}/100", label)
metric_columns[1].metric("Expected Terminal", money(metrics["expected_terminal"]), money(metrics["expected_terminal"] - config.initial_investment))
metric_columns[2].metric("Loss Probability", pct(metrics["probability_loss"]), delta_color="inverse")
metric_columns[3].metric("Liquidity Shortfall", pct(metrics["liquidity_shortfall_probability"]), delta_color="inverse")
metric_columns[4].metric("Forced Sale", pct(metrics["forced_sale_probability"]), delta_color="inverse")
recovery_display = f"{metrics['median_recovery_days']:.0f} days" if np.isfinite(metrics["median_recovery_days"]) else "Not recovered"
metric_columns[5].metric("Median Recovery", recovery_display)

(
    overview_tab,
    core_analytics_tab,
    risk_analysis_tab,
    historical_performance_tab,
    regimes_tab,
    liquidity_tab,
    stress_tab,
    allocation_tab,
    simulation_tab,
    report_tab,
) = st.tabs(
    [
        "Executive View",
        "Core Analytics",
        "Risk Analysis",
        "Historical Performance",
        "Market Regimes",
        "Liquidity Cascade",
        "Stress Scenarios",
        "Resilient Allocation",
        "Simulation",
        "Report",
    ]
)

with overview_tab:
    left, right = st.columns([1.55, 1])
    with left:
        st.plotly_chart(fan_chart(result.path_percentiles), width="stretch", key="executive_fan_chart")
    with right:
        st.subheader("What the model found")
        st.write(
            f"The portfolio has a **{pct(metrics['probability_loss'])}** probability of ending below its initial value and a "
            f"**{pct(metrics['probability_target'])}** probability of reaching {money(config.target_value)}."
        )
        st.write(
            f"At {config.confidence:.0%} confidence, Monte Carlo VaR is **{money(metrics['var_currency'])}** and Expected Shortfall is "
            f"**{money(metrics['es_currency'])}** after modeled liquidity events."
        )
        st.write(
            f"The Fragility Score is **{metrics['fragility_score']:.0f}/100 ({label})**. Crisis correlation rises from "
            f"**{metrics['normal_correlation']:.2f}** to **{metrics['crisis_correlation']:.2f}** in the stressed dependence model."
        )
        st.caption("Outputs are scenario estimates, not guaranteed forecasts.")
    component_figure = px.bar(
        result.fragility_components,
        x="Risk Score",
        y="Component",
        orientation="h",
        color="Risk Score",
        color_continuous_scale=["#43d5c7", "#f2a65a", "#ff6b6b"],
        range_color=[0, 100],
        title="Fragility Score Components",
    )
    style_figure(component_figure).update_layout(coloraxis_showscale=False)
    st.plotly_chart(component_figure, width="stretch", key="executive_fragility_components")

with core_analytics_tab:
    st.subheader("Return, downside risk, and benchmark analytics")
    core_metrics = st.columns(6)
    core_metrics[0].metric(f"{config.confidence:.0%} VaR", money(metrics["var_currency"]))
    core_metrics[1].metric("Expected Shortfall", money(metrics["es_currency"]))
    core_metrics[2].metric("Annual Return", pct(metrics["historical_return"]))
    core_metrics[3].metric("Annual Volatility", pct(metrics["historical_volatility"]))
    core_metrics[4].metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    core_metrics[5].metric("Benchmark Beta", f"{metrics['beta']:.2f}")

    probability_metrics = st.columns(6)
    probability_metrics[0].metric("Loss Probability", pct(metrics["probability_loss"]))
    probability_metrics[1].metric("Target Probability", pct(metrics["probability_target"]))
    probability_metrics[2].metric("Historical Max Drawdown", pct(metrics["historical_max_drawdown"]))
    probability_metrics[3].metric(f"{config.benchmark} Return", pct(metrics["benchmark_return"]))
    probability_metrics[4].metric(f"{config.benchmark} Volatility", pct(metrics["benchmark_volatility"]))
    probability_metrics[5].metric("Portfolio–Benchmark Corr.", f"{metrics['benchmark_correlation']:.2f}")

    st.plotly_chart(
        portfolio_benchmark_chart(result, config.benchmark),
        width="stretch",
        key="core_portfolio_benchmark",
    )

    left, right = st.columns([1.35, 1])
    with left:
        contribution_figure = px.bar(
            result.risk_contributions,
            x="Ticker",
            y="Risk Contribution",
            color="Risk Contribution",
            color_continuous_scale=["#183047", "#43d5c7"],
            title="Asset Contribution to Portfolio Volatility",
        )
        style_figure(contribution_figure).update_layout(coloraxis_showscale=False)
        contribution_figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(contribution_figure, width="stretch", key="core_risk_contribution")
    with right:
        benchmark_table = pd.DataFrame(
            {
                "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown"],
                "Portfolio": [
                    metrics["historical_return"],
                    metrics["historical_volatility"],
                    metrics["sharpe"],
                    metrics["historical_max_drawdown"],
                ],
                config.benchmark: [
                    metrics["benchmark_return"],
                    metrics["benchmark_volatility"],
                    metrics["benchmark_sharpe"],
                    metrics["benchmark_max_drawdown"],
                ],
            }
        )
        display_table = benchmark_table.copy()
        for row in [0, 1, 3]:
            display_table.loc[row, ["Portfolio", config.benchmark]] = display_table.loc[
                row, ["Portfolio", config.benchmark]
            ].map(lambda value: f"{value:.1%}")
        display_table.loc[2, ["Portfolio", config.benchmark]] = display_table.loc[
            2, ["Portfolio", config.benchmark]
        ].map(lambda value: f"{value:.2f}")
        st.dataframe(display_table, hide_index=True, width="stretch")
        st.metric("Tracking Error", pct(metrics["tracking_error"]))
        information_ratio = metrics["information_ratio"]
        st.metric(
            "Information Ratio",
            f"{information_ratio:.2f}" if np.isfinite(information_ratio) else "N/A",
        )

    st.dataframe(
        result.risk_contributions.style.format(
            {"Weight": "{:.1%}", "Risk Contribution": "{:.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Available simulation models: Regime Switching, Geometric Brownian Motion, "
        "Historical Bootstrap, and Student-t. Change the active model in the sidebar."
    )

with risk_analysis_tab:
    st.subheader("Downside risk and portfolio risk decomposition")
    risk_metrics = st.columns(6)
    risk_metrics[0].metric(f"{config.confidence:.0%} Value at Risk", money(metrics["var_currency"]))
    risk_metrics[1].metric("Expected Shortfall", money(metrics["es_currency"]))
    risk_metrics[2].metric("Median Simulated Drawdown", pct(metrics["median_simulated_drawdown"]))
    risk_metrics[3].metric("Historical Max Drawdown", pct(metrics["historical_max_drawdown"]))
    risk_metrics[4].metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    risk_metrics[5].metric("Benchmark Beta", f"{metrics['beta']:.2f}")

    terminal_returns = result.terminal_values / config.initial_investment - 1
    var_cutoff = -metrics["var_currency"] / config.initial_investment
    es_cutoff = -metrics["es_currency"] / config.initial_investment
    tail_figure = px.histogram(
        x=terminal_returns,
        nbins=65,
        labels={"x": "Terminal return", "y": "Simulation paths"},
        color_discrete_sequence=["#4c78a8"],
        opacity=0.82,
        title="Terminal Return Distribution — VaR and Expected Shortfall",
    )
    style_figure(tail_figure)
    tail_figure.add_vline(
        x=var_cutoff,
        line_dash="dash",
        line_color="#f2a65a",
        annotation_text="VaR cutoff",
    )
    tail_figure.add_vline(
        x=es_cutoff,
        line_dash="dot",
        line_color="#ff6b6b",
        annotation_text="Expected Shortfall",
    )
    tail_figure.update_xaxes(tickformat=".0%")
    st.plotly_chart(tail_figure, width="stretch", key="risk_tail_distribution")

    left, right = st.columns([1.4, 1])
    with left:
        contribution_figure = px.bar(
            result.risk_contributions,
            x="Ticker",
            y="Risk Contribution",
            color="Risk Contribution",
            color_continuous_scale=["#183047", "#43d5c7"],
            title="Asset Risk Contribution",
        )
        style_figure(contribution_figure).update_layout(coloraxis_showscale=False)
        contribution_figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(contribution_figure, width="stretch", key="risk_contribution")
    with right:
        probability_frame = pd.DataFrame(
            {
                "Outcome": ["Loss", "Positive return", "Target reached"],
                "Probability": [
                    metrics["probability_loss"],
                    1 - metrics["probability_loss"],
                    metrics["probability_target"],
                ],
            }
        )
        probability_figure = px.bar(
            probability_frame,
            x="Outcome",
            y="Probability",
            color="Outcome",
            text="Probability",
            title="Outcome Probabilities",
            color_discrete_sequence=["#ff6b6b", "#43d5c7", "#ad7cff"],
        )
        style_figure(probability_figure).update_layout(showlegend=False)
        probability_figure.update_yaxes(tickformat=".0%", range=[0, 1])
        probability_figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(probability_figure, width="stretch", key="risk_outcome_probabilities")

    st.dataframe(
        result.risk_contributions.style.format(
            {"Weight": "{:.1%}", "Risk Contribution": "{:.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )

with historical_performance_tab:
    st.subheader("Historical portfolio and benchmark performance")
    historical_metrics = st.columns(6)
    historical_metrics[0].metric("Portfolio Annual Return", pct(metrics["historical_return"]))
    historical_metrics[1].metric("Portfolio Volatility", pct(metrics["historical_volatility"]))
    historical_metrics[2].metric("Portfolio Sharpe", f"{metrics['sharpe']:.2f}")
    historical_metrics[3].metric("Maximum Drawdown", pct(metrics["historical_max_drawdown"]))
    historical_metrics[4].metric("Beta", f"{metrics['beta']:.2f}")
    historical_metrics[5].metric("Tracking Error", pct(metrics["tracking_error"]))

    st.plotly_chart(
        portfolio_benchmark_chart(result, config.benchmark),
        width="stretch",
        key="historical_portfolio_benchmark",
    )
    st.plotly_chart(
        normalized_chart(result, config.benchmark),
        width="stretch",
        key="historical_normalized_assets",
    )

    benchmark_comparison = pd.DataFrame(
        {
            "Metric": [
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Correlation",
                "Beta",
                "Tracking Error",
                "Information Ratio",
            ],
            "Portfolio": [
                f"{metrics['historical_return']:.1%}",
                f"{metrics['historical_volatility']:.1%}",
                f"{metrics['sharpe']:.2f}",
                f"{metrics['historical_max_drawdown']:.1%}",
                f"{metrics['benchmark_correlation']:.2f}",
                f"{metrics['beta']:.2f}",
                f"{metrics['tracking_error']:.1%}",
                f"{metrics['information_ratio']:.2f}" if np.isfinite(metrics["information_ratio"]) else "N/A",
            ],
            config.benchmark: [
                f"{metrics['benchmark_return']:.1%}",
                f"{metrics['benchmark_volatility']:.1%}",
                f"{metrics['benchmark_sharpe']:.2f}",
                f"{metrics['benchmark_max_drawdown']:.1%}",
                "1.00",
                "1.00",
                "0.0%",
                "N/A",
            ],
        }
    )
    st.dataframe(benchmark_comparison, hide_index=True, width="stretch")

with regimes_tab:
    col1, col2 = st.columns([1, 1.4])
    with col1:
        regime_figure = px.pie(
            result.regime_summary,
            names="Regime",
            values="Share of Simulated Days",
            hole=0.62,
            color="Regime",
            color_discrete_map={"Bull": "#43d5c7", "Normal": "#4c78a8", "Crisis": "#ff6b6b", "Recovery": "#f2a65a"},
            title="Simulated Regime Mix",
        )
        style_figure(regime_figure)
        st.plotly_chart(regime_figure, width="stretch", key="regime_probabilities")
    with col2:
        correlation_figure = go.Figure(
            go.Bar(
                x=["Normal market", "Crisis regime"],
                y=[metrics["normal_correlation"], metrics["crisis_correlation"]],
                marker_color=["#4c78a8", "#ff6b6b"],
                text=[f"{metrics['normal_correlation']:.2f}", f"{metrics['crisis_correlation']:.2f}"],
                textposition="outside",
            )
        )
        style_figure(correlation_figure).update_layout(title="Average Cross-Asset Correlation", yaxis_range=[-0.2, 1])
        st.plotly_chart(correlation_figure, width="stretch", key="regime_correlations")
    st.dataframe(
        result.regime_summary.style.format({"Share of Simulated Days": "{:.1%}", "Volatility Multiplier": "{:.2f}x", "Correlation Blend": "{:.0%}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown('<div class="method-note">The regime engine uses a Markov transition process. Crisis days apply a 2.5× volatility multiplier and blend correlations toward one; recovery days retain elevated volatility before returning to normal.</div>', unsafe_allow_html=True)

with liquidity_tab:
    liquidity_columns = st.columns(5)
    liquidity_columns[0].metric("Portfolio Liquidity Score", f"{metrics['weighted_liquidity_score']:.0f}/100")
    liquidity_columns[1].metric("Redemption Event", pct(metrics["redemption_probability_realized"]))
    liquidity_columns[2].metric("Margin Call", pct(metrics["margin_call_probability"]))
    liquidity_columns[3].metric("Shortfall Probability", pct(metrics["liquidity_shortfall_probability"]))
    liquidity_columns[4].metric("Conditional Sale Cost", money(metrics["conditional_forced_sale_cost"]))
    liquidity_figure = px.scatter(
        result.liquidity,
        x="Median Dollar Volume",
        y="Liquidity Score",
        size="5% ADV Capacity",
        color="Estimated Spread (bps)",
        hover_name="Ticker",
        color_continuous_scale="RdYlGn_r",
        log_x=True,
        title="Asset Liquidity Map",
    )
    style_figure(liquidity_figure)
    st.plotly_chart(liquidity_figure, width="stretch", key="liquidity_scores")
    cascade_frame = pd.DataFrame(
        {
            "Event": ["Redemption", "Margin call", "Forced sale", "Liquidity shortfall"],
            "Probability": [metrics["redemption_probability_realized"], metrics["margin_call_probability"], metrics["forced_sale_probability"], metrics["liquidity_shortfall_probability"]],
        }
    )
    cascade_figure = px.bar(cascade_frame, x="Event", y="Probability", color="Probability", color_continuous_scale=["#43d5c7", "#ff6b6b"], title="Liquidity Cascade Event Probabilities")
    style_figure(cascade_figure).update_layout(coloraxis_showscale=False)
    cascade_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(cascade_figure, width="stretch", key="liquidity_cascade")
    st.dataframe(
        result.liquidity.style.format({"Liquidity Score": "{:.1f}", "Median Dollar Volume": "${:,.0f}", "Estimated Spread (bps)": "{:.1f}", "5% ADV Capacity": "${:,.0f}"}),
        hide_index=True,
        width="stretch",
    )

with stress_tab:
    stress_plot = px.bar(
        result.stress_tests,
        x="Scenario",
        y="Net Portfolio Shock",
        color="Net Portfolio Shock",
        color_continuous_scale=["#ff6b6b", "#f2a65a", "#43d5c7"],
        text="Net Portfolio Shock",
        title="Historical-Style Stress Scenario Losses",
    )
    style_figure(stress_plot).update_layout(coloraxis_showscale=False)
    stress_plot.update_yaxes(tickformat=".0%")
    stress_plot.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    st.plotly_chart(stress_plot, width="stretch", key="stress_scenarios")
    st.dataframe(
        result.stress_tests.style.format(
            {
                "Gross Portfolio Shock": "{:.1%}",
                "Liquidation Cost": "{:.2%}",
                "Net Portfolio Shock": "{:.1%}",
                "Stressed Value": "${:,.0f}",
                "Correlation Assumption": "{:.2f}",
                "Liquidity Capacity Multiplier": "{:.0%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Scenarios are transparent educational approximations calibrated to broad historical market moves, not official regulatory loss projections.")

with allocation_tab:
    allocation = pd.DataFrame(
        {
            "Ticker": list(config.tickers) * 2,
            "Weight": list(config.weights) + list(result.resilient_weights),
            "Portfolio": ["Current"] * len(config.tickers) + ["Crisis-Resilient"] * len(config.tickers),
        }
    )
    allocation_figure = px.bar(
        allocation,
        x="Portfolio",
        y="Weight",
        color="Ticker",
        title="Current vs Crisis-Resilient Allocation",
    )
    style_figure(allocation_figure)
    allocation_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(allocation_figure, width="stretch", key="resilient_allocation")
    comparison_figure = px.bar(
        result.allocation_comparison,
        x="Portfolio",
        y="Worst Stress Loss",
        color="Portfolio",
        text="Worst Stress Loss",
        title="Worst Scenario Loss Comparison",
        color_discrete_sequence=["#ff6b6b", "#43d5c7"],
    )
    style_figure(comparison_figure).update_layout(showlegend=False)
    comparison_figure.update_yaxes(tickformat=".0%")
    comparison_figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    st.plotly_chart(comparison_figure, width="stretch", key="resilient_comparison")
    st.dataframe(
        result.allocation_comparison.style.format(
            {
                "Historical Return": "{:.1%}",
                "Historical Volatility": "{:.1%}",
                "Average Stress Loss": "{:.1%}",
                "Worst Stress Loss": "{:.1%}",
                "Weighted Liquidity Score": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("The crisis-resilient allocation is an analytical benchmark based on inverse volatility, beta penalties, liquidity scores, and a 45% position cap—not a recommendation.")

with simulation_tab:
    st.plotly_chart(terminal_chart(result, config), width="stretch", key="simulation_terminal")
    sample_count = min(100, result.paths.shape[1])
    path_figure = base_figure("Sample Portfolio Paths After Liquidity Events")
    for index in range(sample_count):
        path_figure.add_trace(go.Scatter(y=result.paths[:, index], mode="lines", line={"width": 0.7, "color": "rgba(67,213,199,.12)"}, showlegend=False, hoverinfo="skip"))
    path_figure.update_yaxes(tickprefix="$", tickformat=",")
    path_figure.update_xaxes(title="Trading day")
    st.plotly_chart(path_figure, width="stretch", key="simulation_paths")
    st.plotly_chart(
        normalized_chart(result, config.benchmark),
        width="stretch",
        key="simulation_normalized_assets",
    )

with report_tab:
    st.subheader("Download the complete analysis")
    st.write("The workbook includes assumptions, historical data, simulation results, liquidity scores, regime statistics, stress tests, Fragility Score components, and allocation comparisons.")
    st.download_button(
        "Download Excel report",
        data=download_workbook(result, config),
        file_name="portfolio_fragility_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    st.download_button(
        "Download terminal scenarios (CSV)",
        data=pd.DataFrame({"terminal_value": result.terminal_values}).to_csv(index=False),
        file_name="terminal_scenarios.csv",
        mime="text/csv",
    )
    st.markdown("---")
    st.caption("Limitations: historical parameters may not represent the future; liquidity and spread estimates are proxies; stress scenarios are stylized; taxes, options, and security-specific fundamentals are excluded. Educational use only.")
