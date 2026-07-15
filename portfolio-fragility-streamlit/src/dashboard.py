"""Institutional Streamlit presentation layer for Portfolio Fragility Lab."""

from __future__ import annotations

from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .analytics import (
    REGIME_TRANSITION_MATRIX,
    correlation_interpretation,
    correlation_matrices,
    efficient_frontier,
    historical_drawdowns,
    regime_assumptions,
    risk_contribution_table,
    rolling_analytics,
    simulation_statistics,
)
from .reporting import (
    DISCLAIMER,
    build_excel_report,
    build_pdf_report,
    build_powerpoint_report,
    risk_metrics_csv,
)


COLORS = {
    "background": "#050505",
    "secondary": "#0B0D10",
    "card": "#111419",
    "elevated": "#151A21",
    "border": "#252B33",
    "text": "#F3F4F6",
    "secondary_text": "#A8B0BC",
    "muted": "#737C89",
    "accent": "#22C7B8",
    "positive": "#2ECC8F",
    "warning": "#E7A94B",
    "negative": "#F05D64",
    "info": "#5B8DEF",
    "purple": "#8B7CF6",
}

ASSET_COLORS = ["#22C7B8", "#5B8DEF", "#8B7CF6", "#E7A94B", "#2ECC8F", "#A8B0BC", "#C779D0", "#4FA3A5"]


def apply_institutional_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
          --bg:{COLORS['background']}; --surface:{COLORS['secondary']}; --card:{COLORS['card']};
          --elevated:{COLORS['elevated']}; --border:{COLORS['border']}; --text:{COLORS['text']};
          --secondary-text:{COLORS['secondary_text']}; --muted:{COLORS['muted']}; --accent:{COLORS['accent']};
          --positive:{COLORS['positive']}; --warning:{COLORS['warning']}; --negative:{COLORS['negative']};
        }}
        html, body, [class*="css"], .stApp {{
          font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        }}
        .stApp {{ background:var(--bg); color:var(--text); }}
        [data-testid="stAppViewContainer"] {{ background:var(--bg); }}
        [data-testid="stHeader"] {{ background:rgba(5,5,5,.92); }}
        [data-testid="stSidebar"] {{ background:#090A0C; border-right:1px solid var(--border); }}
        .block-container {{ padding-top:1.35rem; padding-bottom:3rem; max-width:1600px; }}
        h1 {{ font-size:clamp(1.85rem,2.6vw,2.65rem)!important; line-height:1.08!important; letter-spacing:-.035em!important; }}
        h2 {{ font-size:1.35rem!important; letter-spacing:-.02em!important; margin-top:.7rem!important; }}
        h3 {{ font-size:1.02rem!important; letter-spacing:-.01em!important; }}
        h1,h2,h3,h4 {{ color:var(--text)!important; }}
        p,.stCaption,[data-testid="stCaptionContainer"] {{ color:var(--secondary-text); }}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:.25rem; overflow-x:auto; scrollbar-width:thin; }}
        [data-testid="stTabs"] button[role="tab"] {{
          color:var(--muted); background:transparent; border-radius:8px 8px 0 0; padding:.55rem .85rem;
          white-space:nowrap; font-size:.88rem;
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{ color:var(--text); background:var(--card); }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:var(--accent); }}
        [data-testid="stMetric"] {{
          background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px;
          min-width:0; overflow:visible; box-shadow:none;
        }}
        [data-testid="stMetricLabel"] {{ color:var(--muted); }}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
          color:var(--text); font-variant-numeric:tabular-nums; white-space:nowrap!important;
          overflow:visible!important; text-overflow:clip!important;
        }}
        [data-testid="stMetricValue"] > div {{ font-size:clamp(1.25rem,1.6vw,1.9rem)!important; line-height:1.15!important; }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:10px; overflow:auto; }}
        [data-testid="stDataFrame"] [role="columnheader"] {{ background:var(--elevated)!important; color:var(--text)!important; }}
        [data-testid="stDataFrame"] [role="gridcell"] {{ min-height:30px!important; padding-top:4px!important; padding-bottom:4px!important; }}
        .stButton>button, .stDownloadButton>button {{
          border-radius:9px; min-height:40px; font-weight:650; border:1px solid var(--border);
          background:var(--card); color:var(--text);
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{ border-color:var(--accent); color:var(--accent); }}
        .stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"],
        [data-testid="stBaseButton-primary"] {{ background:var(--accent)!important; color:#041311!important; border-color:var(--accent)!important; }}
        [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span {{ color:#041311!important; }}
        [data-testid="stExpander"], [data-testid="stExpander"] details,
        [data-testid="stExpander"] details > div, [data-testid="stExpander"] summary,
        [data-testid="stExpanderDetails"] {{
          background:#0B0D10!important; color:#F3F4F6!important;
        }}
        [data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
        [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] label, [data-testid="stExpander"] [data-testid="stWidgetLabel"] p {{
          color:var(--text)!important;
        }}
        [data-testid="stExpander"] summary svg {{ fill:var(--secondary-text)!important; color:var(--secondary-text)!important; }}
        .hero {{ padding:.25rem 0 .9rem; border-bottom:1px solid var(--border); margin-bottom:1rem; }}
        .eyebrow {{ color:var(--accent); font-size:.72rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }}
        .hero p {{ max-width:900px; margin:.45rem 0 0; font-size:.95rem; }}
        .section-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin:1.2rem 0 .55rem; }}
        .section-head h3 {{ margin:0; font-size:1.04rem!important; }}
        .section-head p {{ margin:0; font-size:.8rem; color:var(--muted); }}
        .kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:.45rem 0 1rem; }}
        .kpi-card {{ background:var(--card); border:1px solid var(--border); border-radius:11px; padding:.78rem .9rem; min-width:0; }}
        .kpi-top {{ display:flex; justify-content:space-between; gap:.5rem; align-items:center; }}
        .kpi-label {{ color:var(--muted); font-size:.72rem; line-height:1.2; white-space:normal; }}
        .kpi-value {{ color:var(--text); font-size:clamp(1.12rem,1.55vw,1.75rem); font-weight:680; line-height:1.15; margin-top:.38rem; font-variant-numeric:tabular-nums; white-space:normal; overflow-wrap:anywhere; }}
        .badge {{ font-size:.62rem; line-height:1; font-weight:750; padding:.3rem .42rem; border-radius:999px; border:1px solid currentColor; white-space:nowrap; }}
        .positive {{ color:var(--positive); }} .warning {{ color:var(--warning); }} .negative {{ color:var(--negative); }} .neutral {{ color:var(--accent); }}
        .insight-card {{ background:var(--card); border:1px solid var(--border); border-radius:11px; padding:.9rem 1rem; height:100%; }}
        .insight-card h4 {{ margin:0 0 .45rem; font-size:.86rem; }}
        .insight-card ul {{ margin:.25rem 0 .45rem; padding-left:1.15rem; color:var(--secondary-text); }}
        .insight-card li {{ margin:.28rem 0; font-size:.84rem; }}
        .callout {{ border-left:3px solid var(--accent); background:var(--card); padding:.72rem .9rem; border-radius:0 9px 9px 0; color:var(--secondary-text); font-size:.84rem; }}
        .method-note {{ padding:.72rem .85rem; border:1px solid var(--border); border-radius:9px; background:#0B0D10!important; color:#A8B0BC!important; font-size:.8rem; }}
        .mini-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.5rem; }}
        .mini-stat {{ background:var(--surface); border:1px solid var(--border); border-radius:9px; padding:.65rem .7rem; }}
        .mini-stat span {{ display:block; color:var(--muted); font-size:.67rem; }}
        .mini-stat strong {{ display:block; color:var(--text); font-size:1rem; margin-top:.22rem; font-variant-numeric:tabular-nums; }}
        .footer-note {{ margin-top:1.5rem; padding-top:.75rem; border-top:1px solid var(--border); color:var(--muted); font-size:.72rem; }}
        @media (max-width:1280px) {{
          .kpi-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
          .kpi-value {{ font-size:1.3rem; }}
        }}
        @media (max-width:720px) {{
          .block-container {{ padding-left:.8rem; padding-right:.8rem; }}
          .kpi-grid,.mini-grid {{ grid-template-columns:1fr 1fr; }}
          [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def money(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"${value:,.0f}"


def pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.1%}"


def probability(value: float) -> str:
    """Format probabilities without rounding a near-certain result to 100%."""
    if not np.isfinite(value):
        return "N/A"
    value = float(np.clip(value, 0, 1))
    if 0 < value < 0.0005:
        return "<0.1%"
    if 0.9995 <= value < 1:
        return ">99.9%"
    return f"{value:.1%}"


def probability_partition_labels(values: list[float]) -> list[str]:
    """Round a mutually exclusive partition to tenths that total exactly 100.0%."""
    probabilities = np.clip(np.asarray(values, dtype=float), 0, 1)
    total = probabilities.sum()
    probabilities = probabilities / total if total > 0 else np.zeros_like(probabilities)
    scaled = probabilities * 1000
    units = np.floor(scaled).astype(int)
    remainder = 1000 - int(units.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - units))
        units[order[:remainder]] += 1
    return [f"{unit / 10:.1f}%" for unit in units]


def ratio(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}"


def days(value: float) -> str:
    return "Not recovered" if not np.isfinite(value) else f"{value:.0f} days"


def compact_money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return money(value)


def score_status(score: float) -> tuple[str, str]:
    if score <= 25:
        return "Resilient", "positive"
    if score <= 45:
        return "Stable", "positive"
    if score <= 65:
        return "Watch", "warning"
    if score <= 80:
        return "Elevated", "warning"
    return "Critical", "negative"


def section_header(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><h3>{escape(title)}</h3><p>{escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_grid(items: list[dict[str, str]]) -> None:
    cards = []
    for item in items:
        cards.append(
            f'<div class="kpi-card" title="{escape(item.get("help", ""))}">'
            f'<div class="kpi-top"><span class="kpi-label">{escape(item["label"])}</span>'
            f'<span class="badge {item.get("tone", "neutral")}">{escape(item.get("badge", "Modeled"))}</span></div>'
            f'<div class="kpi-value">{escape(item["value"])}</div></div>'
        )
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def style_figure(figure: go.Figure, title: str | None = None, height: int = 410, hovermode: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 15, "color": COLORS["text"]}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["secondary"],
        font={"family": "Inter, system-ui, sans-serif", "color": COLORS["secondary_text"], "size": 11},
        height=height,
        margin={"l": 45, "r": 20, "t": 55 if title else 25, "b": 45},
        hovermode=hovermode,
        hoverlabel={"bgcolor": COLORS["elevated"], "bordercolor": COLORS["border"], "font_color": COLORS["text"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1, "bgcolor": "rgba(0,0,0,0)"},
        coloraxis_colorbar={"outlinewidth": 0, "thickness": 12},
    )
    figure.update_xaxes(gridcolor="#1D2229", zerolinecolor=COLORS["border"], linecolor=COLORS["border"])
    figure.update_yaxes(gridcolor="#1D2229", zerolinecolor=COLORS["border"], linecolor=COLORS["border"])
    return figure


def fan_chart(result, title: str = "Simulated Portfolio Value Range", height: int = 430) -> go.Figure:
    p = result.path_percentiles
    x = np.arange(len(p))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=p["P95"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=x, y=p["P5"], fill="tonexty", fillcolor="rgba(91,141,239,.13)", line={"width": 0}, name="5th–95th", hovertemplate="Day %{x}<br>P5: $%{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=x, y=p["P75"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=x, y=p["P25"], fill="tonexty", fillcolor="rgba(34,199,184,.23)", line={"width": 0}, name="25th–75th", hovertemplate="Day %{x}<br>P25: $%{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=x, y=p["Median"], line={"color": COLORS["accent"], "width": 2.4}, name="Median", hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>"))
    style_figure(figure, title, height, "x unified")
    figure.update_xaxes(title="Trading day")
    figure.update_yaxes(tickprefix="$", tickformat=",")
    return figure


def fragility_gauge(result, height: int = 300) -> go.Figure:
    score = result.metrics["fragility_score"]
    status, _ = score_status(score)
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 35, "color": COLORS["text"]}},
        title={"text": f"Fragility Score · {status}", "font": {"size": 14, "color": COLORS["secondary_text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": COLORS["muted"], "tickfont": {"size": 9}},
            "bar": {"color": COLORS["text"], "thickness": 0.18},
            "bgcolor": COLORS["secondary"], "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(46,204,143,.45)"},
                {"range": [25, 45], "color": "rgba(34,199,184,.40)"},
                {"range": [45, 65], "color": "rgba(231,169,75,.42)"},
                {"range": [65, 80], "color": "rgba(240,93,100,.35)"},
                {"range": [80, 100], "color": "rgba(240,93,100,.58)"},
            ],
        },
    ))
    style_figure(figure, None, height)
    figure.update_layout(margin={"l": 30, "r": 30, "t": 40, "b": 10})
    return figure


def allocation_donut(config) -> go.Figure:
    figure = go.Figure(go.Pie(
        labels=list(config.tickers), values=list(config.weights), hole=.66,
        marker={"colors": ASSET_COLORS[:len(config.tickers)], "line": {"color": COLORS["card"], "width": 2}},
        textinfo="label+percent", textposition="outside", hovertemplate="%{label}<br>Weight: %{percent}<extra></extra>",
    ))
    style_figure(figure, "Current Allocation", 390)
    figure.update_layout(showlegend=False, margin={"l": 35, "r": 35, "t": 55, "b": 35})
    return figure


def benchmark_growth_chart(result, benchmark: str) -> go.Figure:
    aligned = pd.concat([result.portfolio_returns.rename("Portfolio"), result.benchmark_returns.rename(benchmark)], axis=1).dropna()
    growth = (1 + aligned).cumprod() * 100
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=growth.index, y=growth["Portfolio"], name="Portfolio", line={"color": COLORS["accent"], "width": 2.5}))
    figure.add_trace(go.Scatter(x=growth.index, y=growth[benchmark], name=benchmark, line={"color": COLORS["info"], "width": 2, "dash": "dot"}))
    return style_figure(figure, "Portfolio vs Benchmark — Growth of 100", 440, "x unified")


def asset_growth_chart(result, benchmark: str) -> go.Figure:
    normalized = result.prices / result.prices.iloc[0] * 100
    benchmark_series = (1 + result.benchmark_returns).cumprod() * 100
    figure = go.Figure()
    for index, column in enumerate(normalized.columns):
        figure.add_trace(go.Scatter(x=normalized.index, y=normalized[column], name=column, line={"color": ASSET_COLORS[index % len(ASSET_COLORS)], "width": 1.6}))
    figure.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, name=benchmark, line={"color": COLORS["text"], "width": 2, "dash": "dot"}))
    return style_figure(figure, "Asset-Level Historical Growth of 100", 440, "x unified")


def terminal_distribution(result, config) -> go.Figure:
    values = result.terminal_values
    figure = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color=COLORS["info"], opacity=.78, hovertemplate="Terminal wealth: $%{x:,.0f}<br>Paths: %{y}<extra></extra>"))
    references = [
        (config.initial_investment, "Initial", COLORS["warning"], -35),
        (config.target_value, "Target", COLORS["purple"], 35),
        (float(np.mean(values)), "Mean", COLORS["accent"], -55),
        (float(np.median(values)), "Median", COLORS["text"], 55),
    ]
    for value, label, color, ay in references:
        figure.add_vline(x=value, line_dash="dash", line_color=color, line_width=1.4)
        figure.add_annotation(x=value, y=1, yref="paper", text=label, showarrow=True, arrowhead=0, ax=0, ay=ay, font={"size": 10, "color": color}, bgcolor=COLORS["elevated"], bordercolor=color)
    style_figure(figure, "Terminal Wealth Distribution", 455)
    figure.update_xaxes(tickprefix="$", tickformat=",")
    figure.update_layout(showlegend=False, bargap=.03)
    return figure


def outcome_probability_chart(result, config) -> go.Figure:
    metrics = result.metrics
    probabilities = [
        metrics["probability_loss"],
        metrics["probability_below_target_without_loss"],
        metrics["probability_target"],
    ]
    labels = probability_partition_labels(probabilities)
    path_count = len(result.terminal_values)
    frame = pd.DataFrame({
        "Outcome": ["Loss", "No loss, below target", "Target reached"],
        "Probability": probabilities,
        "Display": labels,
        "Paths": [int(round(value * path_count)) for value in probabilities],
    })
    figure = px.bar(frame, x="Outcome", y="Probability", color="Outcome", text="Display",
                    custom_data=["Paths"],
                    color_discrete_map={"Loss": COLORS["negative"], "No loss, below target": COLORS["accent"], "Target reached": COLORS["purple"]})
    style_figure(figure, "Mutually Exclusive Terminal Outcomes — Total 100%", 390)
    figure.update_yaxes(tickformat=".0%", range=[0, 1.08])
    figure.update_traces(
        texttemplate="%{text}", textposition="outside",
        hovertemplate="%{x}: %{y:.2%}<br>Paths: %{customdata[0]:,}<extra></extra>",
    )
    figure.update_layout(showlegend=False)
    return figure


def render_executive(result, config) -> None:
    metrics = result.metrics
    status, status_tone = score_status(metrics["fragility_score"])
    relative_return = metrics["historical_return"] - metrics["benchmark_return"]
    kpi_grid([
        {"label": "Fragility Score", "value": f"{metrics['fragility_score']:.0f}/100", "badge": status, "tone": status_tone, "help": "Composite modeled sensitivity to drawdown, liquidity, forced sales, correlation instability, and concentration."},
        {"label": "Expected Terminal Wealth", "value": money(metrics["expected_terminal"]), "badge": "Modeled", "tone": "neutral", "help": "Mean terminal wealth across simulated paths after modeled liquidity events."},
        {"label": "Probability of Loss", "value": probability(metrics["probability_loss"]), "badge": "High" if metrics["probability_loss"] > .4 else "Watch" if metrics["probability_loss"] > .25 else "Low", "tone": "negative" if metrics["probability_loss"] > .4 else "warning", "help": f"{round(metrics['probability_loss'] * config.simulations):,} of {config.simulations:,} simulated paths ended below {money(config.initial_investment)}."},
        {"label": "Probability of Target", "value": probability(metrics["probability_target"]), "badge": "Target", "tone": "neutral", "help": f"{round(metrics['probability_target'] * config.simulations):,} of {config.simulations:,} simulated paths ended at or above the selected {money(config.target_value)} target."},
        {"label": f"Monte Carlo VaR ({config.confidence:.0%})", "value": money(metrics["var_currency"]), "badge": "Tail", "tone": "warning", "help": "Modeled loss threshold exceeded in the worst tail defined by the selected confidence level."},
        {"label": "Expected Shortfall", "value": money(metrics["es_currency"]), "badge": "Tail", "tone": "negative", "help": "Average modeled loss beyond the VaR threshold."},
        {"label": "Median Recovery Time", "value": days(metrics["median_recovery_days"]), "badge": "Modeled", "tone": "neutral", "help": "Median trading days from simulated trough to recovery of the prior peak, among recovered paths."},
        {"label": f"Return vs {config.benchmark}", "value": pct(relative_return), "badge": "Above" if relative_return >= 0 else "Below", "tone": "positive" if relative_return >= 0 else "negative", "help": "Historical annualized portfolio return minus benchmark return over the selected period."},
    ])
    driver_row = result.fragility_components.loc[result.fragility_components["Weighted Contribution"].idxmax()]
    risk_table = risk_contribution_table(result)
    largest_risk = risk_table.loc[risk_table["Risk Contribution"].idxmax()]
    left, right = st.columns([1.05, 1.45])
    with left:
        st.plotly_chart(fragility_gauge(result), use_container_width=True, key="executive_fragility_gauge")
    with right:
        findings = [
            f"Probability of loss: <b>{probability(metrics['probability_loss'])}</b>; target probability: <b>{probability(metrics['probability_target'])}</b>.",
            f"{config.confidence:.0%} VaR: <b>{money(metrics['var_currency'])}</b>; Expected Shortfall: <b>{money(metrics['es_currency'])}</b>.",
            f"Crisis correlation rises from <b>{metrics['normal_correlation']:.2f}</b> to <b>{metrics['crisis_correlation']:.2f}</b>.",
            f"Largest risk contributor: <b>{largest_risk['Ticker']}</b> at <b>{pct(largest_risk['Risk Contribution'])}</b> of modeled volatility.",
            f"Main fragility driver: <b>{driver_row['Component']}</b>.",
        ]
        recommendation = (
            "Modeled fragility is driven primarily by " + str(driver_row["Component"]).lower() +
            ". Compare the crisis-resilient allocation before accepting lower diversification in stressed markets."
        )
        st.markdown(
            '<div class="insight-card"><h4>Key Findings</h4><ul>' + "".join(f"<li>{item}</li>" for item in findings) +
            f'</ul><div class="callout"><b>Decision note:</b> {recommendation} This is an analytical comparison, not investment advice.</div></div>',
            unsafe_allow_html=True,
        )
    left, right = st.columns([1.55, .8])
    with left:
        st.plotly_chart(fan_chart(result), use_container_width=True, key="executive_fan")
    with right:
        st.plotly_chart(allocation_donut(config), use_container_width=True, key="executive_allocation")
    percentile_values = [metrics["p5_terminal"], metrics["median_terminal"], metrics["expected_terminal"], metrics["p95_terminal"]]
    st.markdown(
        '<div class="mini-grid">' + "".join(
            f'<div class="mini-stat"><span>{label}</span><strong>{money(value)}</strong></div>'
            for label, value in zip(["5th percentile", "Median", "Mean", "95th percentile"], percentile_values)
        ) + '</div>', unsafe_allow_html=True,
    )
    left, right = st.columns([1.5, .8])
    with left:
        components = result.fragility_components.sort_values("Risk Score")
        figure = px.bar(components, x="Risk Score", y="Component", orientation="h", color="Risk Score",
                        color_continuous_scale=[[0, COLORS["positive"]], [.55, COLORS["warning"]], [1, COLORS["negative"]]], range_color=[0, 100])
        style_figure(figure, "Fragility Score Components", 400)
        figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True, key="executive_fragility_components")
    with right:
        benchmark_frame = pd.DataFrame({"Series": ["Portfolio", config.benchmark], "Annual Return": [metrics["historical_return"], metrics["benchmark_return"]]})
        figure = px.bar(benchmark_frame, x="Series", y="Annual Return", color="Series", text="Annual Return",
                        color_discrete_map={"Portfolio": COLORS["accent"], config.benchmark: COLORS["info"]})
        style_figure(figure, "Historical Return Snapshot", 400)
        figure.update_yaxes(tickformat=".0%")
        figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, use_container_width=True, key="executive_benchmark_snapshot")


def render_portfolio_analytics(result, config) -> None:
    risk_tab, correlation_tab, frontier_tab, rolling_tab = st.tabs(["Risk Decomposition", "Correlation", "Efficient Frontier", "Rolling Analytics"])
    with risk_tab:
        table = risk_contribution_table(result)
        figure_data = table.melt(id_vars=["Ticker"], value_vars=["Weight", "Risk Contribution"], var_name="Measure", value_name="Share")
        figure = px.bar(figure_data, x="Ticker", y="Share", color="Measure", barmode="group",
                        color_discrete_map={"Weight": COLORS["info"], "Risk Contribution": COLORS["accent"]})
        style_figure(figure, "Portfolio Weight vs Volatility Contribution", 430)
        figure.update_yaxes(tickformat=".0%")
        figure.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>")
        st.plotly_chart(figure, use_container_width=True, key="analytics_risk_contribution")
        largest = table.loc[table["Risk Contribution"].idxmax()]
        st.markdown(
            f'<div class="callout"><b>{largest["Ticker"]}</b> represents {pct(largest["Weight"])} of weight but {pct(largest["Risk Contribution"])} of modeled volatility; '
            f'the risk concentration gap is {largest["Risk Concentration Gap"]:+.1%}. Contributions are normalized to approximately 100%.</div>',
            unsafe_allow_html=True,
        )
        display = table.copy()
        st.dataframe(display.style.format({"Weight": "{:.1%}", "Risk Contribution": "{:.1%}", "Risk Concentration Gap": "{:+.1%}"})
                     .map(lambda value: f"color:{COLORS['negative']};font-weight:600" if isinstance(value, (float, np.floating)) and value > .05 else "", subset=["Risk Concentration Gap"]),
                     hide_index=True, width="stretch")
        st.caption("Risk contribution is the component contribution to annualized portfolio volatility under the historical covariance estimate.")
    with correlation_tab:
        normal, stressed = correlation_matrices(result)
        mode = st.radio("Correlation view", ["Historical", "Crisis-stressed"], horizontal=True, key="correlation_view", help="The stressed view blends historical correlations 72% toward one, matching the crisis-regime assumption.")
        matrix = normal if mode == "Historical" else stressed
        figure = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, zmin=-1, zmax=1,
                                      colorscale=[[0, "#5B8DEF"], [.5, "#171B22"], [1, "#F05D64"]],
                                      text=np.round(matrix.values, 2), texttemplate="%{text:.2f}",
                                      hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>", colorbar={"title": "Corr."}))
        style_figure(figure, f"{mode} Asset Correlation", 470)
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(figure, use_container_width=True, key=f"correlation_heatmap_{mode}")
        interpretation = correlation_interpretation(normal)
        implication = "Diversification is limited by clustered co-movement." if interpretation["highest"] > .75 else "Historical pairwise dispersion provides some diversification, but may weaken in crisis regimes."
        st.markdown(f'<div class="callout">Highest pair: <b>{interpretation["highest_pair"]}</b> ({interpretation["highest"]:.2f}); lowest pair: <b>{interpretation["lowest_pair"]}</b> ({interpretation["lowest"]:.2f}). {implication}</div>', unsafe_allow_html=True)
    with frontier_tab:
        cloud, markers, weights = efficient_frontier(result, config)
        figure = px.scatter(cloud, x="Annual Volatility", y="Annual Return", color="Sharpe Ratio",
                            color_continuous_scale=[[0, COLORS["muted"]], [.5, COLORS["info"]], [1, COLORS["accent"]]],
                            opacity=.42, render_mode="webgl")
        marker_colors = {"Current": COLORS["text"], "Minimum Variance": COLORS["positive"], "Maximum Sharpe": COLORS["accent"], "Equal Weight": COLORS["info"], "Crisis-Resilient": COLORS["purple"]}
        for _, row in markers.iterrows():
            figure.add_trace(go.Scatter(x=[row["Annual Volatility"]], y=[row["Annual Return"]], mode="markers+text", name=row["Portfolio"], text=[row["Portfolio"]], textposition="top center",
                                        marker={"size": 12, "color": marker_colors[row["Portfolio"]], "line": {"color": COLORS["background"], "width": 1.5}},
                                        hovertemplate=f"{row['Portfolio']}<br>Return: %{{y:.1%}}<br>Volatility: %{{x:.1%}}<br>Sharpe: {row['Sharpe Ratio']:.2f}<extra></extra>"))
        style_figure(figure, "Long-Only Efficient Frontier", 500)
        figure.update_xaxes(tickformat=".0%")
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True, key="efficient_frontier")
        optimized = weights.pivot(index="Ticker", columns="Portfolio", values="Weight").reset_index()
        st.dataframe(optimized.style.format({column: "{:.1%}" for column in optimized.columns if column != "Ticker"}), hide_index=True, width="stretch")
        st.caption("The frontier uses historical annualized estimates, long-only weights, and a 45% position cap. It is sensitive to the selected sample and is not an allocation recommendation.")
    with rolling_tab:
        rolling = rolling_analytics(result, config.risk_free_rate)
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, subplot_titles=("63-Day Annualized Volatility", "126-Day Rolling Sharpe"))
        figure.add_trace(go.Scatter(x=rolling.index, y=rolling["Rolling 63D Volatility"], line={"color": COLORS["warning"], "width": 1.8}, name="Volatility"), row=1, col=1)
        figure.add_trace(go.Scatter(x=rolling.index, y=rolling["Rolling 126D Sharpe"], line={"color": COLORS["accent"], "width": 1.8}, name="Sharpe"), row=2, col=1)
        style_figure(figure, "Rolling Risk and Risk-Adjusted Return", 500, "x unified")
        figure.update_yaxes(tickformat=".0%", row=1, col=1)
        st.plotly_chart(figure, use_container_width=True, key="rolling_vol_sharpe")
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=rolling.index, y=rolling["Rolling 126D Beta"], name="Beta", line={"color": COLORS["info"], "width": 1.8}))
        figure.add_trace(go.Scatter(x=rolling.index, y=rolling["Rolling 126D Correlation"], name="Correlation", line={"color": COLORS["purple"], "width": 1.8}))
        style_figure(figure, f"Rolling Market Sensitivity vs {config.benchmark}", 410, "x unified")
        st.plotly_chart(figure, use_container_width=True, key="rolling_beta_correlation")
        latest = rolling.dropna().iloc[-1]
        st.markdown(f'<div class="callout">Latest rolling estimates: volatility <b>{pct(latest["Rolling 63D Volatility"])}</b>, Sharpe <b>{ratio(latest["Rolling 126D Sharpe"])}</b>, beta <b>{ratio(latest["Rolling 126D Beta"])}</b>, and benchmark correlation <b>{ratio(latest["Rolling 126D Correlation"])}</b>. These are historical, window-dependent estimates.</div>', unsafe_allow_html=True)


def render_performance(result, config) -> None:
    metrics = result.metrics
    kpi_grid([
        {"label": "Annualized Return", "value": pct(metrics["historical_return"]), "badge": "Historical", "tone": "neutral", "help": "Compounded annualized portfolio return over the selected history."},
        {"label": "Annualized Volatility", "value": pct(metrics["historical_volatility"]), "badge": "Historical", "tone": "warning", "help": "Annualized standard deviation of daily portfolio returns."},
        {"label": "Sharpe Ratio", "value": ratio(metrics["sharpe"]), "badge": "Risk-adjusted", "tone": "positive" if metrics["sharpe"] >= 1 else "neutral", "help": "Historical annualized excess return divided by volatility."},
        {"label": "Maximum Drawdown", "value": pct(metrics["historical_max_drawdown"]), "badge": "Historical", "tone": "negative", "help": "Largest historical peak-to-trough decline."},
        {"label": "Beta", "value": ratio(metrics["beta"]), "badge": config.benchmark, "tone": "neutral", "help": "Historical sensitivity to benchmark returns."},
        {"label": "Tracking Error", "value": pct(metrics["tracking_error"]), "badge": config.benchmark, "tone": "neutral", "help": "Annualized volatility of portfolio returns minus benchmark returns."},
        {"label": "Information Ratio", "value": ratio(metrics["information_ratio"]), "badge": config.benchmark, "tone": "positive" if metrics["information_ratio"] > 0 else "warning", "help": "Historical active return divided by tracking error."},
        {"label": "Benchmark Correlation", "value": ratio(metrics["benchmark_correlation"]), "badge": config.benchmark, "tone": "neutral", "help": "Historical return correlation between portfolio and benchmark."},
    ])
    growth_tab, assets_tab = st.tabs(["Portfolio vs Benchmark", "Asset-Level Performance"])
    with growth_tab:
        st.plotly_chart(benchmark_growth_chart(result, config.benchmark), use_container_width=True, key="performance_benchmark_growth")
    with assets_tab:
        st.plotly_chart(asset_growth_chart(result, config.benchmark), use_container_width=True, key="performance_asset_growth")
        annual_asset_returns = (1 + result.asset_returns).prod() ** (252 / len(result.asset_returns)) - 1
        annual_asset_volatility = result.asset_returns.std() * np.sqrt(252)
        asset_table = pd.DataFrame({"Ticker": result.asset_returns.columns, "Annual Return": annual_asset_returns.values, "Annual Volatility": annual_asset_volatility.values})
        st.dataframe(asset_table.style.format({"Annual Return": "{:.1%}", "Annual Volatility": "{:.1%}"}), hide_index=True, width="stretch")
    drawdowns, episode = historical_drawdowns(result)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns["Portfolio"], name="Portfolio", fill="tozeroy", fillcolor="rgba(240,93,100,.18)", line={"color": COLORS["negative"], "width": 1.8}))
    figure.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns["Benchmark"], name=config.benchmark, line={"color": COLORS["info"], "width": 1.5, "dash": "dot"}))
    figure.add_vrect(x0=episode["start"], x1=episode["recovery"] or drawdowns.index[-1], fillcolor="rgba(240,93,100,.08)", line_width=0)
    style_figure(figure, "Historical Drawdown", 440, "x unified")
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True, key="performance_drawdown")
    recovery_text = episode["recovery"].strftime("%Y-%m-%d") if episode["recovery"] is not None else "Not recovered"
    st.markdown(f'<div class="mini-grid"><div class="mini-stat"><span>Maximum drawdown</span><strong>{pct(episode["maximum_drawdown"])}</strong></div><div class="mini-stat"><span>Peak date</span><strong>{episode["start"].strftime("%Y-%m-%d")}</strong></div><div class="mini-stat"><span>Trough date</span><strong>{episode["trough"].strftime("%Y-%m-%d")}</strong></div><div class="mini-stat"><span>Recovery / duration</span><strong>{recovery_text} · {episode["duration_days"]} days</strong></div></div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown"],
        "Portfolio": [pct(metrics["historical_return"]), pct(metrics["historical_volatility"]), ratio(metrics["sharpe"]), pct(metrics["historical_max_drawdown"])],
        config.benchmark: [pct(metrics["benchmark_return"]), pct(metrics["benchmark_volatility"]), ratio(metrics["benchmark_sharpe"]), pct(metrics["benchmark_max_drawdown"])],
    })
    st.dataframe(comparison, hide_index=True, width="stretch")
    relative = metrics["historical_return"] - metrics["benchmark_return"]
    st.markdown(f'<div class="callout">The portfolio historically {"outperformed" if relative >= 0 else "underperformed"} {config.benchmark} by <b>{abs(relative):.1%}</b> annualized, with {"lower" if metrics["historical_volatility"] <= metrics["benchmark_volatility"] else "higher"} volatility. Results are sensitive to the selected period.</div>', unsafe_allow_html=True)


def render_stress_fragility(result, config) -> None:
    overview, regimes, liquidity, scenarios, resilient = st.tabs(["Fragility", "Market Regimes", "Liquidity Cascade", "Stress Scenarios", "Resilient Allocation"])
    with overview:
        left, right = st.columns([.9, 1.35])
        with left:
            st.plotly_chart(fragility_gauge(result), use_container_width=True, key="stress_fragility_gauge")
        with right:
            components = result.fragility_components.sort_values("Risk Score")
            figure = px.bar(components, x="Risk Score", y="Component", orientation="h", color="Risk Score", color_continuous_scale=[[0, COLORS["positive"]], [.55, COLORS["warning"]], [1, COLORS["negative"]]], range_color=[0, 100])
            style_figure(figure, "Weighted Fragility Drivers", 360)
            figure.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure, use_container_width=True, key="stress_fragility_components")
        current = result.allocation_comparison.iloc[0]
        defensive = result.allocation_comparison.iloc[1]
        st.markdown(f'<div class="callout">The crisis-resilient reference allocation changes the worst modeled stress loss from <b>{pct(current["Worst Stress Loss"])}</b> to <b>{pct(defensive["Worst Stress Loss"])}</b>. This is a model-based reference, not investment advice.</div>', unsafe_allow_html=True)
    with regimes:
        summary = regime_assumptions(result)
        left, right = st.columns([.85, 1.15])
        with left:
            figure = px.pie(summary, names="Regime", values="Share of Simulated Days", hole=.66,
                            color="Regime", color_discrete_map={"Bull": COLORS["positive"], "Normal": COLORS["info"], "Crisis": COLORS["negative"], "Recovery": COLORS["warning"]})
            style_figure(figure, "Simulated Share of Trading Days", 410)
            st.plotly_chart(figure, use_container_width=True, key="regime_mix")
        with right:
            figure = go.Figure(go.Heatmap(z=REGIME_TRANSITION_MATRIX.values, x=REGIME_TRANSITION_MATRIX.columns, y=REGIME_TRANSITION_MATRIX.index,
                                          zmin=0, zmax=1, colorscale=[[0, COLORS["secondary"]], [1, COLORS["accent"]]], text=REGIME_TRANSITION_MATRIX.values, texttemplate="%{text:.1%}", colorbar={"title": "Transition"}))
            style_figure(figure, "Markov Transition Matrix", 410)
            figure.update_yaxes(autorange="reversed")
            st.plotly_chart(figure, use_container_width=True, key="regime_transition")
        st.dataframe(summary.style.format({"Share of Simulated Days": "{:.1%}", "Volatility Multiplier": "{:.2f}×", "Correlation Blend": "{:.0%}", "Return Adjustment": "{:+.3%}", "Average Duration (days)": "{:.1f}"}), hide_index=True, width="stretch")
        st.markdown('<div class="method-note">Regime mix is the simulated share of trading days—not the probability of mutually exclusive terminal outcomes. The Markov process is an assumption calibrated for scenario exploration; crisis frequency is not a forecast certainty.</div>', unsafe_allow_html=True)
    with liquidity:
        m = result.metrics
        shortfall_value = "No shortfall in modeled scenarios" if m["liquidity_shortfall_probability"] <= 0 else pct(m["liquidity_shortfall_probability"])
        kpi_grid([
            {"label": "Liquidity Score", "value": f"{m['weighted_liquidity_score']:.0f}/100", "badge": "Proxy", "tone": "neutral", "help": "Weighted tradability proxy based on median dollar volume and volatility."},
            {"label": "Redemption Event", "value": pct(m["redemption_probability_realized"]), "badge": "Conditional", "tone": "warning", "help": "Share of paths with a drawdown breach followed by the modeled investor redemption event."},
            {"label": "Synthetic Collateral Call", "value": pct(m["margin_call_probability"]), "badge": "Assumption", "tone": "warning", "help": "Share of breached paths receiving the stated margin-call percentage. This is a synthetic collateral assumption, not observed leverage."},
            {"label": "Forced Sale", "value": pct(m["forced_sale_probability"]), "badge": "Liquidity", "tone": "negative" if m["forced_sale_probability"] > .25 else "warning", "help": "Share of paths requiring asset sales after redemption/collateral needs exceed the cash buffer; it is not loss probability."},
            {"label": "Liquidity Shortfall", "value": shortfall_value, "badge": "Capacity", "tone": "positive" if m["liquidity_shortfall_probability"] <= 0 else "negative", "help": "Share of modeled forced sales exceeding estimated participation capacity."},
            {"label": "Conditional Sale Cost", "value": money(m["conditional_forced_sale_cost"]), "badge": "Estimated", "tone": "warning", "help": "Average spread, slippage, and market-impact cost among paths with forced sales."},
        ])
        least = result.liquidity.loc[result.liquidity["Liquidity Score"].idxmin()]
        left, right = st.columns([1.35, 1])
        with left:
            figure = px.scatter(result.liquidity, x="Median Dollar Volume", y="Liquidity Score", size="5% ADV Capacity", color="Estimated Spread (bps)", text="Ticker", log_x=True,
                                color_continuous_scale=[[0, COLORS["accent"]], [.6, COLORS["warning"]], [1, COLORS["negative"]]],
                                hover_data={"Median Dollar Volume": ":$,.0f", "Estimated Spread (bps)": ":.1f", "5% ADV Capacity": ":$,.0f"})
            style_figure(figure, "Asset Liquidity Map", 445)
            figure.update_traces(textposition="top center")
            st.plotly_chart(figure, use_container_width=True, key="liquidity_map")
        with right:
            frame = pd.DataFrame({"Event": ["Redemption", "Collateral call", "Forced sale", "Shortfall"], "Probability": [m["redemption_probability_realized"], m["margin_call_probability"], m["forced_sale_probability"], m["liquidity_shortfall_probability"]]})
            figure = px.bar(frame, x="Event", y="Probability", text="Probability", color="Event", color_discrete_sequence=[COLORS["warning"], COLORS["info"], COLORS["negative"], "#A9444A"])
            style_figure(figure, "Cascade Event Incidence", 445)
            figure.update_yaxes(tickformat=".0%")
            figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            figure.update_layout(showlegend=False)
            st.plotly_chart(figure, use_container_width=True, key="liquidity_cascade")
        st.markdown(f'<div class="callout"><b>{least["Ticker"]}</b> is the least liquid holding under the proxy: score {least["Liquidity Score"]:.1f}/100, median dollar volume {compact_money(least["Median Dollar Volume"])}, and estimated spread {least["Estimated Spread (bps)"]:.1f} bps. Bubble size represents estimated 5% ADV sale capacity.</div>', unsafe_allow_html=True)
        st.dataframe(result.liquidity.style.format({"Liquidity Score": "{:.1f}", "Median Dollar Volume": "${:,.0f}", "Estimated Spread (bps)": "{:.1f}", "5% ADV Capacity": "${:,.0f}"}), hide_index=True, width="stretch")
        with st.expander("Modeled leverage, redemption, and execution assumptions"):
            st.write(f"Drawdown trigger: {config.margin_trigger:.1%}; conditional redemption probability: {config.redemption_probability:.0%}; redemption size: {config.redemption_pct:.0%}; synthetic collateral call: {config.margin_call_pct:.0%}; cash buffer: {config.cash_buffer_pct:.0%}.")
            st.caption("The collateral call is a scenario proxy. If the portfolio has no leverage, set Margin call (%) to 0 in Liquidity cascade assumptions.")
    with scenarios:
        worst = result.stress_tests.loc[result.stress_tests["Net Portfolio Shock"].idxmin()]
        kpi_grid([
            {"label": "Worst Scenario", "value": str(worst["Scenario"]), "badge": "Stylized", "tone": "negative", "help": "Scenario producing the lowest modeled net portfolio shock."},
            {"label": "Net Portfolio Shock", "value": pct(worst["Net Portfolio Shock"]), "badge": "Worst", "tone": "negative", "help": "Gross asset shock less estimated liquidation cost."},
            {"label": "Liquidation Cost", "value": pct(worst["Liquidation Cost"]), "badge": "Estimated", "tone": "warning", "help": "Modeled spread, slippage, and market-impact cost under stressed capacity."},
            {"label": "Stressed Value", "value": money(worst["Stressed Value"]), "badge": "Modeled", "tone": "negative", "help": "Initial portfolio value after the stylized net shock."},
        ])
        figure = px.bar(result.stress_tests, x="Scenario", y="Net Portfolio Shock", color="Net Portfolio Shock", text="Net Portfolio Shock", color_continuous_scale=[[0, COLORS["negative"]], [1, COLORS["warning"]]])
        style_figure(figure, "Historical-Style Stress Scenario Comparison", 440)
        figure.update_yaxes(tickformat=".0%")
        figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True, key="stress_scenarios")
        formatted = result.stress_tests.style.format({"Gross Portfolio Shock": "{:.1%}", "Liquidation Cost": "{:.2%}", "Net Portfolio Shock": "{:.1%}", "Stressed Value": "${:,.0f}", "Correlation Assumption": "{:.2f}", "Liquidity Capacity Multiplier": "{:.0%}"})
        st.dataframe(formatted, hide_index=True, width="stretch")
        driver_text = "; ".join(f"{row['Scenario']}: {row['Largest Loss Driver']}" for _, row in result.stress_tests.iterrows())
        st.markdown(f'<div class="callout">Largest modeled loss driver by scenario — {driver_text}. Contributions reflect portfolio weights, historical benchmark sensitivity, and transparent asset-class shock adjustments.</div>', unsafe_allow_html=True)
        st.caption("These are transparent stylized approximations—not official regulatory stress tests or forecasts.")
    with resilient:
        allocation = pd.DataFrame({"Ticker": config.tickers, "Current": config.weights, "Crisis-Resilient": result.resilient_weights})
        allocation["Change"] = allocation["Crisis-Resilient"] - allocation["Current"]
        melted = allocation.melt(id_vars="Ticker", value_vars=["Current", "Crisis-Resilient"], var_name="Portfolio", value_name="Weight")
        figure = px.bar(melted, x="Ticker", y="Weight", color="Portfolio", barmode="group", color_discrete_map={"Current": COLORS["info"], "Crisis-Resilient": COLORS["accent"]})
        style_figure(figure, "Current vs Crisis-Resilient Weights", 430)
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True, key="resilient_weights")
        st.dataframe(allocation.style.format({"Current": "{:.1%}", "Crisis-Resilient": "{:.1%}", "Change": "{:+.1%}"}).map(lambda value: f"color:{COLORS['positive']}" if isinstance(value, (float, np.floating)) and value > 0 else f"color:{COLORS['negative']}" if isinstance(value, (float, np.floating)) and value < 0 else "", subset=["Change"]), hide_index=True, width="stretch")
        comparison = result.allocation_comparison
        st.dataframe(comparison.style.format({"Historical Return": "{:.1%}", "Historical Volatility": "{:.1%}", "Sharpe Ratio": "{:.2f}", "Average Stress Loss": "{:.1%}", "Worst Stress Loss": "{:.1%}", "Weighted Liquidity Score": "{:.1f}", "Target Probability": "{:.1%}"}), hide_index=True, width="stretch")
        current, defensive = comparison.iloc[0], comparison.iloc[1]
        return_change = defensive["Historical Return"] - current["Historical Return"]
        vol_change = defensive["Historical Volatility"] - current["Historical Volatility"]
        tail_change = defensive["Worst Stress Loss"] - current["Worst Stress Loss"]
        increased = ", ".join(allocation.loc[allocation["Change"] > .01, "Ticker"]) or "none"
        reduced = ", ".join(allocation.loc[allocation["Change"] < -.01, "Ticker"]) or "none"
        st.markdown(f'<div class="callout"><b>Trade-off:</b> historical return changes by {return_change:+.1%}, volatility by {vol_change:+.1%}, and worst stress loss improves by {tail_change:+.1%}. Increased: {increased}; reduced: {reduced}. The rules favor liquidity, lower volatility, and lower positive beta, subject to a 45% cap.</div>', unsafe_allow_html=True)
        st.caption("The crisis-resilient allocation is an analytical reference and is not investment advice.")


def render_simulation(result, config) -> None:
    method_alias = {"Geometric Brownian Motion": "GBM", "Historical Bootstrap": "Bootstrap", "Student-t": "Student-t", "Regime Switching": "Regime Switching"}.get(config.method, config.method)
    st.markdown(f'<div class="callout"><b>Model:</b> {escape(method_alias)} · <b>Paths:</b> {config.simulations:,} · <b>Horizon:</b> {config.simulation_days} trading days · <b>Initial:</b> {money(config.initial_investment)} · <b>Target:</b> {money(config.target_value)} · <b>Confidence:</b> {config.confidence:.0%}</div>', unsafe_allow_html=True)
    distribution_tab, paths_tab, drawdown_tab = st.tabs(["Distribution", "Paths & Percentiles", "Drawdown Risk"])
    with distribution_tab:
        left, right = st.columns([1.35, .85])
        with left:
            st.plotly_chart(terminal_distribution(result, config), use_container_width=True, key="simulation_terminal_distribution")
        with right:
            st.plotly_chart(outcome_probability_chart(result, config), use_container_width=True, key="simulation_outcomes")
            st.caption("Every simulated path appears in exactly one category: loss, no loss but below target, or target reached. Displayed percentages are rounded to total exactly 100.0%.")
        stats = simulation_statistics(result, config)
        display_values = []
        for _, row in stats.iterrows():
            if row["Format"] == "currency":
                display_values.append(money(row["Value"]))
            elif row["Format"] == "probability":
                display_values.append(probability(row["Value"]))
            elif row["Format"] == "percent":
                display_values.append(pct(row["Value"]))
            else:
                display_values.append(ratio(row["Value"]))
        stats_display = pd.DataFrame({"Statistic": stats["Statistic"], "Value": display_values})
        st.dataframe(stats_display, hide_index=True, width="stretch")
    with paths_tab:
        sample_count = min(150, result.paths.shape[1])
        path_colors = [
            "rgba(34,199,184,.13)", "rgba(91,141,239,.12)", "rgba(139,124,246,.12)",
            "rgba(46,204,143,.11)", "rgba(231,169,75,.10)", "rgba(168,176,188,.09)",
        ]
        figure = go.Figure()
        for index in range(sample_count):
            figure.add_trace(go.Scatter(
                y=result.paths[:, index], mode="lines",
                line={"width": .65, "color": path_colors[index % len(path_colors)]},
                showlegend=False, hoverinfo="skip",
            ))
        figure.add_trace(go.Scatter(y=result.path_percentiles["Median"], mode="lines", line={"width": 2.5, "color": COLORS["accent"]}, name="Median"))
        figure.add_hline(y=config.initial_investment, line_dash="dot", line_color=COLORS["muted"], annotation_text="Initial")
        figure.add_hline(y=config.target_value, line_dash="dash", line_color=COLORS["purple"], annotation_text="Target")
        style_figure(figure, f"Monte Carlo Path Cloud ({sample_count} of {config.simulations:,} Paths)", 500)
        figure.update_xaxes(title="Trading day")
        figure.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(figure, use_container_width=True, key="simulation_sample_paths")
        st.caption("A representative subset of paths is shown to preserve readability and browser performance; the statistics use every simulated path.")
        st.plotly_chart(fan_chart(result, "Simulation Percentile Fan Chart", 450), use_container_width=True, key="simulation_fan")
    with drawdown_tab:
        drawdowns = result.paths / np.maximum.accumulate(result.paths, axis=0) - 1
        maximum = drawdowns.min(axis=0)
        figure = go.Figure(go.Histogram(x=maximum, nbinsx=55, marker_color=COLORS["negative"], opacity=.72))
        figure.add_vline(x=float(np.median(maximum)), line_dash="dash", line_color=COLORS["text"], annotation_text="Median")
        style_figure(figure, "Maximum Drawdown Distribution", 430)
        figure.update_xaxes(tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True, key="simulation_drawdown_distribution")
        st.markdown(f'<div class="callout">Median simulated maximum drawdown is <b>{pct(result.metrics["median_simulated_drawdown"])}</b>. Median modeled recovery time is <b>{days(result.metrics["median_recovery_days"])}</b>, with a recovery incidence of <b>{pct(result.metrics["recovery_probability"])}</b> within the selected horizon.</div>', unsafe_allow_html=True)
    with st.expander("Simulation assumptions and methodology"):
        st.write(f"Model: {config.method}; random seed: {config.seed}; Student-t degrees of freedom: {config.student_df}; annual risk-free rate: {config.risk_free_rate:.2%}.")
        st.write(f"Liquidity overlay: redemption probability {config.redemption_probability:.0%}, redemption size {config.redemption_pct:.0%}, synthetic collateral call {config.margin_call_pct:.0%}, drawdown trigger {config.margin_trigger:.1%}, cash buffer {config.cash_buffer_pct:.0%}.")
        st.caption("Outputs are simulated estimates conditional on historical inputs and stated assumptions.")


def render_report(result, config) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    section_header("Export Center", "Analysis packages for review, presentation, and audit")
    metadata = pd.DataFrame({
        "Field": ["Portfolio", "Generated", "Tickers", "Benchmark", "Method", "Paths", "Horizon", "Confidence"],
        "Value": ["Portfolio Fragility Lab", generated, ", ".join(config.tickers), config.benchmark, config.method, f"{config.simulations:,}", f"{config.simulation_days} trading days", f"{config.confidence:.0%}"],
    })
    st.dataframe(metadata, hide_index=True, width="stretch")
    terminal_csv = pd.DataFrame({"terminal_value": result.terminal_values}).to_csv(index=False).encode("utf-8")
    report_key = (
        config.tickers, config.weights, config.benchmark, config.method, config.seed,
        config.simulation_days, config.simulations, round(result.metrics["expected_terminal"], 2),
    )
    if st.session_state.get("report_bundle_key") != report_key:
        report_bundle: dict[str, bytes | None] = {}
        report_errors: dict[str, str] = {}
        for name, builder in {
            "excel": build_excel_report,
            "pdf": build_pdf_report,
            "powerpoint": build_powerpoint_report,
        }.items():
            try:
                report_bundle[name] = builder(result, config)
            except Exception as error:
                report_bundle[name] = None
                report_errors[name] = str(error)
        st.session_state.report_bundle_key = report_key
        st.session_state.report_bundle = report_bundle
        st.session_state.report_errors = report_errors
    report_bundle = st.session_state.report_bundle
    report_errors = st.session_state.report_errors
    excel = report_bundle["excel"]
    pdf = report_bundle["pdf"]
    powerpoint = report_bundle["powerpoint"]
    columns = st.columns(3)
    with columns[0]:
        if excel is not None:
            st.download_button("Excel workbook", excel, "portfolio_fragility_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", width="stretch")
    with columns[1]:
        st.download_button("Terminal scenarios CSV", terminal_csv, "terminal_scenarios.csv", "text/csv", width="stretch")
    with columns[2]:
        st.download_button("Risk metrics CSV", risk_metrics_csv(result), "risk_metrics.csv", "text/csv", width="stretch")
    presentation_columns = st.columns(2)
    with presentation_columns[0]:
        if pdf is not None:
            st.download_button("PDF executive report", pdf, "portfolio_fragility_executive.pdf", "application/pdf", width="stretch")
    with presentation_columns[1]:
        if powerpoint is not None:
            st.download_button("PowerPoint summary", powerpoint, "portfolio_fragility_summary.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", width="stretch")
    if report_errors:
        failed = ", ".join(name.upper() for name in report_errors)
        st.warning(f"The following export formats could not be generated: {failed}. Other downloads remain available.")
        with st.expander("Technical report-generation detail"):
            for name, detail in report_errors.items():
                st.code(f"{name.upper()}: {detail}")
    with st.expander("Methodology and assumptions", expanded=True):
        st.write(f"Historical data begins {config.start_date}; benchmark {config.benchmark}; simulation model {config.method}; {config.simulations:,} paths over {config.simulation_days} trading days; confidence {config.confidence:.0%}.")
        st.write("Fragility combines drawdown severity, liquidity shortfall, forced-sale exposure, crisis-correlation instability, and concentration. Stress scenarios are stylized historical approximations. Liquidity scores and execution costs are transparent proxies based on dollar volume, volatility, spread, slippage, and market impact.")
    with st.expander("Limitations and disclaimer", expanded=True):
        st.write(DISCLAIMER)
        st.write("Historical estimates may not represent future conditions. Liquidity proxies do not replace security-level market-depth analysis. Taxes, options, security-specific fundamentals, transaction timing, and regulatory capital rules are outside the model.")


def render_dashboard(result, config) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Institutional portfolio risk intelligence</div>'
        '<h1>Portfolio Fragility Lab</h1><p>Regime-aware simulation, historical risk decomposition, liquidity cascades, and transparent stress testing in one decision-oriented workspace.</p></div>',
        unsafe_allow_html=True,
    )
    executive, analytics, performance, stress, simulation, report = st.tabs([
        "Executive", "Portfolio Analytics", "Performance", "Stress & Fragility", "Simulation", "Report"
    ])
    with executive:
        render_executive(result, config)
    with analytics:
        render_portfolio_analytics(result, config)
    with performance:
        render_performance(result, config)
    with stress:
        render_stress_fragility(result, config)
    with simulation:
        render_simulation(result, config)
    with report:
        render_report(result, config)
    st.markdown(f'<div class="footer-note">{escape(DISCLAIMER)}</div>', unsafe_allow_html=True)
