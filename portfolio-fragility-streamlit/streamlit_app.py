"""Results-first Streamlit interface for Portfolio Fragility Lab."""

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
    download_prices,
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
    :root { --ink:#e9eef6; --muted:#91a0b6; --cyan:#43d5c7; --navy:#07111f; }
    .stApp { background:linear-gradient(135deg,#07111f 0%,#0b1727 55%,#101c2c 100%); }
    [data-testid="stSidebar"] { background:#081321; border-right:1px solid #1e3046; }
    .block-container { padding-top:2.2rem; padding-bottom:4rem; max-width:1500px; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-0.025em; }
    p,.stCaption { color:var(--muted); }
    [data-testid="stMetric"] { background:rgba(15,31,50,.86); border:1px solid #20354d;
      border-radius:14px; padding:18px 20px; box-shadow:0 12px 30px rgba(0,0,0,.18); }
    [data-testid="stMetricLabel"] { color:#91a0b6; }
    [data-testid="stMetricValue"] { color:#f5f8fc; }
    .hero { padding:18px 0 24px; }
    .eyebrow { color:#43d5c7; font-size:.78rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2.2rem,4vw,4rem); line-height:1; margin:.5rem 0 1rem; }
    .hero p { max-width:760px; font-size:1.05rem; }
    .notice { padding:14px 16px; border:1px solid #24435a; border-radius:12px; background:#0c1c2c; color:#a9b8cb; }
    .stButton>button { border-radius:10px; font-weight:700; min-height:46px; }
    [data-testid="stDataFrame"] { border:1px solid #20354d; border-radius:12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers: tuple[str, ...], benchmark: str, start: date):
    return download_prices(tickers, benchmark, start)


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
        margin={"l": 30, "r": 20, "t": 58, "b": 35},
        hovermode="x unified",
    )
    return figure


def fan_chart(percentiles: pd.DataFrame) -> go.Figure:
    x = np.arange(len(percentiles))
    figure = base_figure("Portfolio Value Range Over Time")
    figure.add_trace(go.Scatter(x=x, y=percentiles["P95"], line={"width": 0}, showlegend=False))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P5"], fill="tonexty", fillcolor="rgba(67,213,199,.12)", line={"width": 0}, name="5th–95th percentile"))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P75"], line={"width": 0}, showlegend=False))
    figure.add_trace(go.Scatter(x=x, y=percentiles["P25"], fill="tonexty", fillcolor="rgba(67,213,199,.24)", line={"width": 0}, name="25th–75th percentile"))
    figure.add_trace(go.Scatter(x=x, y=percentiles["Median"], line={"color": "#43d5c7", "width": 2.4}, name="Median"))
    figure.update_yaxes(tickprefix="$", tickformat=",")
    figure.update_xaxes(title="Trading day")
    return figure


def terminal_chart(result, config) -> go.Figure:
    figure = px.histogram(
        x=result.terminal_values,
        nbins=60,
        labels={"x": "Terminal portfolio value", "y": "Paths"},
        color_discrete_sequence=["#43d5c7"],
        opacity=0.78,
    )
    figure.update_layout(
        title="Terminal Wealth Distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,.55)",
        margin={"l": 30, "r": 20, "t": 58, "b": 35},
        showlegend=False,
    )
    for value, label, color in [
        (config.initial_investment, "Initial", "#f5a65b"),
        (result.metrics["median_terminal"], "Median", "#ffffff"),
        (config.target_value, "Target", "#ad7cff"),
    ]:
        figure.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=label)
    figure.update_xaxes(tickprefix="$", tickformat=",")
    return figure


def normalized_chart(result, benchmark: str) -> go.Figure:
    asset_normalized = result.prices / result.prices.iloc[0] * 100
    benchmark_series = (1 + result.benchmark_returns).cumprod() * 100
    figure = base_figure("Historical Growth of 100")
    for column in asset_normalized:
        figure.add_trace(go.Scatter(x=asset_normalized.index, y=asset_normalized[column], name=column, line={"width": 1.5}))
    figure.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, name=benchmark, line={"color": "#ffffff", "width": 2.5, "dash": "dot"}))
    return figure


def download_workbook(result, config) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([config.__dict__]).to_excel(writer, "Configuration", index=False)
        result.prices.to_excel(writer, "Prices")
        result.asset_returns.to_excel(writer, "Returns")
        pd.Series(result.metrics, name="Value").to_excel(writer, "Risk Metrics")
        result.risk_contributions.to_excel(writer, "Risk Contributions", index=False)
        pd.Series(result.terminal_values, name="Terminal Value").to_excel(writer, "Terminal Values", index=False)
    return output.getvalue()


with st.sidebar:
    st.markdown("### Portfolio setup")
    st.caption("Satır ekleyebilir, silebilir ve ağırlıkları değiştirebilirsin.")
    default_portfolio = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GLD", "TLT"],
            "Weight (%)": [30.0, 30.0, 20.0, 20.0],
        }
    )
    portfolio_table = st.data_editor(
        default_portfolio,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance sembolü"),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
    )
    total_weight = pd.to_numeric(portfolio_table.get("Weight (%)"), errors="coerce").sum()
    st.caption(f"Toplam ağırlık: %{total_weight:.1f}")
    st.divider()
    initial_investment = st.number_input("Başlangıç yatırımı ($)", min_value=1_000, value=100_000, step=5_000)
    target_value = st.number_input("Hedef portföy değeri ($)", min_value=1_000, value=115_000, step=5_000)
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    start_date = st.date_input("Geçmiş veri başlangıcı", value=date(2018, 1, 1), max_value=date.today())
    horizon_label = st.selectbox("Simülasyon süresi", ["3 ay", "6 ay", "1 yıl", "2 yıl", "3 yıl"], index=2)
    horizon_map = {"3 ay": 63, "6 ay": 126, "1 yıl": 252, "2 yıl": 504, "3 yıl": 756}
    simulations = st.select_slider("Simülasyon sayısı", [1_000, 5_000, 10_000, 25_000], value=10_000)
    method = st.selectbox("Simülasyon modeli", ["Geometric Brownian Motion", "Historical Bootstrap", "Student-t"])
    with st.expander("Gelişmiş ayarlar"):
        confidence = st.slider("Güven seviyesi", 0.90, 0.99, 0.95, 0.01)
        risk_free_rate = st.number_input("Yıllık risksiz faiz (%)", 0.0, 30.0, 4.0, 0.25) / 100
        seed = st.number_input("Random seed", 0, 1_000_000, 42)
        student_df = st.slider("Student-t serbestlik derecesi", 3, 20, 5)
    run_button = st.button("Analizi Başlat", type="primary", use_container_width=True)
    st.caption("Eğitim ve araştırma amaçlıdır; yatırım tavsiyesi değildir.")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Institutional portfolio analytics</div>
      <h1>Portfolio Fragility Lab</h1>
      <p>Kod görmeden portföyünü test et. Binlerce olası geleceği simüle et, aşağı yönlü riski ölç ve portföyünün hedefe ulaşma olasılığını incele.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run_button and "analysis" not in st.session_state:
    st.markdown('<div class="notice">Sol panelden portföyünü oluştur ve <b>Analizi Başlat</b> butonuna bas.</div>', unsafe_allow_html=True)
    st.stop()

if run_button:
    try:
        tickers, weights = validate_portfolio(portfolio_table)
        if benchmark in tickers:
            st.info("Benchmark portföyde de bulunuyor; karşılaştırma yine hesaplanacaktır.")
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
        )
        with st.spinner("Piyasa verileri indiriliyor ve senaryolar hesaplanıyor..."):
            prices, benchmark_prices = cached_prices(tickers, benchmark, start_date)
            result = run_analysis(config, prices, benchmark_prices)
        st.session_state.analysis = result
        st.session_state.analysis_config = config
    except PortfolioError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error("Analiz tamamlanamadı. Ticker'ları ve internet bağlantısını kontrol et.")
        with st.expander("Teknik ayrıntı"):
            st.code(str(error))
        st.stop()

result = st.session_state.analysis
config = st.session_state.analysis_config
metrics = result.metrics

st.success(f"{', '.join(config.tickers)} portföyü için {config.simulations:,} senaryo tamamlandı.")

metric_columns = st.columns(5)
metric_columns[0].metric("Beklenen değer", money(metrics["expected_terminal"]), money(metrics["expected_terminal"] - config.initial_investment))
metric_columns[1].metric("Medyan değer", money(metrics["median_terminal"]))
metric_columns[2].metric("Zarar olasılığı", pct(metrics["probability_loss"]), delta_color="inverse")
metric_columns[3].metric("Hedefe ulaşma", pct(metrics["probability_target"]))
metric_columns[4].metric(f"{config.confidence:.0%} VaR", money(metrics["var_currency"]), delta_color="inverse")

overview_tab, simulation_tab, risk_tab, history_tab, report_tab = st.tabs(
    ["Genel Bakış", "Simülasyon", "Risk Analizi", "Geçmiş Performans", "Rapor"]
)

with overview_tab:
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(fan_chart(result.path_percentiles), use_container_width=True)
    with right:
        st.subheader("Sonuçların yorumu")
        st.write(
            f"{config.simulation_days} işlem günü sonunda medyan portföy değeri "
            f"**{money(metrics['median_terminal'])}**. Portföyün başlangıç değerinin altında "
            f"kalma olasılığı **{pct(metrics['probability_loss'])}**, "
            f"{money(config.target_value)} hedefine ulaşma olasılığı ise "
            f"**{pct(metrics['probability_target'])}** olarak hesaplandı."
        )
        st.write(
            f"{config.confidence:.0%} güven seviyesinde tahmini VaR "
            f"**{money(metrics['var_currency'])}**, daha kötü kuyruk senaryolarındaki "
            f"ortalama kayıp olan Expected Shortfall ise **{money(metrics['es_currency'])}**."
        )
        st.caption("Bunlar olasılıksal senaryolardır; kesin fiyat tahmini değildir.")

with simulation_tab:
    st.plotly_chart(terminal_chart(result, config), use_container_width=True)
    sample_count = min(100, result.paths.shape[1])
    path_figure = base_figure("Sample Simulation Paths")
    for index in range(sample_count):
        path_figure.add_trace(go.Scatter(y=result.paths[:, index], mode="lines", line={"width": 0.7, "color": "rgba(67,213,199,.12)"}, showlegend=False, hoverinfo="skip"))
    path_figure.update_yaxes(tickprefix="$", tickformat=",")
    path_figure.update_xaxes(title="Trading day")
    st.plotly_chart(path_figure, use_container_width=True)

with risk_tab:
    risk_columns = st.columns(4)
    risk_columns[0].metric("Expected Shortfall", money(metrics["es_currency"]))
    risk_columns[1].metric("Medyan max drawdown", pct(metrics["median_simulated_drawdown"]))
    risk_columns[2].metric("Historical Sharpe", f"{metrics['sharpe']:.2f}")
    risk_columns[3].metric("Benchmark beta", f"{metrics['beta']:.2f}")
    contribution_figure = px.bar(
        result.risk_contributions,
        x="Ticker",
        y="Risk Contribution",
        color="Risk Contribution",
        color_continuous_scale=["#183047", "#43d5c7"],
        title="Contribution to Portfolio Volatility",
    )
    contribution_figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,17,31,.55)", coloraxis_showscale=False)
    contribution_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(contribution_figure, use_container_width=True)
    st.dataframe(
        result.risk_contributions.style.format({"Weight": "{:.1%}", "Risk Contribution": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

with history_tab:
    st.plotly_chart(normalized_chart(result, config.benchmark), use_container_width=True)
    historical_columns = st.columns(4)
    historical_columns[0].metric("Yıllık getiri", pct(metrics["historical_return"]))
    historical_columns[1].metric("Yıllık volatilite", pct(metrics["historical_volatility"]))
    historical_columns[2].metric("Maksimum drawdown", pct(metrics["historical_max_drawdown"]))
    historical_columns[3].metric("Beta", f"{metrics['beta']:.2f}")

with report_tab:
    st.subheader("Analiz dosyalarını indir")
    st.write("Portföy ayarları, fiyatlar, getiriler, risk metrikleri ve terminal senaryoları ayrı Excel sayfalarında sunulur.")
    st.download_button(
        "Excel raporunu indir",
        data=download_workbook(result, config),
        file_name="portfolio_fragility_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    st.download_button(
        "Terminal senaryolarını CSV indir",
        data=pd.DataFrame({"terminal_value": result.terminal_values}).to_csv(index=False),
        file_name="terminal_scenarios.csv",
        mime="text/csv",
    )
    st.markdown("---")
    st.caption("Model sınırlamaları: tarihsel parametreler geleceği garanti etmez; korelasyonlar ve volatilite krizlerde değişebilir; işlem maliyetleri ve likidite etkileri bu sürümde yer almaz.")
