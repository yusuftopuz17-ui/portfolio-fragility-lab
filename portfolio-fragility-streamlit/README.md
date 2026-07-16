# Portfolio Fragility Lab

A results-first institutional portfolio stress-testing application. The platform goes beyond ordinary terminal-value Monte Carlo analysis by modeling market regimes, crisis correlation, investor redemptions, margin calls, forced liquidation, trading frictions, liquidity shortfalls, and recovery behavior.

## Distinctive features

- Bull, normal, crisis, and recovery regimes driven by a Markov transition process
- Crisis volatility multipliers and correlations that converge during stress
- Asset liquidity scores derived from median dollar volume and volatility
- Investor redemption and collateral-assumption margin-call triggers
- Forced-sale logic with cash buffers and daily liquidation capacity
- Bid–ask spread, slippage, and square-root market-impact proxies
- Stylized 2008, March 2020, and 2022 stress scenarios
- Liquidity-shortfall and forced-sale probabilities
- Median time-to-recovery and recovery probability
- Transparent 0–100 Portfolio Fragility Score
- Current versus crisis-resilient allocation comparison
- GBM, historical bootstrap, Student-t, and regime-switching simulations
- Global Yahoo Finance instrument search by company or asset name, covering supported equities, funds, indices, commodities, currencies, and digital assets
- USD-normalized security pricing for mixed-currency portfolios, with quoted FX-pair return exposures preserved
- Localized portfolio weights accepting both decimal commas and decimal points
- Absolute-value or percentage-growth targets and custom hour/day/week/month/year horizons on a 252-step annual basis, including fractional-day scaling for short horizons
- Six-section black-and-charcoal institutional dashboard with responsive KPI cards
- Historical and stressed correlation heatmaps with dynamic diversification commentary
- Long-only efficient frontier, rolling volatility, rolling Sharpe, beta, and benchmark correlation
- Historical portfolio/benchmark and separate asset-level drawdowns with peak, trough, recovery, and duration diagnostics
- Interactive Plotly dashboards and Excel, CSV, PDF, and PowerPoint exports

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Upload the complete folder to GitHub.
2. Sign in at `https://share.streamlit.io`.
3. Select the repository and `main` branch.
4. If this folder is nested in the repository, use:

   ```text
   portfolio-fragility-streamlit/streamlit_app.py
   ```

5. Choose **Deploy**. Later GitHub commits are redeployed automatically.

## Model transparency

The stress scenarios are educational approximations based on broad historical market behavior. Tradable security prices are normalized to USD before cross-market return calculations; Yahoo FX pairs retain their quoted return exposure. Liquidity scores, spreads, liquidation capacity, and market impact are transparent proxies rather than security-level dealer quotes, particularly for indices or instruments without reported volume. The collateral-assumption leverage input affects only synthetic margin-call and liquidity calculations, not historical or simulated asset returns. Forced-sale proceeds remain investor cash and only execution costs are treated as economic loss. The crisis-resilient allocation uses inverse volatility, beta penalties, liquidity scores, and a feasible position cap. It is an analytical comparison, not an investment recommendation. Report files are generated locally in memory; no external AI or reporting API is called.

## Disclaimer

Results are model-based estimates derived from historical data and stated assumptions. They are not forecasts, guarantees, regulatory risk measures, or investment advice.
