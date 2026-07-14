# Portfolio Fragility Lab

A results-first institutional portfolio stress-testing application. The platform goes beyond ordinary terminal-value Monte Carlo analysis by modeling market regimes, crisis correlation, investor redemptions, margin calls, forced liquidation, trading frictions, liquidity shortfalls, and recovery behavior.

## Distinctive features

- Bull, normal, crisis, and recovery regimes driven by a Markov transition process
- Crisis volatility multipliers and correlations that converge during stress
- Asset liquidity scores derived from median dollar volume and volatility
- Investor redemption and margin-call triggers
- Forced-sale logic with cash buffers and daily liquidation capacity
- Bid–ask spread, slippage, and square-root market-impact proxies
- Stylized 2008, March 2020, and 2022 stress scenarios
- Liquidity-shortfall and forced-sale probabilities
- Median time-to-recovery and recovery probability
- Transparent 0–100 Portfolio Fragility Score
- Current versus crisis-resilient allocation comparison
- GBM, historical bootstrap, Student-t, and regime-switching simulations
- Interactive Plotly dashboards and downloadable Excel/CSV reports

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

The stress scenarios are educational approximations based on broad historical market behavior. Liquidity scores, spreads, liquidation capacity, and market impact are transparent proxies rather than security-level dealer quotes. The crisis-resilient allocation uses inverse volatility, beta penalties, liquidity scores, and a 45% position cap. It is an analytical comparison, not an investment recommendation.

## Disclaimer

For educational and research purposes only. Historical estimates and simulated scenarios do not guarantee future results and do not constitute investment advice.
