# Portfolio Fragility Lab — Streamlit

Results-first web interface for multi-asset Monte Carlo portfolio analysis. Users enter tickers and weights, run an analysis, and receive interactive simulations, downside-risk metrics, historical comparisons, and downloadable reports without seeing notebook code.

## Local use

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this complete folder to a GitHub repository.
2. Sign in at `https://share.streamlit.io` using GitHub.
3. Choose **Create app** and select the repository.
4. Set the entrypoint to `streamlit_app.py`.
5. Choose **Deploy**.

Do not upload only the `streamlit_app.py` file: the `src` directory, `requirements.txt`, and `.streamlit/config.toml` are also required.

## Models

- Correlated Geometric Brownian Motion
- Historical vector bootstrap
- Correlated Student-t simulation

## Disclaimer

For educational and research purposes only. Results are probabilistic scenarios, not financial advice or guaranteed forecasts.
