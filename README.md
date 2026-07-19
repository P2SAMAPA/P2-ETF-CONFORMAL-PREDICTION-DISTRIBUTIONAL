# Conformal Predictive Distribution for ETFs

Produces full predictive distributions with finite-sample validity guarantees, not just prediction intervals. Extends conformal prediction to full distributions via quantile regression. The per‑ETF score is the predictive density at the observed next‑day return.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Quantile regression forests for predictive distributions
- Conformal predictive distribution with finite-sample guarantees
- Score = predictive density (higher = more accurate prediction)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-conformal-prediction-distributional-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to RF training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High predictive density → the model assigns high probability to the actual outcome.
- Low predictive density → the actual outcome was unlikely under the model.

## Requirements

See `requirements.txt`.
