# Launch: uvicorn api.main:app --reload --port 8000
# (run from project root, not from api/)

from fastapi import FastAPI, HTTPException, Query
from datetime import date
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="ChipChainReaction API",
    description=(
        "Quantitative API: 3-Layer system - "
        "HMM market regime detection, LSTM crash shield, "
        "and constrained portfolio optimisation (Markowitz / Risk Parity)."
    ),
    version="1.0.0"
)

# ── Data loading ───────────────────────────────────────────────────────────────
# Robust path: works whether uvicorn is launched from root or from api/
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
API_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "api_data")

if not os.path.exists(API_DATA_DIR):
    raise RuntimeError(
        f"Directory not found: {API_DATA_DIR}. "
        "Run Notebook 13 (Master Pipeline) first to generate the pre-computed data."
    )

# Global DataFrames (loaded once at startup for instant responses)
df_inputs     = pd.DataFrame()
df_equity_with = pd.DataFrame()
df_equity_no   = pd.DataFrame()
df_weights_with = {}
df_weights_no   = {}

STRATEGIES = [
    "Markowitz_LongOnly",
    "Markowitz_LongShort",
    "RiskParity_LongOnly",
    "RiskParity_LongShort",
]

@app.on_event("startup")
def load_data():
    """Load all CSVs at startup for instant API responses."""
    global df_inputs, df_equity_with, df_equity_no
    global df_weights_with, df_weights_no

    # Inputs (HMM regime, LSTM signal, etc.)
    df_inputs = pd.read_csv(os.path.join(API_DATA_DIR, "inputs_oracles.csv"))
    df_inputs['Date'] = pd.to_datetime(df_inputs['Date']).dt.date

    # Pre-computed equity curves
    df_equity_with = pd.read_csv(os.path.join(API_DATA_DIR, "equity_with_lstm.csv"))
    df_equity_with['Date'] = pd.to_datetime(df_equity_with['Date']).dt.date

    df_equity_no = pd.read_csv(os.path.join(API_DATA_DIR, "equity_no_lstm.csv"))
    df_equity_no['Date'] = pd.to_datetime(df_equity_no['Date']).dt.date

    # Pre-computed daily weights per strategy
    for strat in STRATEGIES:
        path_with = os.path.join(API_DATA_DIR, f"weights_with_lstm_{strat}.csv")
        path_no   = os.path.join(API_DATA_DIR, f"weights_no_lstm_{strat}.csv")
        if os.path.exists(path_with):
            df_w = pd.read_csv(path_with)
            df_w['Date'] = pd.to_datetime(df_w['Date']).dt.date
            df_weights_with[strat] = df_w
        if os.path.exists(path_no):
            df_w = pd.read_csv(path_no)
            df_w['Date'] = pd.to_datetime(df_w['Date']).dt.date
            df_weights_no[strat] = df_w


# ── Utility ───────────────────────────────────────────────────────────────────
def calculate_financial_metrics(equity_series: pd.Series) -> dict:
    """
    Compute the 3 fundamental KPIs from an equity curve (portfolio value in $).

    Returns:
        Total_Return_pct  : cumulative return in %
        Sharpe_Ratio      : annualised Sharpe ratio (risk-free rate = 0)
        Max_Drawdown_pct  : worst peak-to-trough drawdown in % (negative value)
    """
    initial_val = equity_series.iloc[0]
    final_val   = equity_series.iloc[-1]
    total_return = (final_val / initial_val) - 1.0

    daily_returns = equity_series.pct_change().dropna()

    if not daily_returns.empty and daily_returns.std() != 0:
        mean_ann = daily_returns.mean() * 252
        vol_ann  = daily_returns.std()  * np.sqrt(252)
        sharpe   = mean_ann / vol_ann
    else:
        sharpe = 0.0

    equity_curve = equity_series / initial_val
    running_max  = equity_curve.cummax()
    drawdowns    = (equity_curve - running_max) / running_max
    max_dd       = drawdowns.min()

    return {
        "Total_Return_pct":  round(total_return * 100, 2),
        "Sharpe_Ratio":      round(sharpe, 3),
        "Max_Drawdown_pct":  round(max_dd * 100, 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "project": "ChipChainReaction",
        "version": "1.0.0",
        "routes": ["/backtest_metrics", "/allocation", "/docs"]
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/backtest_metrics", tags=["Backtest"])
def get_backtest_metrics(
    start_date: date = Query(..., description="Investment start date (YYYY-MM-DD)"),
    end_date:   date = Query(..., description="Investment end date (YYYY-MM-DD)"),
    strategy:   str  = Query("Markowitz_LongOnly", enum=STRATEGIES,
                              description="Portfolio optimisation strategy")
):
    """
    Simulate an investment between two dates and compare performance:
    WITH LSTM Shield vs WITHOUT LSTM (aggressive).

    Returns cumulative return, annualised Sharpe ratio, and Max Drawdown
    for both portfolios, plus an automated conclusion.
    """
    # 1. Date consistency check
    if start_date >= end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be strictly before end_date."
        )

    # 2. Retrieve global DataFrames
    df_with = df_equity_with[['Date', strategy]]
    df_no   = df_equity_no[['Date', strategy]]

    # 3. Filter on requested period
    mask_with = (df_with['Date'] >= start_date) & (df_with['Date'] <= end_date)
    mask_no   = (df_no['Date']   >= start_date) & (df_no['Date']   <= end_date)

    period_data_with = df_with.loc[mask_with, strategy]
    period_data_no   = df_no.loc[mask_no,     strategy]

    # 4. Guard: need at least 2 data points to compute metrics
    if len(period_data_with) < 2 or len(period_data_no) < 2:
        raise HTTPException(
            status_code=404,
            detail=(
                "Date range too short or not found in the dataset. "
                "At least 2 trading days are required."
            )
        )

    # 5. Compute metrics
    metrics_with = calculate_financial_metrics(period_data_with)
    metrics_no   = calculate_financial_metrics(period_data_no)

    # 6. Automated conclusion
    # Note: drawdowns are negative (e.g. -10 > -20 means LESS severe)
    lstm_reduced_drawdown = (
        metrics_with["Max_Drawdown_pct"] > metrics_no["Max_Drawdown_pct"]
    )
    if lstm_reduced_drawdown:
        conclusion = (
            f"The LSTM shield was useful: it reduced Max Drawdown from "
            f"{metrics_no['Max_Drawdown_pct']}% to {metrics_with['Max_Drawdown_pct']}%, "
            f"at the cost of {metrics_no['Total_Return_pct'] - metrics_with['Total_Return_pct']:.2f}pp "
            f"in return."
        )
    else:
        conclusion = (
            f"The market was too bullish: the LSTM shield created unnecessary opportunity cost "
            f"({metrics_with['Total_Return_pct']}% vs {metrics_no['Total_Return_pct']}% without LSTM). "
            f"Consider the Aggressive portfolio in strong uptrends."
        )

    return {
        "period": {
            "start_requested":       str(start_date),
            "end_requested":         str(end_date),
            "trading_days_analyzed": len(period_data_with)
        },
        "strategy_analyzed": strategy,
        "performance_comparison": {
            "WITH_LSTM_Shield":       metrics_with,
            "WITHOUT_LSTM_Aggressive": metrics_no
        },
        "conclusion_bot": conclusion
    }


@app.get("/allocation", tags=["Allocation"])
def get_allocation(
    target_date: date = Query(..., description="Target date (YYYY-MM-DD)"),
    strategy:    str  = Query("Markowitz_LongOnly", enum=STRATEGIES),
    use_lstm:    bool = Query(True, description="Whether to apply the LSTM shield")
):
    """
    Return the recommended portfolio allocation on a given date,
    using the 3-Layer system (HMM + LSTM + optimiser).

    If the requested date is not a trading day, the closest available
    prior date is used automatically.
    """
    df_weights = df_weights_with if use_lstm else df_weights_no

    if strategy not in df_weights:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Weight file for strategy '{strategy}' not found. "
                "Re-run Notebook 13 to regenerate pre-computed weights."
            )
        )

    df_w = df_weights[strategy]

    # Find closest available date (last trading day on or before target_date)
    available_dates = df_w['Date'].tolist()
    valid_dates = [d for d in available_dates if d <= target_date]

    if not valid_dates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trading data available on or before {target_date}. "
                f"Earliest available date: {min(available_dates)}."
            )
        )

    actual_date = max(valid_dates)
    row = df_w[df_w['Date'] == actual_date].iloc[0]

    # Extract weights (all columns except 'Date')
    weight_cols = [c for c in df_w.columns if c != 'Date']
    weights_only = {
        col: round(float(row[col]) * 100, 2)
        for col in weight_cols
    }

    return {
        "requested_date":    str(target_date),
        "actual_market_date": str(actual_date),
        "strategy_applied":  strategy,
        "lstm_shield_active": use_lstm,
        "allocation":        weights_only
    }