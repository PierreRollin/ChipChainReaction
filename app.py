import os
import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

# Robust path: works regardless of the directory Streamlit is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="ChipChainReaction",
    page_icon="⚡",
    layout="wide"
)

# ── Strategies ────────────────────────────────────────────────────────────────
STRATEGIES = [
    "Markowitz_LongOnly",
    "Markowitz_LongShort",
    "RiskParity_LongOnly",
    "RiskParity_LongShort",
]

STRATEGY_LABELS = {
    "Markowitz_LongOnly":   "Markowitz - Long Only",
    "Markowitz_LongShort":  "Markowitz - Long/Short (±30%)",
    "RiskParity_LongOnly":  "Risk Parity - Long Only",
    "RiskParity_LongShort": "Risk Parity - Long/Short (±30%)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ ChipChainReaction")
st.sidebar.caption("Semiconductor Supply Chain · Quantitative Trading")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Backtest", "📐 Daily Allocation"]
)

def call_api(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            "❌ Cannot connect to the API. "
            "Run `uvicorn api.main:app --reload` in a separate terminal."
        )
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get('detail', str(e))
        return None, f"❌ API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, f"❌ Unexpected error: {e}"

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("⚡ ChipChainReaction")
    st.markdown(
        "**Full-stack quantitative trading system · "
        "Semiconductor Supply Chain · "
        "HMM + LSTM + Markowitz/Risk Parity**"
    )
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Decision Pipeline")
        st.markdown("""
        **Layer 1 - Macro Regime Detection (HMM)**  
        A Hidden Markov Model classifies the market into three latent regimes
        (*Expansion, Consolidation, Contraction*) from VIX and TNX signals.
        Each regime produces a dedicated covariance matrix for the optimiser.
        Walk-Forward expanding window, no Viterbi to prevent look-ahead bias.

        **Layer 2 - Sector Crash Signal (LSTM)**  
        A CNN1D + LSTM generates a binary early-warning signal for the
        Fabless sub-sector (NVDA, AMD) from 21-day supply chain tensors.
        When active, it forces net exposure to zero on flagged assets
        before the optimiser runs ("emergency brake").

        **Layer 3 - Constrained Portfolio Optimisation**  
        Scipy SLSQP solves the mean-variance problem using the 
        regime-specific covariance matrix under the LSTM position constraints.
        Four strategies: Markowitz and Risk Parity, each in Long-Only 
        or Long/Short variant.
        """)

    with col_b:
        st.subheader("Key Results : Ablation Test")
        st.markdown("""
        The ablation test compares every strategy **with** and **without**
        the LSTM shield, validated by Walk-Forward Validation.

        **The LSTM is not an alpha generator : it is a crash shield.**

        In a strongly bullish market (AI bull run 2023–2026), the pure
        mathematical model outperforms in absolute return. The LSTM
        creates an opportunity cost through its caution.

        **However**, in high-risk strategies (Markowitz Long/Short with
        30% short exposure), the LSTM becomes critical:
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Drawdown - without LSTM", "−32.00%",
                      delta="Markowitz L/S", delta_color="off")
            st.metric("Sharpe Ratio - without LSTM", "1.14")
        with col2:
            st.metric("Max Drawdown - with LSTM", "−13.69%",
                      delta="−18.31pp reduction ✓", delta_color="normal")
            st.metric("Sharpe Ratio - with LSTM", "1.81",
                      delta="+0.67 ✓", delta_color="normal")

        st.info(
            "**Conclusion:** In quantitative finance, complex AI (Deep Learning) "
            "is often more valuable as a dynamic risk management tool (hedge) "
            "than as a directional speculation engine."
        )

    st.divider()
    st.subheader("Supply Chain Universe (18 tickers)")
    st.markdown("""
    | Layer | Tickers | Role |
    |---|---|---|
    | Commodities & Macro | USO, CGW, PICK, ^VIX, ^TNX | Raw materials + macro signals |
    | Foundries & Equipment | ASML, TSM, 0981.HK (SMIC) | Manufacturing layer |
    | Fabless Designers | NVDA, AMD, INTC, AVGO | Chip design (LSTM target) |
    | Integrators & AI | AAPL, MSFT, GOOGL, TSLA | End consumers |
    | Inverse Hedge | GLD, TLT | Counter-cyclical assets |
    """)

# ── BACKTEST ──────────────────────────────────────────────────────────────────
elif page == "📊 Backtest":
    st.title("📊 Backtest : LSTM Shield vs Aggressive Portfolio")
    st.divider()

    # Load equity curves locally (fast - avoids re-fetching on every interaction)
    try:
        df_with = pd.read_csv(
            os.path.join(BASE_DIR, "data", "api_data", "equity_with_lstm.csv")
        )
        df_no = pd.read_csv(
            os.path.join(BASE_DIR, "data", "api_data", "equity_no_lstm.csv")
        )
        df_with['Date'] = pd.to_datetime(df_with['Date']).dt.date
        df_no['Date']   = pd.to_datetime(df_no['Date']).dt.date
        DATA_LOADED = True
    except FileNotFoundError:
        st.warning(
            "Equity curve files not found in `data/api_data/`. "
            "Run Notebook 13 (Master Pipeline) first to generate them."
        )
        DATA_LOADED = False

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_strat = st.selectbox(
            "Strategy",
            options=STRATEGIES,
            format_func=lambda x: STRATEGY_LABELS[x]
        )
    with col2:
        min_date = date(2018, 1, 1)
        max_date = date.today()
        start_date = st.date_input(
            "Start date", value=date(2020, 1, 1),
            min_value=min_date, max_value=max_date
        )
    with col3:
        end_date = st.date_input(
            "End date", value=date.today(),
            min_value=min_date, max_value=max_date
        )

    if st.button("🚀 Run Backtest", type="primary"):
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            # Fetch metrics from API
            data, err = call_api("/backtest_metrics", {
                "start_date": str(start_date),
                "end_date":   str(end_date),
                "strategy":   selected_strat
            })

            if err:
                st.error(err)
            else:
                metrics_with = data['performance_comparison']['WITH_LSTM_Shield']
                metrics_no   = data['performance_comparison']['WITHOUT_LSTM_Aggressive']

                st.subheader(
                    f"Results : {STRATEGY_LABELS[selected_strat]} "
                    f"({data['period']['trading_days_analyzed']} trading days)"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.info("🛡 PROTECTED PORTFOLIO (With LSTM Shield)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Return",       f"{metrics_with['Total_Return_pct']}%")
                    c2.metric("Sharpe Ratio",  metrics_with['Sharpe_Ratio'])
                    c3.metric("Max Drawdown", f"{metrics_with['Max_Drawdown_pct']}%",
                              delta="Risk", delta_color="off")

                with col2:
                    st.warning("⚡ AGGRESSIVE PORTFOLIO (Without LSTM)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Return",       f"{metrics_no['Total_Return_pct']}%")
                    c2.metric("Sharpe Ratio",  metrics_no['Sharpe_Ratio'])
                    c3.metric("Max Drawdown", f"{metrics_no['Max_Drawdown_pct']}%",
                              delta="Risk", delta_color="off")

                # Conclusion from API
                conclusion = data['conclusion_bot']
                st.success(f"**Model conclusion:** {conclusion}")

                # Equity curve chart
                if DATA_LOADED:
                    st.markdown("---")
                    st.subheader("Equity Curves (Portfolio Value)")

                    mask_with = (df_with['Date'] >= start_date) & (df_with['Date'] <= end_date)
                    mask_no   = (df_no['Date']   >= start_date) & (df_no['Date']   <= end_date)

                    if mask_with.sum() < 2 or mask_no.sum() < 2:
                        st.warning("Not enough data in this date range to plot equity curves.")
                    else:
                        chart_data = pd.DataFrame({
                            "With LSTM (Protected)":   df_with.loc[mask_with, selected_strat].values,
                            "Without LSTM (Aggressive)": df_no.loc[mask_no,   selected_strat].values,
                        }, index=df_with.loc[mask_with, 'Date'])

                        st.line_chart(chart_data, color=["#1f77b4", "#ff7f0e"])

# ── DAILY ALLOCATION ──────────────────────────────────────────────────────────
elif page == "📐 Daily Allocation":
    st.title("📐 Daily Portfolio Allocation")
    st.caption(
        "Query the 3-Layer system for the recommended allocation on a given date."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        alloc_date = st.date_input(
            "Target date",
            value=date.today() - timedelta(days=1),
            max_value=date.today()
        )
    with col2:
        alloc_strat = st.selectbox(
            "Strategy",
            options=STRATEGIES,
            format_func=lambda x: STRATEGY_LABELS[x],
            key="alloc_strat"
        )
    with col3:
        use_lstm = st.toggle("Activate LSTM Shield", value=True)

    if st.button("📡 Get Allocation", type="primary"):
        data, err = call_api("/allocation", {
            "target_date": str(alloc_date),
            "strategy":    alloc_strat,
            "use_lstm":    str(use_lstm).lower()
        })

        if err:
            st.error(err)
        else:
            st.subheader(
                f"Allocation : {STRATEGY_LABELS[alloc_strat]} "
                f"({'LSTM active' if data['lstm_shield_active'] else 'No LSTM'}) "
                f"- {data['actual_market_date']}"
            )

            if data['requested_date'] != data['actual_market_date']:
                st.caption(
                    f"Note: {data['requested_date']} is not a trading day. "
                    f"Using closest available date: {data['actual_market_date']}."
                )

            alloc = data['allocation']
            df_alloc = pd.DataFrame(
                list(alloc.items()), columns=["Ticker", "Weight (%)"]
            ).sort_values("Weight (%)", ascending=False)
            df_alloc["Weight (%)"] = df_alloc["Weight (%)"].apply(
                lambda x: f"{x:+.2f}%"
            )

            col_table, col_bar = st.columns([1, 2])
            with col_table:
                st.dataframe(df_alloc, use_container_width=True, hide_index=True)
            with col_bar:
                raw_alloc = pd.DataFrame(
                    list(alloc.items()), columns=["Ticker", "Weight"]
                ).sort_values("Weight", ascending=True)
                st.bar_chart(raw_alloc.set_index("Ticker"))