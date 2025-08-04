import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta

# -------------------------------------------------------------
# ⚙️  Configuración global de la página
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros WACC por defecto (ajustables en el sidebar)
# -------------------------------------------------------------
Rf = 0.0435   # Tasa libre de riesgo
Rm = 0.085    # Retorno esperado del mercado
Tc = 0.21     # Tasa impositiva corporativa

# -------------------------------------------------------------
# 🧮  Función para calcular WACC y ROIC
# -------------------------------------------------------------
def calcular_wacc_y_roic(ticker: str,
                         rf: float | None = None,
                         rm: float | None = None,
                         tc: float | None = None):
    """
    Calcula WACC y ROIC usando únicamente datos de yfinance.
    Devuelve (wacc, roic, total_debt).
    Se incluye time.sleep(1) para evitar rate-limiting.
    """
    rf = rf if rf is not None else Rf
    rm = rm if rm is not None else Rm
    tc = tc if tc is not None else Tc

    time.sleep(1)  # Protección ante la API

    empresa = yf.Ticker(ticker)
    info = empresa.info
    bs   = empresa.balance_sheet
    fin  = empresa.financials

    # Coste de capital propio (Ke)
    beta = info.get("beta", 1)
    ke   = rf + beta * (rm - rf)

    # Balance
    total_debt = bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else 0
    cash_eq    = bs.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in bs.index else 0
    equity_bs  = bs.loc["Common Stock Equity"].iloc[0] if "Common Stock Equity" in bs.index else 0

    # Coste de la deuda (Kd)
    interest_expense = fin.loc["Interest Expense"].iloc[0] if "Interest Expense" in fin.index else 0
    kd = interest_expense / total_debt if total_debt else 0

    # Tasa efectiva
    ebt        = fin.loc["Ebt"].iloc[0] if "Ebt" in fin.index else 0
    taxes      = fin.loc["Income Tax Expense"].iloc[0] if "Income Tax Expense" in fin.index else 0
    tax_rate   = taxes / ebt if ebt else tc

    # WACC
    market_cap    = info.get("marketCap", 0)
    total_capital = market_cap + total_debt
    if total_capital == 0:
        return None, None, total_debt

    wacc = (market_cap / total_capital) * ke + (total_debt / total_capital) * kd * (1 - tax_rate)

    # ROIC
    ebit            = fin.loc["EBIT"].iloc[0] if "EBIT" in fin.index else 0
    nopat           = ebit * (1 - tax_rate)
    invested_capital = equity_bs + (total_debt - cash_eq)
    roic = nopat / invested_capital if invested_capital else None

    return wacc, roic, total_debt

# -------------------------------------------------------------
# 📦  Descarga y procesamiento de datos de un ticker
# -------------------------------------------------------------
def obtener_datos_financieros(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        bs    = stock.balance_sheet
        fin   = stock.financials
        cf    = stock.cashflow

        # Datos básicos
        price    = info.get("currentPrice")
        name     = info.get("longName", ticker)
        sector   = info.get("sector", "N/D")
        country  = info.get("country", "N/D")
        industry = info.get("industry", "N/D")

        # Valoración
        pe       = info.get("trailingPE")
        pb       = info.get("priceToBook")
        dividend = info.get("dividendRate")
        div_yld  = info.get("dividendYield")
        payout   = info.get("payoutRatio")

        # Rentabilidad
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")

        # Liquidez
        current_ratio = info.get("currentRatio")
        quick_ratio   = info.get("quickRatio")

        # Deuda
        ltde = info.get("longTermDebtToEquity")
        de   = info.get("debtToEquity")

        # Márgenes
        op_margin     = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        # Flujo de caja y P/FCF
        fcf    = cf.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cf.index else None
        shares = info.get("sharesOutstanding")
        pfcf   = price / (fcf / shares) if fcf and shares else None

        # Cálculo WACC y ROIC
        wacc, roic, total_debt = calcular_wacc_y_roic(ticker, Rf, Rm, Tc)

        # EVA
        equity_bs = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
        cap_inv   = total_debt + equity_bs if total_debt and equity_bs else None
        ebit      = fin.loc["EBIT"].iloc[0] if "EBIT" in fin.index else None
        eva       = (roic - wacc) * cap_inv if roic is not None and wacc is not None and cap_inv else None

        # CAGR helper
        def _cagr(df, row):
            if row not in df.index:
                return None
            serie = df.loc[row].dropna().iloc[:4]
            if len(serie) < 2 or serie.iloc[-1] == 0:
                return None
            return (serie.iloc[0] / serie.iloc[-1]) ** (1 / (len(serie) - 1)) - 1

        revenue_growth = _cagr(fin, "Total Revenue")
        eps_growth     = _cagr(fin, "Net Income")
        fcf_growth     = _cagr(cf,  "Free Cash Flow") or _cagr(cf, "Operating Cash Flow")

        # Liquidez avanzada
        cash_ratio          = info.get("cashRatio")
        operating_cf        = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        current_liabilities = bs.loc["Total Current Liabilities"].iloc[0] if "Total Current Liabilities" in bs.index else None
        cash_flow_ratio     = operating_cf / current_liabilities if operating_cf and current_liabilities else None

        return {
            # Identificación
            "Ticker": ticker,
            "Nombre": name,
            "Sector": sector,
            "País": country,
            "Industria": industry,
            # Precio & valoración
            "Precio": price,
            "P/E": pe,
            "P/B": pb,
            "P/FCF": pfcf,
            "Dividend Year": dividend,
            "Dividend Yield %": div_yld,
            "Payout Ratio": payout,
            # Rentabilidad
            "ROA": roa,
            "ROE": roe,
            # Liquidez & deuda
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "LtDebt/Eq": ltde,
            "Debt/Eq": de,
            # Márgenes
            "Oper Margin": op_margin,
            "Profit Margin": profit_margin,
            # Métricas avanzadas
            "WACC": wacc,
            "ROIC": roic,
            "EVA": eva,
            "Deuda Total": total_debt,
            "Patrimonio Neto": equity_bs,
            # Crecimiento
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            # Otros ratios
            "Cash Ratio": cash_ratio,
            "Cash Flow Ratio": cash_flow_ratio,
            "Operating Cash Flow": operating_cf,
            "Current Liabilities": current_liabilities,
        }

    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# -------------------------------------------------------------
# 🎛️  Interfaz principal
# -------------------------------------------------------------
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configuración")

        tickers_input = st.text_area(
            "🔎 Ingresa tickers (separados por coma)",
            "AAPL, MSFT, GOOGL, AMZN, TSLA",
            help="Ejemplo: AAPL, MSFT, GOOG",
        )
        max_tickers = st.slider("Número máximo de tickers", 1, 100, 50)

        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0) / 100

    # Lista limpia de tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:max_tickers]

    # ---------- Botón principal ----------
    if st.button("🔍 Analizar Acciones", type="primary"):
        if not tickers:
            st.warning("Por favor ingresa al menos un ticker")
            return

        resultados = {}
        progress_bar = st.progress(0)
        status_text  = st.empty()

        batch_size = 10
        for batch_start in range(0, len(tickers), batch_size):
            batch_end     = min(batch_start + batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]

            for i, t in enumerate(batch_tickers):
                status_text.text(f"⏳ Procesando {t} ({batch_start + i + 1}/{len(tickers)})…")
                resultados[t] = obtener_datos_financieros(t)
                progress_bar.progress((batch_start + i + 1) / len(tickers))
                time.sleep(1)  # 2ª defensa ante rate-limiting

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        # ---------- DataFrame limpio ----------
        datos = list(resultados.values())
        datos_validos = [d for d in datos if "Error" not in d]
        if not datos_validos:
            st.error("No se pudo obtener datos válidos para ningún ticker")
            return

        df = pd.DataFrame(datos_validos)

        # --------------------------------------------------
        # 1⃣  Resumen General
        # --------------------------------------------------
        st.header("📋 Resumen General")

        porcentaje_cols = [
            "Dividend Yield %", "ROA", "ROE", "Payout Ratio"
            "Oper Margin", "Profit Margin",
            "WACC", "ROIC", "EVA",
        ]
        for col in porcentaje_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/D")

        cols_show = [
            "Ticker", "Nombre", "Sector", "Precio",
            "P/E", "P/B", "P/FCF",
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
            "Current Ratio", "LtDebt/Eq", "Debt/Eq", "Oper Margin", "Profit Margin",
            "WACC", "ROIC",
        ]
        st.dataframe(df[cols_show].dropna(how="all", axis=1),
                     use_container_width=True, height=400)

        # --------------------------------------------------
        # 2⃣  Análisis de Valoración
        # --------------------------------------------------
        st.header("💰 Análisis de Valoración")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(10, 4))
            (df[["Ticker", "P/E", "P/B", "P/FCF"]]
             .set_index("Ticker")
             .apply(pd.to_numeric, errors="coerce")
             .plot(kind="bar", ax=ax, rot=45))
            ax.set_title("Comparativa de Ratios de Valoración")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Dividendos")
            fig, ax = plt.subplots(figsize=(10, 4))
            div_df = df[["Ticker", "Dividend Yield %"]].set_index("Ticker")
            div_df["Dividend Yield %"] = (
                div_df["Dividend Yield %"]
                .replace("N/D", 0)
                .str.rstrip("%")
                .astype(float)
            )
            div_df.plot(kind="bar", ax=ax, rot=45, color="green")
            ax.set_title("Rendimiento de Dividendos (%)")
            ax.set_ylabel("Dividend Yield %")
            st.pyplot(fig)
            plt.close()

        # --------------------------------------------------
        # 3⃣  Rentabilidad y Eficiencia
        # --------------------------------------------------
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            roe_roa = df[["Ticker", "ROE", "ROA"]].set_index("Ticker")
            roe_roa["ROE"] = roe_roa["ROE"].str.rstrip("%").astype(float)
            roe_roa["ROA"] = roe_roa["ROA"].str.rstrip("%").astype(float)
            roe_roa.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("ROE vs ROA (%)")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            mars = df[["Ticker", "Oper Margin", "Profit Margin"]].set_index("Ticker")
            mars["Oper Margin"]   = mars["Oper Margin"].str.rstrip("%").astype(float)
            mars["Profit Margin"] = mars["Profit Margin"].str.rstrip("%").astype(float)
            mars.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Margen Operativo vs Margen Neto (%)")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, row in df.iterrows():
                wacc_val = float(row["WACC"].rstrip("%")) if row["WACC"] != "N/D" else None
                roic_val = float(row["ROIC"].rstrip("%")) if row["ROIC"] != "N/D" else None
                if wacc_val is not None and roic_val is not None:
                    color = "green" if roic_val > wacc_val else "red"
                    ax.bar(row["Ticker"], roic_val, color=color, alpha=0.6, label="ROIC")
                    ax.bar(row["Ticker"], wacc_val, color="gray", alpha=0.3, label="WACC")
            ax.set_title("Creación de Valor: ROIC vs WACC (%)")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        # --------------------------------------------------
        # 4⃣  Estructura de Capital y Deuda
        # --------------------------------------------------
        st.header("🏦 Estructura de Capital y Deuda")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Apalancamiento (Debt/Eq)")
            fig, ax = plt.subplots(figsize=(10, 5))
            (df[["Ticker", "Debt/Eq", "LtDebt/Eq"]]
             .set_index("Ticker")
             .apply(pd.to_numeric, errors="coerce")
             .plot(kind="bar", stacked=True, ax=ax, rot=45)
            )
            ax.axhline(1, color="red", linestyle="--")
            ax.set_title("Deuda / Patrimonio")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(10, 5))
            (df[["Ticker", "Current Ratio", "Quick Ratio", "Cash Ratio"]]
             .set_index("Ticker")
             .apply(pd.to_numeric, errors="coerce")
             .plot(kind="bar", ax=ax, rot=45))
            ax.axhline(1, color="green", linestyle="--")
            ax.set_title("Ratios de Liquidez")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        # --------------------------------------------------
        # 5⃣  Crecimiento Histórico
        # --------------------------------------------------
        st.header("🚀 Crecimiento Histórico")
        growth_metrics = ["Revenue Growth", "EPS Growth", "FCF Growth"]
        growth_df = df[["Ticker"] + growth_metrics].set_index("Ticker") * 100  # %
        fig, ax = plt.subplots(figsize=(12, 6))
        growth_df.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Tasas de Crecimiento Anual (%)")
        ax.set_ylabel("%")
        st.pyplot(fig)
        plt.close()

        # --------------------------------------------------
        # 6⃣  Análisis Individual
        # --------------------------------------------------
        st.header("🔍 Análisis por Empresa")
        sel_ticker = st.selectbox("Selecciona una empresa", df["Ticker"].unique())
        empresa    = df[df["Ticker"] == sel_ticker].iloc[0]

        st.subheader(f"Análisis Detallado: {empresa['Nombre']}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Precio", f"${empresa['Precio']:,.2f}" if empresa['Precio'] else "N/D")
            st.metric("P/E", empresa["P/E"])
            st.metric("P/B", empresa["P/B"])

        with col2:
            st.metric("ROE",  empresa["ROE"])
            st.metric("ROIC", empresa["ROIC"])
            st.metric("WACC", empresa["WACC"])

        with col3:
            st.metric("Deuda/Patrimonio", empresa["Debt/Eq"])
            st.metric("Margen Neto",       empresa["Profit Margin"])
            st.metric("Dividend Yield",    empresa["Dividend Yield %"])

        # Gráfico de creación de valor individual
        st.subheader("Creación de Valor")
        fig, ax = plt.subplots(figsize=(6, 4))
        if empresa["ROIC"] != "N/D" and empresa["WACC"] != "N/D":
            roic_val = float(empresa["ROIC"].rstrip("%"))
            wacc_val = float(empresa["WACC"].rstrip("%"))
            color    = "green" if roic_val > wacc_val else "red"
            ax.bar(["ROIC", "WACC"], [roic_val, wacc_val], color=[color, "gray"])
            ax.set_title("ROIC vs WACC")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

            if roic_val > wacc_val:
                st.success("✅ La empresa está creando valor (ROIC > WACC)")
            else:
                st.error("❌ La empresa está destruyendo valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para análisis ROIC/WACC")

# -------------------------------------------------------------
# 🚀  Lanzar la app
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
