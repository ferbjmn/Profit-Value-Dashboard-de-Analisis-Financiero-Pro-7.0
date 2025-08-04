import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time

# ------------- Configuración global de la página -------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Configuración")

tickers_input = st.sidebar.text_area(
    "🔎 Ingresa tickers (separados por coma)",
    "AAPL, MSFT, GOOGL, AMZN, TSLA",
    help="Ejemplo: AAPL, MSFT, GOOG",
)
max_tickers = st.sidebar.slider("Número máximo de tickers", 1, 100, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros de costo de capital**")

Rf = st.sidebar.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35) / 100
erp = st.sidebar.number_input("Prima de riesgo mercado (%)", 0.0, 20.0, 5.0) / 100  # NUEVO
Tc = st.sidebar.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0) / 100

# ------------------ Utilidades -------------------
def pct(x):
    """Convierte float a string porcentaje o N/D."""
    return f"{x:.2%}" if pd.notnull(x) else "N/D"

# ---------- WACC & ROIC con tu lógica ----------
def calcular_wacc_roic(info, bs, fin):
    """
    Devuelve wacc, roic, diff (roic - wacc) según lógica proporcionada.
    """
    # Valor de mercado del patrimonio
    market_cap = info.get("marketCap", 0)
    beta = info.get("beta", 1)

    ke = Rf + beta * erp  # CAPM

    deuda_total = bs.get("Total Debt", pd.Series([0])).iloc[0]
    efectivo = bs.get("Cash And Cash Equivalents", pd.Series([0])).iloc[0]
    patrimonio = bs.get("Common Stock Equity", pd.Series([0])).iloc[0]

    gastos_intereses = fin.get("Interest Expense", pd.Series([0])).iloc[0]
    ebt = fin.get("Ebt", pd.Series([np.nan])).iloc[0]
    impuestos = fin.get("Income Tax Expense", pd.Series([np.nan])).iloc[0]
    ebit = fin.get("EBIT", pd.Series([np.nan])).iloc[0]

    kd = gastos_intereses / deuda_total if deuda_total else 0
    tasa_imp = impuestos / ebt if pd.notnull(ebt) and ebt != 0 else Tc

    total_capital = market_cap + deuda_total if (market_cap + deuda_total) else np.nan
    wacc = ((market_cap / total_capital) * ke) + ((deuda_total / total_capital) * kd * (1 - tasa_imp)) if pd.notnull(total_capital) else np.nan

    nopat = ebit * (1 - tasa_imp) if pd.notnull(ebit) else np.nan
    capital_invertido = patrimonio + (deuda_total - efectivo)
    roic = nopat / capital_invertido if capital_invertido else np.nan

    diff = roic - wacc if pd.notnull(roic) and pd.notnull(wacc) else np.nan
    return wacc, roic, diff

# ------------------ CAGR ------------------------
def calcular_cagr(frame, metric):
    if metric not in frame.index:
        return np.nan
    serie = frame.loc[metric].dropna().iloc[:4]
    if len(serie) < 2 or serie.iloc[-1] == 0:
        return np.nan
    años = len(serie) - 1
    return (serie.iloc[0] / serie.iloc[-1]) ** (1 / años) - 1

# ------------- Datos financieros -----------------
def obtener_datos_financieros(ticker):
    try:
        stock = yf.Ticker(ticker)
        info, bs, fin, cf = stock.info, stock.balance_sheet, stock.financials, stock.cashflow
        time.sleep(0.5)  # evitar bloqueo

        price = info.get("currentPrice")
        fcf = cf.get("Free Cash Flow", pd.Series([np.nan])).iloc[0]
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf / shares) if fcf and shares else np.nan

        wacc, roic, diff = calcular_wacc_roic(info, bs, fin)

        rev_g = calcular_cagr(fin, "Total Revenue")
        eps_g = calcular_cagr(fin, "Net Income")
        fcf_g = calcular_cagr(cf, "Free Cash Flow") or calcular_cagr(cf, "Operating Cash Flow")

        return {
            "Ticker": ticker,
            "Nombre": info.get("longName", ticker),
            "Sector": info.get("sector", "N/D"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Yield %": info.get("dividendYield"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Debt/Eq": info.get("debtToEquity"),
            "Profit Margin": info.get("profitMargins"),
            "WACC": wacc,
            "ROIC": roic,
            "Creación valor": diff,
            "Rev Growth": rev_g,
            "EPS Growth": eps_g,
            "FCF Growth": fcf_g,
        }
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# =================================================
#                     APP
# =================================================
st.title("📊 Dashboard de Análisis Financiero Avanzado")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:max_tickers]

if st.button("🔍 Analizar Acciones", type="primary"):
    if not tickers:
        st.warning("Ingresa al menos un ticker válido.")
        st.stop()

    resultados, errores = [], []
    progreso = st.progress(0)
    for i, tk in enumerate(tickers, start=1):
        data = obtener_datos_financieros(tk)
        if "Error" in data:
            errores.append(f"{tk}: {data['Error']}")
        else:
            resultados.append(data)
        progreso.progress(i / len(tickers))
    progreso.empty()

    if errores:
        with st.expander("❌ Errores", expanded=False):
            st.write("\n".join(errores))

    if resultados:
        df = pd.DataFrame(resultados)

        # Formateo de porcentajes
        pct_cols = [
            "Yield %", "ROE", "ROA", "Profit Margin",
            "WACC", "ROIC", "Creación valor",
            "Rev Growth", "EPS Growth", "FCF Growth"
        ]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(pct)

        # ---------------- Resumen -----------------
        st.header("📋 Resumen General")
        cols_show = [
            "Ticker", "Nombre", "Sector", "Precio",
            "P/E", "P/B", "P/FCF",
            "Yield %", "ROE", "Debt/Eq",
            "Profit Margin", "WACC", "ROIC", "Creación valor"
        ]
        st.dataframe(df[cols_show], use_container_width=True, height=400)

        # ------------- Gráfico ROIC vs WACC -------------
        st.header("📈 Creación de Valor (ROIC vs WACC)")
        graf = df[["Ticker", "ROIC", "WACC"]].copy().dropna()
        graf[["ROIC", "WACC"]] = graf[["ROIC", "WACC"]].apply(
            lambda col: col.str.rstrip("%").astype(float))

        fig, ax = plt.subplots(figsize=(10, 5))
        for _, r in graf.iterrows():
            color = "green" if r["ROIC"] > r["WACC"] else "red"
            ax.bar(r["Ticker"], r["ROIC"], color=color, alpha=0.6)
            ax.bar(r["Ticker"], r["WACC"], color="gray", alpha=0.3)
        ax.set_ylabel("%")
        st.pyplot(fig)

        # ------------- Análisis individual --------------
        st.header("🔍 Análisis por Empresa")
        selected = st.selectbox("Selecciona una empresa", df["Ticker"])
        fila = df[df["Ticker"] == selected].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${fila['Precio']:,.2f}" if pd.notnull(fila['Precio']) else "N/D")
        c1.metric("P/E", fila["P/E"])
        c1.metric("P/B", fila["P/B"])

        c2.metric("ROIC", fila["ROIC"])
        c2.metric("WACC", fila["WACC"])
        c2.metric("Creación valor", fila["Creación valor"])

        c3.metric("Debt/Eq", fila["Debt/Eq"])
        c3.metric("Margen Neto", fila["Profit Margin"])
        c3.metric("Dividend Yield", fila["Yield %"])
