diff --git a/scanner.py b/scanner.py
index e3692ba62408509e8f6c97682388055cc9f71692..03891d6460694d96162ff77cfe0c8a0cda6829ab 100644
--- a/scanner.py
+++ b/scanner.py
@@ -1,129 +1,201 @@
-import yfinance as yf
-import pandas as pd
-import smtplib
-import os
-from email.mime.text import MIMEText
-from datetime import datetime
-
-WATCHLIST = ["NVDA", "AVGO", "AMD", "ALAB", "AI", "COIN"]
-MARKET = "QQQ"
-
-
-# -----------------------------------
-# DATA FUNCTIONS
-# -----------------------------------
-
-def get_data(ticker):
-    data = yf.download(ticker, period="1y", interval="1d", progress=False)
-
-    if data.empty:
-        return None
-
-    data["200DMA"] = data["Close"].rolling(200).mean()
-    data["50DMA"] = data["Close"].rolling(50).mean()
-
-    return data
-
-
-def market_ok():
-    data = get_data(MARKET)
+from __future__ import annotations
 
-    if data is None or len(data) < 200:
-        return False
+from dataclasses import dataclass
+from typing import Dict, List
 
-    latest = data.iloc[-1]
-
-    if pd.isna(latest["200DMA"]):
-        return False
-
-    return latest["Close"] > latest["200DMA"]
-
-
-def check_stock(ticker):
-    data = get_data(ticker)
-
-    if data is None or len(data) < 200:
-        return {
-            "ticker": ticker,
-            "price": "N/A",
-            "buy": False,
-            "exit": False
-        }
-
-    latest = data.iloc[-1]
-
-    dma_200_today = latest["200DMA"]
-    dma_200_20d_ago = data["200DMA"].iloc[-20]
-
-    if pd.isna(dma_200_today) or pd.isna(dma_200_20d_ago):
-        return {
-            "ticker": ticker,
-            "price": round(latest["Close"], 2),
-            "buy": False,
-            "exit": False
-        }
+import pandas as pd
+import yfinance as yf
 
-    buy = (
-        latest["Close"] > dma_200_today and
-        dma_200_today > dma_200_20d_ago and
-        latest["Close"] > latest["50DMA"]
+# ----------------------------
+# CONFIG
+# ----------------------------
+MONTHLY_CONTRIBUTION_GBP = 200.0
+LOOKBACK_YEARS = 7  # includes warmup period for 200DMA
+BACKTEST_YEARS = 5
+
+# US mega caps commonly available on UK platforms
+MEGA_CAP_TICKERS = [
+    "AAPL",
+    "MSFT",
+    "NVDA",
+    "AMZN",
+    "GOOGL",
+    "META",
+    "BRK-B",
+    "LLY",
+    "AVGO",
+    "TSM",
+]
+
+# Optional market regime filter
+USE_MARKET_FILTER = True
+MARKET_PROXY = "SPY"
+
+
+@dataclass
+class BacktestResult:
+    cagr: float
+    invested: float
+    final_value: float
+    max_drawdown: float
+    monthly_contribution_count: int
+
+
+def download_close_prices(tickers: List[str], years: int = LOOKBACK_YEARS) -> pd.DataFrame:
+    data = yf.download(
+        tickers,
+        period=f"{years}y",
+        interval="1d",
+        auto_adjust=True,
+        progress=False,
     )
 
-    exit_risk = (
-        latest["Close"] < dma_200_today or
-        latest["Close"] < latest["50DMA"]
+    if data.empty:
+        raise ValueError("No data returned from yfinance.")
+
+    if isinstance(data.columns, pd.MultiIndex):
+        close = data["Close"].copy()
+    else:
+        close = data.rename(columns={"Close": tickers[0]})[[tickers[0]]].copy()
+
+    close = close.dropna(how="all").ffill()
+    return close
+
+
+def build_signals(close: pd.DataFrame) -> pd.DataFrame:
+    """
+    Signal = price above 200DMA AND 200DMA rising vs 20 trading days ago.
+    """
+    dma_200 = close.rolling(200).mean()
+    above_200 = close > dma_200
+    dma_rising = dma_200 > dma_200.shift(20)
+    return above_200 & dma_rising
+
+
+def max_drawdown(equity_curve: pd.Series) -> float:
+    roll_max = equity_curve.cummax()
+    dd = equity_curve / roll_max - 1.0
+    return float(dd.min()) if not dd.empty else 0.0
+
+
+def run_backtest(
+    close: pd.DataFrame,
+    signals: pd.DataFrame,
+    monthly_contribution: float = MONTHLY_CONTRIBUTION_GBP,
+    years: int = BACKTEST_YEARS,
+) -> BacktestResult:
+    if close.empty:
+        raise ValueError("Close prices are empty.")
+
+    end_date = close.index.max()
+    start_date = end_date - pd.DateOffset(years=years)
+
+    px = close.loc[close.index >= start_date].copy()
+    sig = signals.loc[px.index].copy()
+
+    if len(px) < 252:
+        raise ValueError("Not enough data for a meaningful backtest window.")
+
+    # First trading day of each month (contribution day)
+    contribution_days = px.index.to_series().groupby(px.index.to_period("M")).min().values
+
+    # Portfolio state
+    cash = 0.0
+    shares: Dict[str, float] = {t: 0.0 for t in px.columns}
+    equity_points = []
+
+    for day in px.index:
+        prices = px.loc[day]
+        tradable = prices.dropna().index.tolist()
+
+        # Exit positions whose signal turned off
+        for t in tradable:
+            if shares[t] > 0 and not bool(sig.at[day, t]):
+                cash += shares[t] * prices[t]
+                shares[t] = 0.0
+
+        # Add monthly contribution and buy active signals equally
+        if day in contribution_days:
+            cash += monthly_contribution
+            active = [t for t in tradable if bool(sig.at[day, t])]
+
+            if active and cash > 0:
+                per_name = cash / len(active)
+                for t in active:
+                    shares[t] += per_name / prices[t]
+                cash = 0.0
+
+        equity = cash + sum(shares[t] * prices[t] for t in tradable)
+        equity_points.append((day, equity))
+
+    equity_curve = pd.Series({d: v for d, v in equity_points}).sort_index()
+    invested = monthly_contribution * len(contribution_days)
+    final_value = float(equity_curve.iloc[-1])
+
+    period_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
+    cagr = (final_value / invested) ** (1 / period_years) - 1 if invested > 0 and period_years > 0 else 0.0
+
+    return BacktestResult(
+        cagr=float(cagr),
+        invested=float(invested),
+        final_value=final_value,
+        max_drawdown=max_drawdown(equity_curve),
+        monthly_contribution_count=len(contribution_days),
     )
 
-    return {
-        "ticker": ticker,
-        "price": round(latest["Close"], 2),
-        "buy": buy,
-        "exit": exit_risk
-    }
 
+def latest_signal_table(close: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
+    d = close.index[-1]
+    return pd.DataFrame({"Price": close.loc[d], "Trend_OK": signals.loc[d]}).sort_index()
 
-# -----------------------------------
-# EMAIL FUNCTION
-# -----------------------------------
 
-def send_email(body):
-    sender = os.getenv("EMAIL_ADDRESS")
-    password = os.getenv("EMAIL_PASSWORD")
+def apply_market_filter(stock_signals: pd.DataFrame) -> pd.DataFrame:
+    if not USE_MARKET_FILTER:
+        return stock_signals
 
-    if not sender or not password:
-        raise ValueError("Email credentials missing. Check GitHub Secrets.")
+    market_close = download_close_prices([MARKET_PROXY], years=LOOKBACK_YEARS)
+    market_signal = build_signals(market_close).iloc[:, 0]  # bool series
 
-    msg = MIMEText(body)
-    msg["Subject"] = "Daily 200DMA Trading Report"
-    msg["From"] = sender
-    msg["To"] = sender
+    # Align and filter: only allow stock signals when market is in trend
+    aligned = stock_signals.copy()
+    aligned = aligned.loc[aligned.index.intersection(market_signal.index)]
+    for day in aligned.index:
+        if not bool(market_signal.loc[day]):
+            aligned.loc[day, :] = False
+    return aligned
 
-    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
-        server.login(sender, password)
-        server.sendmail(sender, sender, msg.as_string())
 
+def main() -> None:
+    close = download_close_prices(MEGA_CAP_TICKERS, years=LOOKBACK_YEARS)
+    signals = build_signals(close)
+    signals = apply_market_filter(signals)
 
-# -----------------------------------
-# MAIN RUN
-# -----------------------------------
+    result = run_backtest(close.loc[signals.index], signals)
+    table = latest_signal_table(close.loc[signals.index], signals)
 
-def run():
-    report = f"Daily 200DMA Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
+    print("\n=== 200DMA Mega-Cap Strategy (UK ISA style £200/month) ===")
+    print(f"Universe: {', '.join(MEGA_CAP_TICKERS)}")
+    print(f"Monthly contribution: £{MONTHLY_CONTRIBUTION_GBP:.2f}")
+    print(f"Backtest years: {BACKTEST_YEARS}")
+    print(f"Market filter: {'ON' if USE_MARKET_FILTER else 'OFF'} ({MARKET_PROXY})")
 
-    regime = market_ok()
-    report += f"Market Regime OK (QQQ > 200DMA): {regime}\n\n"
+    print("\n--- Backtest Summary ---")
+    print(f"Invested: £{result.invested:,.2f}")
+    print(f"Final Value: £{result.final_value:,.2f}")
+    print(f"CAGR: {result.cagr * 100:.2f}%")
+    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
+    print(f"Contribution months: {result.monthly_contribution_count}")
 
-    for ticker in WATCHLIST:
-        result = check_stock(ticker)
-        report += (
-            f"{result['ticker']} | "
-            f"Price: {result['price']} | "
-            f"BUY: {result['buy']} | "
-            f"EXIT: {result['exit']}\n"
-        )
+    print("\n--- Latest Signals ---")
+    print(table.to_string(float_format=lambda x: f"{x:,.2f}"))
 
-    send_email(report)
+    if result.cagr < 0.20:
+        print("\nNote: CAGR is below 20%.")
+    elif result.cagr > 0.30:
+        print("\nNote: CAGR is above 30% (verify robustness and assumptions).")
+    else:
+        print("\nCAGR is within target 20–30% range.")
 
 
 if __name__ == "__main__":
-    run()
+    main()
