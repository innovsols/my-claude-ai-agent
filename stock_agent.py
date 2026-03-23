import asyncio
import os
import json
import yfinance as yf
import pandas as pd
import ta
from dotenv import load_dotenv
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ClaudeAgentOptions, TextBlock

load_dotenv()

# ── Custom Tools ──────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "6mo") -> str:
    """Fetch OHLCV data and compute RSI + 9 EMA of RSI for crossover detection."""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return json.dumps({"error": f"No data found for {ticker}"})

        # ── Fix: flatten MultiIndex columns from newer yfinance versions ──
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Now safely extract Close as a clean 1D Series
        close = df["Close"].squeeze()

        df = pd.DataFrame({"Close": close})
        df.dropna(inplace=True)

        # 14-period RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        # 9 EMA applied ON RSI
        df["RSI_EMA9"] = df["RSI"].ewm(span=9, adjust=False).mean()

        df.dropna(inplace=True)

        # Detect crossover using last 2 rows
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        crossed_up   = (prev["RSI"] <= prev["RSI_EMA9"]) and (curr["RSI"] > curr["RSI_EMA9"])
        crossed_down = (prev["RSI"] >= prev["RSI_EMA9"]) and (curr["RSI"] < curr["RSI_EMA9"])

        signal = interpret_signal(
            rsi=float(curr["RSI"]),
            rsi_ema9=float(curr["RSI_EMA9"]),
            crossed_up=crossed_up,
            crossed_down=crossed_down,
        )

        recent = df[["Close", "RSI", "RSI_EMA9"]].tail(10).round(2)

        result = {
            "ticker": ticker,
            "latest_close": float(curr["Close"]),
            "rsi_14": round(float(curr["RSI"]), 2),
            "rsi_ema9": round(float(curr["RSI_EMA9"]), 2),
            "crossed_up": crossed_up,
            "crossed_down": crossed_down,
            "signal": signal,
            "recent_data": recent.reset_index().to_dict(orient="records"),
        }
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def interpret_signal(rsi: float, rsi_ema9: float, crossed_up: bool, crossed_down: bool) -> str:
    """
    Signal logic:
      - RSI crosses ABOVE its 9 EMA  → Bullish momentum (buy signal)
      - RSI crosses BELOW its 9 EMA  → Bearish momentum (sell signal)
      - No crossover today            → RSI position relative to EMA (continuation)
    """
    signals = []

    # Crossover signal (strongest)
    if crossed_up:
        signals.append("🟢 RSI crossed ABOVE 9 EMA → BULLISH crossover (fresh buy signal today)")
    elif crossed_down:
        signals.append("🔴 RSI crossed BELOW 9 EMA → BEARISH crossover (fresh sell signal today)")
    else:
        # No crossover — show continuation bias
        if rsi > rsi_ema9:
            signals.append(f"RSI ({rsi:.1f}) is ABOVE its 9 EMA ({rsi_ema9:.1f}) → Bullish continuation (no fresh crossover today)")
        else:
            signals.append(f"RSI ({rsi:.1f}) is BELOW its 9 EMA ({rsi_ema9:.1f}) → Bearish continuation (no fresh crossover today)")

    # Overbought / Oversold context
    if rsi > 70:
        signals.append("⚠️  RSI > 70 → Overbought territory — watch for exhaustion")
    elif rsi < 30:
        signals.append("⚠️  RSI < 30 → Oversold territory — watch for reversal")

    return " | ".join(signals)


# ── Agent ─────────────────────────────────────────────────────────────────────

async def run_stock_agent(ticker: str):
    print(f"\n🤖 Stock Analysis Agent starting for: {ticker}\n{'─'*50}")

    options = ClaudeAgentOptions(
        system_prompt="""You are an expert stock market technical analyst. 
        You use RSI crossover with its 9 EMA as your primary signal on the daily chart:
        - RSI crossing ABOVE its 9 EMA = Bullish momentum → potential buy
        - RSI crossing BELOW its 9 EMA = Bearish momentum → potential sell
        - RSI staying above/below EMA without a fresh cross = trend continuation

        When analyzing, clearly state:
        1. Whether a fresh crossover happened today or if it's a continuation
        2. RSI level and overbought/oversold context
        3. Overall verdict: Bullish / Bearish / Neutral with reasoning
        4. Key things to watch in coming sessions
        Be concise and actionable.""",
        max_turns=5,
    )

    raw_data = fetch_stock_data(ticker)
    data = json.loads(raw_data)

    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return

    prompt = f"""
Analyze this stock using RSI vs 9 EMA of RSI crossover logic on the daily chart:

Ticker       : {data['ticker']}
Latest Close : ₹{data['latest_close']}
RSI (14)     : {data['rsi_14']}
RSI 9 EMA    : {data['rsi_ema9']}
Fresh Crossover Today → Up: {data['crossed_up']} | Down: {data['crossed_down']}
Signal       : {data['signal']}

Recent 10-day RSI & EMA data:
{json.dumps(data['recent_data'], indent=2, default=str)}

Provide a structured technical analysis and trading recommendation.
"""

    print(f"📊 Technical Data:\n"
          f"   Close   : ₹{data['latest_close']}\n"
          f"   RSI 14  : {data['rsi_14']}\n"
          f"   RSI EMA9: {data['rsi_ema9']}\n"
          f"   Signal  : {data['signal']}\n")

    print("🧠 Claude's Analysis:\n" + "─"*50)

    # ── Fixed: use async with + query() + receive_response() ──
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = "HDFCBANK.NS"
    asyncio.run(run_stock_agent(ticker))
""" ```

---

## What Changed & Why

| Area | Before | After |
|---|---|---|
| EMA applied to | Price (Close) | **RSI values** |
| Signal trigger | Price > or < EMA | **RSI crosses above/below its own 9 EMA** |
| Crossover detection | None | **Compares last 2 rows** to catch the exact cross day |
| Signal types | Bullish / Bearish / Neutral | Fresh cross 🟢🔴 **or** continuation bias |
| Claude system prompt | Generic TA | Specifically tuned to **RSI-EMA crossover logic** |

---

## The Logic Explained
```
Daily RSI values:  42 → 45 → 47 → 51 ...
9 EMA of RSI:      46 → 46 → 48 → 49 ...

Day N-1:  RSI (47) <= EMA (48)  ← RSI was below
Day N:    RSI (51) >  EMA (49)  ← RSI is now above
                ↑
         BULLISH CROSSOVER detected ✅ """