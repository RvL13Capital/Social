import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
from itertools import product

# --- 1. Logik-Import aus v5 ---
# Wir kopieren die Indikatoren-Logik aus unserem v5-Skript,
# damit dieser Backtester eigenständig lauffähig ist.

def calculate_market_indicators(df):
    """
    Berechnet alle von der Epidemiologie-Logik geforderten Marktindikatoren.
    (Kopiert von psych_ai_v5.py)
    """
    print("Berechne Markt-Indikatoren für Backtest...")
    df = df.sort_values('Date').reset_index(drop=True)
    
    # MAs
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    # Volumen MA
    df['Vol_MA50'] = df['Volume'].rolling(50).mean()
    df['Vol_MA50_Low_252d'] = df['Vol_MA50'].rolling(252).min()
    
    # Bollinger Bands (Stage 0)
    bb_mean = df['MA20']
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_mean + (bb_std * 2)
    df['BB_Lower'] = bb_mean - (bb_std * 2)
    df['BB_Width_Percent'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    df['BB_Width_Percent_Low_252d'] = df['BB_Width_Percent'].rolling(252).min()
    
    # ATR (Stage 0 & 4)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_20'] = tr.rolling(20).mean()
    df['ATR_Percent'] = (df['ATR_20'] / df['Close']) * 100
    df['ATR_Percent_Low_126d'] = df['ATR_Percent'].rolling(126).min() # 6-Monats-Tief
    
    # Tägliche Range (Stage 4)
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_vs_ATR'] = df.apply(
        lambda row: row['Daily_Range'] / row['ATR_20'] if row['ATR_20'] > 0 else 0, axis=1
    )
    
    # OBV (Stage 1)
    delta = df['Close'].diff()
    df['OBV'] = (np.sign(delta) * df['Volume']).cumsum()
    
    # RSI (Stage 3)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain.div(avg_loss).replace(np.inf, 0)
    df['RSI'] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, 'RSI'] = 100
    
    return df

# --- 2. VBT Signal-Generator ---

@vbt.IndicatorFactory(
    # Definiere, welche Daten-Spalten wir benötigen
    input_names=[
        "close", "open", "low", "high", "volume", 
        "ma20", "ma50", "ma200", 
        "vol_ma50", 
        "rsi", 
        "obv",
        "daily_range", "daily_range_vs_atr"
    ],
    # Definiere, welche Parameter wir optimieren wollen
    param_names=[
        "ignition_price_factor", "ignition_vol_factor",
        "churn_vol_factor", "churn_body_range_factor"
    ],
    # Definiere, was wir ausgeben (Entry- und Exit-Signale)
    output_names=["entries", "exits"]
)
def create_epidemiology_signals(
    close, open, low, high, volume, 
    ma20, ma50, ma200, 
    vol_ma50, 
    rsi, 
    obv,
    daily_range, daily_range_vs_atr,
    # Parameter werden automatisch von vbt übergeben
    ignition_price_factor, ignition_vol_factor,
    churn_vol_factor, churn_body_range_factor,
    **kwargs # Erfasse alle anderen Parameter
):
    """
    Vectorbt-kompatible Funktion, die unsere Stage-Logik in
    Entry- und Exit-Signale (True/False) umwandelt.
    """
    
    # --- Stage 1: Infection (Entry-Signal) ---
    is_ignition_spike = (
        (close > (ma50 * ignition_price_factor)) &
        (volume > (vol_ma50 * ignition_vol_factor))
    )
    # HINWEIS: OBV-Divergenz ist komplexer (path-dependent) und wird
    # hier zur Vereinfachung der Vektorisierung weggelassen.
    # Wir konzentrieren uns auf den "Ignition Spike" als primären Trigger.
    entries = is_ignition_spike
    
    # --- Stage 3: Saturation (Exit-Signal) ---
    is_churn_day = (
        (volume > (vol_ma50 * churn_vol_factor)) &
        (daily_range > 0) &
        ((abs(close - open) / daily_range) < churn_body_range_factor)
    )
    
    # RSI-Divergenz (einfachere Vektorisierung): Preis auf neuem 30-Tage-Hoch, RSI nicht
    rsi_30d_high = rsi.rolling(30).max()
    close_30d_high = close.rolling(30).max()
    is_rsi_divergence = (
        (close == close_30d_high) & # Preis *ist* das Hoch
        (rsi < rsi_30d_high)       # RSI *ist nicht* das Hoch
    )
    
    # Struktur-Bruch: Kreuzt unter MA50
    is_structure_break = (close.shift(1) > ma50.shift(1)) & (close < ma50)
    
    exits_saturation = is_churn_day | is_rsi_divergence | is_structure_break
    
    # --- Stage 4: Collapse (Exit-Signal) ---
    is_death_cross = (ma20.shift(1) > ma50.shift(1)) & (ma20 < ma50)
    is_collapse = is_death_cross & (close < ma200)
    
    exits = exits_saturation | is_collapse
    
    return entries, exits

# --- 3. Haupt-Backtesting-Funktion ---

def run_backtest(ticker="NVDA", start_date="2018-01-01"):
    print(f"--- Backtesting (Schritt 2) für {ticker} ---")
    
    # 1. Daten laden (lange Historie)
    print(f"Lade 5+ Jahre Historie für {ticker}...")
    data = yf.download(ticker, start=start_date)
    if data.empty:
        print("Fehler: Keine Daten geladen.")
        return
        
    # 2. Alle Indikatoren vorberechnen
    # Wir müssen 'Date' als Index für vectorbt setzen
    data_indexed = data.set_index(pd.to_datetime(data.index))
    market_df = calculate_market_indicators(data_indexed)
    
    # 3. Parameter-Gitter definieren (Werte, die wir testen wollen)
    print("Definiere Parameter-Gitter für Optimierung...")
    param_grid = {
        'ignition_price_factor': np.arange(1.01, 1.06, 0.01), # [1.01, 1.02, 1.03, 1.04, 1.05]
        'ignition_vol_factor': np.arange(2.0, 3.1, 0.5),     # [2.0, 2.5, 3.0]
        'churn_vol_factor': np.arange(2.5, 4.1, 0.5),        # [2.5, 3.0, 3.5, 4.0]
        'churn_body_range_factor': np.arange(0.1, 0.4, 0.1)  # [0.1, 0.2, 0.3]
    }
    
    # Berechne die Anzahl der Kombinationen
    combinations = list(product(*param_grid.values()))
    print(f"Teste {len(combinations)} Parameter-Kombinationen...")

    # 4. Indikator-Factory initialisieren
    ind = create_epidemiology_signals.from_params(
        # Übergebe die vor-berechneten Indikatoren
        close=market_df['Close'],
        open=market_df['Open'],
        low=market_df['Low'],
        high=market_df['High'],
        volume=market_df['Volume'],
        ma20=market_df['MA20'],
        ma50=market_df['MA50'],
        ma200=market_df['MA200'],
        vol_ma50=market_df['Vol_MA50'],
        rsi=market_df['RSI'],
        obv=market_df['OBV'],
        daily_range=market_df['Daily_Range'],
        daily_range_vs_atr=market_df['Daily_Range_vs_ATR'],
        # Übergebe das Parameter-Gitter
        **param_grid,
        # Setze `broadcast_named_params=True`, damit vbt alle Kombinationen testet
        broadcast_named_params=True 
    )

    # 5. Portfolio-Simulation für alle Kombinationen durchführen
    print("Führe Portfolio-Simulationen für alle Kombinationen durch...")
    pf = vbt.Portfolio.from_signals(
        market_df['Close'], 
        ind.entries, 
        ind.exits,
        freq='1D', # Tägliche Frequenz
        init_cash=100000, # Startkapital
        fees=0.001,       # 0.1% Gebühren pro Trade
        slippage=0.001    # 0.1% Slippage pro Trade
    )
    
    # 6. Ergebnisse analysieren
    print("Analyse der Ergebnisse...")
    
    # Hole Metriken für alle Simulationen
    stats_df = pf.stats(metrics=['Total Return [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Total Trades'])
    
    # Finde die Simulation mit der höchsten Gesamtrendite
    best_params_idx = pf.total_return.idxmax()
    best_params = stats_df.loc[best_params_idx].name
    best_stats = stats_df.loc[best_params_idx]
    
    print("\n--- Backtesting-Ergebnisse ---")
    print(f"Asset: {ticker} ({start_date} bis heute)")
    print(f"Anzahl Simulationen: {len(combinations)}")
    
    print("\nBeste Strategie (nach Total Return):")
    print(best_stats)
    
    print("\n--- OPTIMALE PARAMETER (für CONFIG in v5) ---")
    # `best_params` ist ein Tuple, wir müssen es lesbar machen
    optimal_config = {
        'ignition_price_factor': best_params[0],
        'ignition_vol_factor': best_params[1],
        'churn_vol_factor': best_params[2],
        'churn_body_range_factor': best_params[3]
    }
    print(optimal_config)
    print("\nKopiere diese Werte in die `CONFIG`-Sektion von `psych_ai_v5_Live.py`.")
    
    # Optional: Plot der besten Strategie
    # pf[best_params_idx].plot().show()

if __name__ == "__main__":
    # Stelle sicher, dass du 'pip install vectorbt yfinance' ausgeführt hast
    run_backtest(ticker="NVDA", start_date="2018-01-01")

