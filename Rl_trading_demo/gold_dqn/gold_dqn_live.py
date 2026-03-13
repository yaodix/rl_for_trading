"""
Gold ETF (518880.SH) DQN Live Trading Interface

Provides a LiveTrader class that:
  1. Maintains a historical K-line buffer for feature calculation
  2. Loads a trained DQN model
  3. Predicts buy/hold/sell signals per 30-min bar
  4. Reserves interfaces for real broker execution

Usage:
    trader = LiveTrader("gold_project/trained_models/dqn_gold_etf_best/best_model.zip")
    trader.init_history(pd.read_csv("recent_klines.csv"))  # >= 202 bars
    signal = trader.run_once(new_kline_row, cash=8000, shares=900)
    # signal = {"action": "buy", "price": 10.5, "shares": 900, ...}
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gold_project.features import calculate_gold_etf_features, smart_feature_normalization

# Minimum history length: 176 (norm window) + 26 (longest indicator: EMA26) = 202
MIN_HISTORY_BARS = 202
ACTION_MAP = {0: "sell", 1: "hold", 2: "buy"}


class LiveTrader:
    """Live trading wrapper for the trained DQN model."""

    def __init__(
        self,
        model_path: str,
        initial_amount: float = 10000,
        commission_rate: float = 0.00005,
        min_commission: float = 5.0,
        lot_size: int = 100,
    ):
        self.model = DQN.load(model_path)
        self.initial_amount = initial_amount
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.lot_size = lot_size

        # History buffer — raw OHLCV with Title-case columns
        self._history: pd.DataFrame | None = None
        # Latest normalized features (single row)
        self._latest_features: np.ndarray | None = None
        self._feature_columns: list[str] = []

    # ── History management ─────────────────────────────────────────────

    def init_history(self, df: pd.DataFrame):
        """Initialize history buffer with past K-line data.

        Args:
            df: DataFrame with columns trade_time, open, high, low, close, volume.
                Must have at least MIN_HISTORY_BARS rows.
        """
        df = df.copy()
        if "trade_time" in df.columns:
            df.set_index("trade_time", inplace=True)
        if "amount" in df.columns:
            df.drop(columns=["amount"], inplace=True)
        df.columns = df.columns.str.title()

        if len(df) < MIN_HISTORY_BARS:
            raise ValueError(
                f"Need at least {MIN_HISTORY_BARS} bars, got {len(df)}"
            )

        self._history = df
        self._recompute_features()

    def update_kline(self, new_row: dict | pd.Series):
        """Append a new K-line bar and recompute features.

        Args:
            new_row: dict/Series with keys trade_time, open, high, low, close, volume.
        """
        if self._history is None:
            raise RuntimeError("Call init_history() first")

        row = pd.Series(new_row) if isinstance(new_row, dict) else new_row.copy()
        if "amount" in row.index:
            row = row.drop("amount")

        # Ensure Title-case
        row.index = row.index.str.title()

        idx = row.pop("Trade_Time") if "Trade_Time" in row.index else None
        new_df = pd.DataFrame([row], index=[idx] if idx else None)
        new_df.columns = new_df.columns.str.title()
        self._history = pd.concat([self._history, new_df])

        # Trim to keep buffer manageable (keep last 500 bars)
        if len(self._history) > 500:
            self._history = self._history.iloc[-500:]

        self._recompute_features()

    def _recompute_features(self):
        """Recompute features from the full history buffer."""
        feat_df = calculate_gold_etf_features(self._history)
        norm_df = smart_feature_normalization(feat_df)
        if len(norm_df) == 0:
            self._latest_features = None
            return
        self._feature_columns = [c for c in norm_df.columns if c.endswith("_Norm")]
        self._latest_features = norm_df.iloc[-1][self._feature_columns].values.astype(np.float32)

    # ── State construction ─────────────────────────────────────────────

    def get_state(self, cash: float, shares: int) -> np.ndarray:
        """Build observation vector from account state + latest features."""
        if self._latest_features is None:
            raise RuntimeError("Features not ready — call init_history() or update_kline()")

        price = float(self._history.iloc[-1]["Close"])
        cash_ratio = cash / self.initial_amount
        position_ratio = (shares * price) / self.initial_amount
        state = np.concatenate(
            [[cash_ratio, position_ratio], self._latest_features]
        ).astype(np.float32)
        return state

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_action(self, state: np.ndarray) -> str:
        """Return 'buy', 'hold', or 'sell'."""
        action, _ = self.model.predict(state, deterministic=True)
        return ACTION_MAP[int(action)]

    def run_once(
        self, latest_kline: dict | pd.Series, cash: float, shares: int
    ) -> dict:
        """Full prediction cycle: update data → build state → predict.

        Returns:
            dict with keys: action, price, suggested_shares, timestamp
        """
        self.update_kline(latest_kline)
        state = self.get_state(cash, shares)
        action = self.predict_action(state)
        price = float(self._history.iloc[-1]["Close"])

        suggested_shares = 0
        if action == "buy":
            commission = self.min_commission
            max_lots = int((cash - commission) // (self.lot_size * price))
            suggested_shares = max(max_lots, 0) * self.lot_size
        elif action == "sell":
            suggested_shares = shares

        result = {
            "action": action,
            "price": price,
            "suggested_shares": suggested_shares,
            "timestamp": str(self._history.index[-1]),
            "cash": cash,
            "current_shares": shares,
        }
        return result

    # ── Broker interfaces (to be implemented) ──────────────────────────

    def fetch_latest_data(self) -> dict:
        """Fetch latest 30-min K-line from data source.

        TODO: Implement with your broker/data API.
        Returns dict with keys: trade_time, open, high, low, close, volume.
        """
        raise NotImplementedError(
            "Implement this method with your broker/data API "
            "(e.g., QMT, easytrader, tushare, etc.)"
        )

    def execute_trade(self, action: str, price: float, shares: int) -> dict:
        """Execute a trade order via broker API.

        TODO: Implement with your broker API.

        Args:
            action: 'buy' or 'sell'
            price: target price
            shares: number of shares

        Returns:
            dict with order result from broker.
        """
        raise NotImplementedError(
            "Implement this method with your broker API "
            "(e.g., QMT, easytrader, etc.)"
        )

    def run_live_loop(self):
        """Main live trading loop skeleton.

        This is a template — implement fetch_latest_data() and execute_trade()
        before using.
        """
        print("Live trading loop started (skeleton mode)")
        print("Implement fetch_latest_data() and execute_trade() for real trading")

        # Example loop structure:
        # while market_is_open():
        #     kline = self.fetch_latest_data()
        #     cash, shares = self.get_account_status()  # from broker
        #     signal = self.run_once(kline, cash, shares)
        #     print(f"[{signal['timestamp']}] Signal: {signal['action']} "
        #           f"@ {signal['price']:.3f}, shares={signal['suggested_shares']}")
        #     if signal['action'] != 'hold':
        #         result = self.execute_trade(
        #             signal['action'], signal['price'], signal['suggested_shares']
        #         )
        #         print(f"  Order result: {result}")
        #     time.sleep(30 * 60)  # wait for next 30-min bar
