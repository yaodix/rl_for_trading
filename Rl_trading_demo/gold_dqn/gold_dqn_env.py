"""
Gold ETF (518880.SH) DQN Trading Environment

Custom gymnasium environment with Discrete(3) action space for DQN.
Actions: 0=Sell all, 1=Hold, 2=Buy with all available cash.
Supports T+0 trading (gold ETF feature).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class GoldTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 10000,
        commission_rate: float = 0.00005,
        min_commission: float = 5.0,
        lot_size: int = 100,
        reward_scaling: float = 1.0,
        print_verbosity: int = 0,
    ):
        """
        Args:
            df: DataFrame with columns Open, Close, and *_Norm feature columns.
                Index should be integer-based (reset_index after time split).
            initial_amount: Starting cash in RMB.
            commission_rate: Commission rate per trade (0.00005 = 万0.5).
            min_commission: Minimum commission per trade in RMB.
            lot_size: Minimum trading unit (100 shares for A-share ETF).
            reward_scaling: Multiplier on reward signal.
            print_verbosity: Print stats every N episodes (0=silent).
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.lot_size = lot_size
        self.reward_scaling = reward_scaling
        self.print_verbosity = print_verbosity

        # Extract feature columns (all columns ending with _Norm)
        self.feature_columns = [c for c in self.df.columns if c.endswith("_Norm")]
        self.n_features = len(self.feature_columns)

        # State: [cash_ratio, position_ratio, n_features...]
        self.state_dim = 2 + self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Actions: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)

        # Will be set in reset()
        self.day = 0
        self.cash = initial_amount
        self.shares = 0
        self.total_asset = initial_amount
        self.episode = 0

        # Memory for analysis
        self.asset_memory = []
        self.actions_memory = []
        self.date_memory = []
        self.rewards_memory = []
        self.trade_count = 0

    def _get_price(self) -> float:
        return float(self.df.loc[self.day, "Close"])

    def _get_features(self) -> np.ndarray:
        return self.df.loc[self.day, self.feature_columns].values.astype(np.float32)

    def _get_date(self) -> str:
        if "date" in self.df.columns:
            return str(self.df.loc[self.day, "date"])
        return str(self.df.index[self.day])

    def _calc_commission(self, trade_amount: float) -> float:
        """Commission = max(trade_amount * rate, min_commission)."""
        return max(trade_amount * self.commission_rate, self.min_commission)

    def _get_state(self) -> np.ndarray:
        price = self._get_price()
        position_value = self.shares * price
        total = self.cash + position_value
        cash_ratio = self.cash / self.initial_amount
        position_ratio = position_value / self.initial_amount
        features = self._get_features()
        state = np.concatenate(
            [[cash_ratio, position_ratio], features]
        ).astype(np.float32)
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_amount
        self.shares = 0
        self.total_asset = self.initial_amount
        self.trade_count = 0

        self.asset_memory = [self.initial_amount]
        self.actions_memory = []
        self.rewards_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1
        return self._get_state(), {}

    def step(self, action: int):
        price = self._get_price()
        prev_total = self.cash + self.shares * price

        # Execute action
        action_label = "hold"
        if action == 0 and self.shares > 0:
            # Sell all
            trade_amount = self.shares * price
            commission = self._calc_commission(trade_amount)
            self.cash += trade_amount - commission
            self.shares = 0
            self.trade_count += 1
            action_label = "sell"
        elif action == 2 and self.cash > 0:
            # Buy with all available cash
            # Need: lots * lot_size * price + commission >= min_commission
            # Iteratively find max lots we can afford
            max_lots = int(self.cash // (self.lot_size * price))
            if max_lots > 0:
                # Check if we can afford with commission
                while max_lots > 0:
                    buy_amount = max_lots * self.lot_size * price
                    commission = self._calc_commission(buy_amount)
                    if buy_amount + commission <= self.cash:
                        break
                    max_lots -= 1
                if max_lots > 0:
                    buy_shares = max_lots * self.lot_size
                    buy_amount = buy_shares * price
                    commission = self._calc_commission(buy_amount)
                    self.cash -= buy_amount + commission
                    self.shares += buy_shares
                    self.trade_count += 1
                    action_label = "buy"

        # Advance to next day
        self.day += 1
        terminal = self.day >= len(self.df) - 1

        if not terminal:
            new_price = self._get_price()
        else:
            new_price = price

        new_total = self.cash + self.shares * new_price
        reward = (new_total - prev_total) / self.initial_amount * self.reward_scaling
        self.total_asset = new_total

        # Record
        self.asset_memory.append(new_total)
        self.actions_memory.append(action_label)
        self.rewards_memory.append(reward)
        self.date_memory.append(self._get_date() if not terminal else self.date_memory[-1])

        if terminal and self.print_verbosity > 0 and self.episode % self.print_verbosity == 0:
            total_return = (new_total - self.initial_amount) / self.initial_amount * 100
            print(
                f"Episode {self.episode} | "
                f"Final: {new_total:.2f} | "
                f"Return: {total_return:.2f}% | "
                f"Trades: {self.trade_count}"
            )

        state = self._get_state() if not terminal else np.zeros(self.state_dim, dtype=np.float32)
        return state, reward, terminal, False, {}

    def save_asset_memory(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"date": self.date_memory, "account_value": self.asset_memory}
        )

    def save_action_memory(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"date": self.date_memory[1:], "action": self.actions_memory}
        )

    def get_sb_env(self):
        """Return a DummyVecEnv-wrapped version for SB3 compatibility."""
        env = self
        def _make_env():
            return env
        vec_env = DummyVecEnv([_make_env])
        return vec_env, vec_env.reset()
