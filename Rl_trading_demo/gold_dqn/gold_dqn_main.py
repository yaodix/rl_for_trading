"""
Gold ETF (518880.SH) DQN Training & Backtesting Pipeline

Usage:
    python gold_project/gold_dqn_main.py --mode all      # Train + Validate + Test
    python gold_project/gold_dqn_main.py --mode train     # Train only
    python gold_project/gold_dqn_main.py --mode backtest  # Backtest only (requires trained model)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gold_project.features import get_feat_split
from gold_project.gold_dqn_env import GoldTradingEnv

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "518880.SH.30m.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TB_LOG_DIR = os.path.join(os.path.dirname(__file__), "tensorboard_log")

# ── Constants ──────────────────────────────────────────────────────────
BARS_PER_DAY = 8
TRADING_DAYS_PER_YEAR = 242
ANNUAL_FACTOR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR  # 1936 bars/year


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load and split data using features.py pipeline."""
    df_train, df_val, df_test = get_feat_split(
        DATA_PATH, split_ratio=[0.75, 0.9]
    )
    # Keep date info as a column before resetting index
    for df in (df_train, df_val, df_test):
        df.reset_index(inplace=True)
        if "trade_time" in df.columns:
            df.rename(columns={"trade_time": "date"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)
    return df_train, df_val, df_test


def _make_env(df: pd.DataFrame, **kwargs) -> GoldTradingEnv:
    """Create a GoldTradingEnv from a dataframe."""
    return GoldTradingEnv(df=df, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_dqn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    total_timesteps: int = 100_000,
) -> str:
    """Train DQN and return path to best model."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # Training env
    train_env = DummyVecEnv([lambda: _make_env(train_df, print_verbosity=10)])

    # Validation env for EvalCallback
    val_env = DummyVecEnv([lambda: _make_env(val_df)])

    best_model_path = os.path.join(MODEL_DIR, "dqn_gold_etf_best")
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=best_model_path,
        log_path=os.path.join(MODEL_DIR, "eval_logs"),
        eval_freq=20_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50_000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        learning_starts=1_000,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 128]),
        tensorboard_log=TB_LOG_DIR,
        verbose=1,
        seed=42,
    )

    print("=" * 60)
    print("Starting DQN Training")
    print(f"  Train bars : {len(train_df)}")
    print(f"  Val bars   : {len(val_df)}")
    print(f"  Timesteps  : {total_timesteps}")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="dqn_gold",
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, "dqn_gold_etf_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    best_zip = os.path.join(best_model_path, "best_model.zip")
    if os.path.exists(best_zip):
        print(f"Best model saved to {best_zip}")
        return best_zip
    return final_path


# ═══════════════════════════════════════════════════════════════════════
# Evaluation / Backtesting
# ═══════════════════════════════════════════════════════════════════════

def evaluate(
    df: pd.DataFrame,
    model_path: str,
    label: str = "test",
) -> dict:
    """Run model on a dataset and produce metrics + plots.

    Returns dict of performance metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = DQN.load(model_path)
    env = _make_env(df)
    obs, _ = env.reset()

    for _ in range(len(env.df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        if done:
            break

    # ── Collect results ──
    df_account = env.save_asset_memory()
    df_actions = env.save_action_memory()

    # ── Metrics ──
    values = df_account["account_value"].values
    returns = pd.Series(values).pct_change().dropna()

    total_return = (values[-1] - values[0]) / values[0]

    n_bars = len(values) - 1
    annual_return = (1 + total_return) ** (ANNUAL_FACTOR / max(n_bars, 1)) - 1

    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = drawdown.max()

    # Sharpe ratio (annualized)
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(ANNUAL_FACTOR)
    else:
        sharpe = 0.0

    # Win rate (bars with positive return)
    win_rate = (returns > 0).sum() / max(len(returns), 1)

    # Trade count
    trade_count = env.trade_count

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "final_value": values[-1],
        "n_bars": n_bars,
    }

    # ── Print ──
    print(f"\n{'=' * 50}")
    print(f"  [{label.upper()}] Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Total Return   : {total_return * 100:+.2f}%")
    print(f"  Annual Return  : {annual_return * 100:+.2f}%")
    print(f"  Max Drawdown   : {max_drawdown * 100:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Win Rate       : {win_rate * 100:.1f}%")
    print(f"  Trade Count    : {trade_count}")
    print(f"  Final Value    : {values[-1]:.2f}")
    print(f"{'=' * 50}\n")

    # ── Save CSV ──
    df_account.to_csv(os.path.join(RESULTS_DIR, f"{label}_account_value.csv"), index=False)
    df_actions.to_csv(os.path.join(RESULTS_DIR, f"{label}_actions.csv"), index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f"{label}_metrics.csv"), index=False)

    # ── Plot: Account Value ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(values, label="DQN Strategy", linewidth=1.2)
    ax.set_title(f"[{label.upper()}] Account Value")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Account Value (RMB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{label}_account_value.png"), dpi=150)
    plt.close(fig)

    # ── Plot: DQN vs Buy-and-Hold ──
    first_close = df.iloc[0]["Close"]
    buy_hold_shares = int(env.initial_amount // (first_close * env.lot_size)) * env.lot_size
    buy_hold_cost = buy_hold_shares * first_close + max(buy_hold_shares * first_close * env.commission_rate, env.min_commission)
    buy_hold_cash = env.initial_amount - buy_hold_cost
    close_prices = df["Close"].values[: len(values)]
    buy_hold_values = buy_hold_cash + buy_hold_shares * close_prices

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(values, label="DQN Strategy", linewidth=1.2)
    ax.plot(buy_hold_values, label="Buy & Hold", linewidth=1.2, linestyle="--")
    ax.set_title(f"[{label.upper()}] DQN vs Buy & Hold")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Account Value (RMB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{label}_vs_buyhold.png"), dpi=150)
    plt.close(fig)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gold ETF DQN Trading System")
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "all"],
        default="all",
        help="train: train only | backtest: evaluate only | all: train + evaluate",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for backtest mode (default: auto-detect best model)",
    )
    args = parser.parse_args()

    # ── Load data ──
    print("Loading and preprocessing data...")
    df_train, df_val, df_test = load_data()
    print(f"  Train: {len(df_train)} bars")
    print(f"  Val  : {len(df_val)} bars")
    print(f"  Test : {len(df_test)} bars")

    if args.mode in ("train", "all"):
        best_model_path = train_dqn(df_train, df_val, total_timesteps=args.timesteps)
    else:
        best_model_path = args.model_path
        if best_model_path is None:
            best_zip = os.path.join(MODEL_DIR, "dqn_gold_etf_best", "best_model.zip")
            final_zip = os.path.join(MODEL_DIR, "dqn_gold_etf_final.zip")
            if os.path.exists(best_zip):
                best_model_path = best_zip
            elif os.path.exists(final_zip):
                best_model_path = final_zip
            else:
                print("ERROR: No trained model found. Run with --mode train first.")
                sys.exit(1)

    if args.mode in ("backtest", "all"):
        val_metrics = evaluate(df_val, best_model_path, label="val")
        test_metrics = evaluate(df_test, best_model_path, label="test")

        # ── Comparison table ──
        print("\n" + "=" * 60)
        print("  Validation vs Test Comparison")
        print("=" * 60)
        header = f"{'Metric':<20} {'Validation':>15} {'Test':>15}"
        print(header)
        print("-" * 50)
        for key in ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "win_rate", "trade_count"]:
            v = val_metrics[key]
            t = test_metrics[key]
            if key in ("total_return", "annual_return", "max_drawdown", "win_rate"):
                print(f"  {key:<18} {v * 100:>14.2f}% {t * 100:>14.2f}%")
            elif key == "trade_count":
                print(f"  {key:<18} {v:>15d} {t:>15d}")
            else:
                print(f"  {key:<18} {v:>15.3f} {t:>15.3f}")
        print("=" * 60)
        print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
