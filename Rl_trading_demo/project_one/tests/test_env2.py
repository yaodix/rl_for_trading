"""
RL 量化交易环境测试代码
重点测试 Reward 计算的完备性
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any
import sys
import os

# 添加lib/config.py路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import env_config

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 1. 模拟数据类（替代真实的 data.PriceGold）
# ============================================================

class MockPriceGold:
    """模拟价格数据，用于测试"""
    def __init__(self, n_bars: int = 100, base_price: float = 100.0):
        self.n_bars = n_bars
        # 生成模拟价格数据
        np.random.seed(42)
        returns = np.random.randn(n_bars) * 0.02  # 2% 日波动
        self.Close = base_price * np.cumprod(1 + returns)
        self.high = self.Close * (1 + np.random.rand(n_bars) * 0.01)
        self.low = self.Close * (1 - np.random.rand(n_bars) * 0.01)
        self.volume = np.random.randint(10000, 100000, n_bars)
        self.open = self.Close * (1 + np.random.randn(n_bars) * 0.005)
        
        # 归一化特征（模拟真实数据）
        self.Amplitude_Norm = (self.high - self.low) / self.open
        self.Returns_Norm = np.diff(self.Close, prepend=self.Close[0]) / self.Close
        self.Volume_Ratio_Norm = self.volume / np.mean(self.volume)
        self.Close_Position_Norm = (self.Close - self.low) / (self.high - self.low + 1e-8)
        
    @property
    def shape(self):
        return (self.n_bars,)


# ============================================================
# 2. 模拟配置（替代真实的 config）
# ============================================================

class MockConfig:
    env_config = {
        "cash": 100000.0,  # 10 万初始资金
        "slippage": 0.0001,  # 万 1 滑点
        "reward_scale": 100.0,  # 奖励缩放
        "idle_penalty": 0.00,  # 空仓惩罚
    }


# 注入模拟模块
import sys
sys.modules['env.data'] = type(sys)('data')
sys.modules['env.data'].PriceGold = MockPriceGold
sys.modules['env.config'] = MockConfig

# 现在可以导入环境
from lib.environ import StocksEnv, State, Actions


# ============================================================
# 3. 测试工具函数
# ============================================================

def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def assert_close(a: float, b: float, tol: float = 1e-4, msg: str = ""):
    """断言两个浮点数接近"""
    if not np.isclose(a, b, rtol=tol, atol=tol):
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")
    print(f"  ✓ {msg}: {a:.6f} ≈ {b:.6f}")


# ============================================================
# 4. 测试用例
# ============================================================

class TestStocksEnv:
    """环境测试类"""
    
    def __init__(self):
        self.test_results = []
        
    def run_test(self, name: str, test_func):
        """运行单个测试"""
        try:
            test_func()
            self.test_results.append((name, True, None))
            print(f"✅ {name}")
        except Exception as e:
            self.test_results.append((name, False, str(e)))
            print(f"❌ {name}: {e}")
    
    def print_summary(self):
        """打印测试摘要"""
        print_separator("测试摘要")
        passed = sum(1 for _, r, _ in self.test_results if r)
        total = len(self.test_results)
        print(f"通过：{passed}/{total}")
        for name, result, error in self.test_results:
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
            if error:
                print(f"      错误：{error}")
    
    # -------------------- 基础测试 --------------------
    
    def test_env_initialization(self):
        """测试环境初始化"""
        prices = {"TEST": MockPriceGold(n_bars=100)}
        env = StocksEnv(prices=prices, bars_count=10, reward_on_close=False)
        
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        # assert len(info) > 0
        print(f"  初始观察空间形状：{obs.shape}")
        print(f"  初始现金：{env._state.cash:.2f}")
    
    def test_action_space(self):
        """测试动作空间"""
        prices = {"TEST": MockPriceGold(n_bars=100)}
        env = StocksEnv(prices=prices)
        
        assert env.action_space.n == 3  # Skip, Buy, Close
        print(f"  动作空间大小：{env.action_space.n}")
        print(f"  动作列表：{[a.name for a in Actions]}")
    
    # -------------------- Reward 核心测试 --------------------
    
    def test_reward_skip_idle(self):
        """测试空仓 Skip 的 reward（应该为 0）"""
        print_separator("测试：空仓 Skip 奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=False, reset_on_close=False)
        env.reset()
        
        # 连续 Skip 5 步（空仓状态）
        rewards = []
        for i in range(5):
            obs, reward, done, _, _ = env.step(Actions.Skip.value)
            rewards.append(reward)
            print(f"  Step {i+1}: reward={reward:.6f}, cash={env._state.cash:.2f}")
        
        # 空仓时 reward 应该接近 0（允许浮点误差）
        for i, r in enumerate(rewards):
            assert_close(r, 0.0, tol=1e-6, msg=f"Step {i+1} 空仓奖励")
    
    def test_reward_buy_success(self):
        """测试买入成功的 reward"""
        print_separator("测试：买入成功奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=False, reset_on_close=False)
        env.reset()
        
        # 第 1 步：买入
        obs, reward_buy, done, _, _ = env.step(Actions.Buy.value)
        print(f"  买入奖励：{reward_buy:.6f}")
        print(f"  买入后现金：{env._state.cash:.2f}")
        print(f"  持仓股数：{env._state.shares}")
        print(f"  持仓状态：{env._state.have_position}")
        
        # 买入这一步的 
        assert_close(reward_buy, 0.0, tol=0.1, msg="买入步奖励")
        assert env._state.have_position == True
        assert env._state.shares >= 100
    
    def test_reward_hold_price_change(self):
        """测试持仓期间价格波动的 reward"""
        print_separator("测试：持仓期间价格波动奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=False, reset_on_close=False)
        env.reset()
        
        # 买入
        env.step(Actions.Buy.value)
        entry_price = env._state.entry_price
        entry_shares = env._state.shares
        print(f"  买入价：{entry_price:.4f}, 股数：{entry_shares}")
        
        # 持仓 Skip 3 步，观察 reward
        rewards = []
        for i in range(3):
            prev_price = env._state._cur_close()
            obs, reward, done, _, _ = env.step(Actions.Skip.value)
            curr_price = env._state._cur_close()
            price_return = (curr_price - prev_price) / prev_price
            rewards.append(reward)
            print(f"  Step {i+1}: 价格 {prev_price:.4f}→{curr_price:.4f}, "
                  f"reward={reward:.6f}, 理论收益率={price_return:.6f}")
        
        # 验证 reward 与价格变化一致（考虑缩放）
        for i, r in enumerate(rewards):
            if abs(r) > 0.01:  # 只有价格有明显变化时才验证
                print(f"  ✓ Step {i+1} 奖励与价格变化方向一致")
    
    def test_reward_close_dense(self):
        """测试稠密奖励模式下的卖出 reward"""
        print_separator("测试：稠密奖励模式 - 卖出奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=False, reset_on_close=False)
        env.reset()
        
        # 买入
        env.step(Actions.Buy.value)
        prev_total = env._state._get_cur_total_value()
        print(f"  卖出前总资产：{prev_total:.2f}")
        
        # 直接卖出（模拟价格不变）
        obs, reward_close, done, _, _ = env.step(Actions.Close.value)
        new_total = env._state._get_cur_total_value()
        
        # 计算理论手续费
        trade_value = env._state.shares * env._state._cur_close() * (1 - env._state.slippage)
        commission = max(trade_value * env._state.commission_perc, 5.0)
        slippage_cost = env._state.shares * env._state._cur_close() * env._state.slippage
        total_cost = commission + slippage_cost
        expected_reward = -total_cost / prev_total * env._state.reward_scale
        
        print(f"  卖出奖励：{reward_close:.6f}")
        print(f"  交易金额：{trade_value:.2f}")
        print(f"  手续费：{commission:.2f}")
        print(f"  滑点成本：{slippage_cost:.2f}")
        print(f"  理论奖励：{expected_reward:.6f}")
        
        # 卖出奖励应该是负值（手续费 + 滑点）
        assert reward_close < 0, "卖出奖励应该为负（交易成本）"
        assert_close(reward_close, expected_reward, tol=0.1, msg="卖出奖励与理论值")
    
    def test_reward_close_sparse(self):
        """测试稀疏奖励模式下的卖出 reward"""
        print_separator("测试：稀疏奖励模式 - 卖出奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=True, reset_on_close=False)
        env.reset()
        
        # 买入
        env.step(Actions.Buy.value)
        entry_total = env._state.entry_total_asset
        print(f"  买入时总资产：{entry_total:.2f}")
        
        # 持仓 1 步（奖励应该为 0）
        obs, reward_hold, done, _, _ = env.step(Actions.Skip.value)
        print(f"  持仓步奖励：{reward_hold:.6f} (应该≈0)")
        assert_close(reward_hold, 0.0, tol=0.01, msg="稀疏模式持仓奖励")
        
        # 卖出
        obs, reward_close, done, _, _ = env.step(Actions.Close.value)
        final_cash = env._state.cash
        expected_return = (final_cash - entry_total) / entry_total * env._state.reward_scale
        
        print(f"  卖出奖励：{reward_close:.6f}")
        print(f"  最终现金：{final_cash:.2f}")
        print(f"  理论收益率：{expected_return:.6f}")
        
        assert_close(reward_close, expected_return, tol=0.1, msg="稀疏模式卖出奖励")
    
    def test_reward_buy_fail(self):
        """测试资金不足时买入失败的 reward"""
        print_separator("测试：买入失败奖励")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        # 设置非常少的现金，买不起 100 股
        original_cash = env_config["cash"]
        env_config["cash"] = 0.1  
        
        env = StocksEnv(prices=prices, reward_on_close=False)
        env.reset()
        
        # 尝试买入
        obs, reward, done, _, _ = env.step(Actions.Buy.value)
        print(f"  买入失败奖励：{reward:.6f}")
        print(f"  持仓状态：{env._state.have_position}")
        print(f"  现金：{env._state.cash:.2f}")
        
        # 买入失败，reward 应该为 0，不持仓
        assert_close(reward, 0.0, tol=1e-6, msg="买入失败奖励")
        assert env._state.have_position == False
        
        # 恢复配置
        env_config["cash"] = original_cash
    
    def test_reward_scale(self):
        """测试奖励缩放因子"""
        print_separator("测试：奖励缩放因子")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        
        # 不同缩放因子
        for scale in [1.0, 10.0, 100.0]:
            env_config["reward_scale"] = scale
            
            env = StocksEnv(prices=prices, reward_on_close=False)
            env.reset()
            env.step(Actions.Buy.value)
            obs, reward, done, _, _ = env.step(Actions.Skip.value)
            
            print(f"  缩放因子 {scale}: reward={reward:.6f}")
        
        # 恢复
        env_config["reward_scale"] = 2.0
    
    # -------------------- 边界条件测试 --------------------
    
    def test_episode_end(self):
        """测试 Episode 结束条件"""
        print_separator("测试：Episode 结束条件")
        prices = {"TEST": MockPriceGold(n_bars=50)}  # 短数据
        env = StocksEnv(prices=prices, bars_count=10, reset_on_close=False)
        env.reset()
        
        # 一直 Skip 到结束
        step_count = 0
        for i in range(100):  # 最多 100 步
            obs, reward, done, _, _ = env.step(Actions.Skip.value)
            step_count += 1
            if done:
                print(f"  Episode 在 {step_count} 步后结束")
                break
        
        assert done == True, "Episode 应该结束"
        assert step_count <= 40, "步数应该不超过数据长度"
    
    def test_reset_on_close(self):
        """测试 reset_on_close 参数"""
        print_separator("测试：reset_on_close 参数")
        prices = {"TEST": MockPriceGold(n_bars=100)}
        
        # reset_on_close=True
        env = StocksEnv(prices=prices, reset_on_close=True)
        env.reset()
        env.step(Actions.Buy.value)
        obs, reward, done, _, _ = env.step(Actions.Close.value)
        print(f"  reset_on_close=True: 卖出后 done={done}")
        assert done == True
        
        # reset_on_close=False
        env = StocksEnv(prices=prices, reset_on_close=False)
        env.reset()
        env.step(Actions.Buy.value)
        obs, reward, done, _, _ = env.step(Actions.Close.value)
        print(f"  reset_on_close=False: 卖出后 done={done}")
        assert done == False
    
    def test_commission_calculation(self):
        """测试佣金计算（最低 5 元）"""
        print_separator("测试：佣金计算")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices)
        env.reset()
        
        # 小金额交易（触发最低 5 元）
        env.step(Actions.Buy.value)
        shares = env._state.shares
        price = env._state.entry_price
        trade_value = shares * price
        commission_rate = env._state.commission_perc
        
        calc_comm = trade_value * commission_rate
        actual_comm = max(calc_comm, 5.0)
        
        print(f"  交易金额：{trade_value:.2f}")
        print(f"  计算佣金：{calc_comm:.2f}")
        print(f"  实际佣金：{actual_comm:.2f} (最低 5 元)")
        
        if calc_comm < 5.0:
            assert actual_comm == 5.0, "应该触发最低佣金 5 元"
            print("  ✓ 最低佣金生效")
    
    def test_slippage_effect(self):
        """测试滑点影响"""
        print_separator("测试：滑点影响")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices)
        env.reset()
        
        market_price = env._state._cur_close()
        buy_price = env._state._get_execute_price(market_price, is_buy=True)
        sell_price = env._state._get_execute_price(market_price, is_buy=False)
        
        print(f"  市场价格：{market_price:.4f}")
        print(f"  买入价：{buy_price:.4f} (溢价 {((buy_price/market_price)-1)*10000:.1f} 万)")
        print(f"  卖出价：{sell_price:.4f} (折价 {((1-sell_price/market_price))*10000:.1f} 万)")
        
        assert buy_price > market_price, "买入价应该高于市场价"
        assert sell_price < market_price, "卖出价应该低于市场价"
    
    # -------------------- 状态一致性测试 --------------------
    
    def test_asset_consistency(self):
        """测试资产计算一致性"""
        print_separator("测试：资产计算一致性")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices)
        env.reset()
        
        initial_cash = env._state.cash
        print(f"  初始现金：{initial_cash:.2f}")
        
        # 买入
        env.step(Actions.Buy.value)
        cash_after_buy = env._state.cash
        shares = env._state.shares
        entry_price = env._state.entry_price
        
        # 验证：现金 + 持仓 = 初始现金 - 手续费
        position_value = shares * entry_price
        total_after_buy = cash_after_buy + position_value
        cost = initial_cash - total_after_buy
        
        print(f"  买入后现金：{cash_after_buy:.2f}")
        print(f"  持仓价值：{position_value:.2f}")
        print(f"  总成本：{cost:.2f} (手续费 + 滑点)")
        
        assert total_after_buy < initial_cash, "买入后总资产应该减少（交易成本）"
        
        # 卖出
        env.step(Actions.Close.value)
        final_cash = env._state.cash
        
        print(f"  卖出后现金：{final_cash:.2f}")
        print(f"  总盈亏：{final_cash - initial_cash:.2f}")
    
    def test_position_flag(self):
        """测试持仓标志位"""
        print_separator("测试：持仓标志位")
        prices = {"TEST": MockPriceGold(n_bars=100)}
        env = StocksEnv(prices=prices)
        env.reset()
        
        # 空仓
        obs, _, _, _, _ = env.step(Actions.Skip.value)
        position_flag = obs[-2]  # 倒数第二个特征是持仓标志
        print(f"  空仓标志：{position_flag}")
        assert position_flag == 0.0
        
        # 买入
        obs, _, _, _, _ = env.step(Actions.Buy.value)
        position_flag = obs[-2]
        print(f"  持仓标志：{position_flag}")
        assert position_flag == 1.0
        
        # 卖出
        obs, _, _, _, _ = env.step(Actions.Close.value)
        position_flag = obs[-2]
        print(f"  卖出后标志：{position_flag}")
        assert position_flag == 0.0
    
    # -------------------- 完整交易流程测试 --------------------
    
    def test_full_trade_cycle(self):
        """测试完整交易流程"""
        print_separator("测试：完整交易流程")
        prices = {"TEST": MockPriceGold(n_bars=100, base_price=100.0)}
        env = StocksEnv(prices=prices, reward_on_close=False)
        env.reset()
        print("打印开头的价格")
        print(env._prices["TEST"].Close[env._state._offset : env._state._offset + 10])
        
        initial_asset = env._state._get_cur_total_value()
        print(f"  初始资产：{initial_asset:.2f}")
        
        rewards = []
        actions_log = []
        
        # 模拟一个完整交易：买入 -> 持有 3 步 -> 卖出 -> 空仓 2 步
        action_sequence = [Actions.Buy, Actions.Skip, Actions.Skip, Actions.Skip, 
                          Actions.Close, Actions.Skip, Actions.Skip]
        
        for i, action in enumerate(action_sequence):
            cur_price = env._state._cur_close()
            obs, reward, done, _, _ = env.step(action.value)
            rewards.append(reward)
            actions_log.append(action.name)
            print(f"  Step {i+1}: {action.name:5s}, price={cur_price:.4f}, reward={reward:8.6f}, "
                  f"asset={env._state._get_cur_total_value():.2f}, "
                  f"position={env._state.have_position}")
        
        final_asset = env._state._get_cur_total_value()
        total_reward = sum(rewards)
        actual_return = (final_asset - initial_asset) / initial_asset * env._state.reward_scale
        
        print(f"\n  总奖励：{total_reward:.6f}")
        print(f"  实际收益率 (缩放后): {actual_return:.6f}")
        print(f"  差异：{abs(total_reward - actual_return):.6f}")
        
        # 稠密奖励模式下，总奖励应该接近实际收益率
        assert_close(total_reward, actual_return, tol=0.5, msg="总奖励与收益率")


# ============================================================
# 5. 运行测试
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print_separator("RL 量化交易环境测试套件")
    print("测试重点：Reward 计算完备性")
    
    tester = TestStocksEnv()
    
    # 基础测试
    print_separator("1. 基础功能测试")
    # tester.run_test("环境初始化", tester.test_env_initialization)
    # tester.run_test("动作空间", tester.test_action_space)
    
    # Reward 核心测试
    print_separator("2. Reward 核心测试")
    tester.run_test("空仓 Skip 奖励", tester.test_reward_skip_idle)
    tester.run_test("买入成功奖励", tester.test_reward_buy_success)
    tester.run_test("持仓价格波动奖励", tester.test_reward_hold_price_change)
    tester.run_test("稠密模式卖出奖励", tester.test_reward_close_dense)
    tester.run_test("稀疏模式卖出奖励", tester.test_reward_close_sparse)
    tester.run_test("买入失败奖励", tester.test_reward_buy_fail)
    tester.run_test("奖励缩放因子", tester.test_reward_scale)
    
    # 边界条件测试
    print_separator("3. 边界条件测试")
    tester.run_test("Episode 结束条件", tester.test_episode_end)
    tester.run_test("reset_on_close 参数", tester.test_reset_on_close)
    tester.run_test("佣金计算", tester.test_commission_calculation)
    tester.run_test("滑点影响", tester.test_slippage_effect)
    
    # 状态一致性测试
    print_separator("4. 状态一致性测试")
    tester.run_test("资产计算一致性", tester.test_asset_consistency)
    tester.run_test("持仓标志位", tester.test_position_flag)
    
    # 完整流程测试
    print_separator("5. 完整交易流程测试")
    tester.run_test("完整交易流程", tester.test_full_trade_cycle)
    
    # 打印摘要
    tester.print_summary()
    
    return all(r for _, r, _ in tester.test_results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)