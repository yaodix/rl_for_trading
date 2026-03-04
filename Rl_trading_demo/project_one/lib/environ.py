import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
import enum
import numpy as np

from . import data
from . import config

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = config.env_config["commission"]  # 是万0.5吗

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count: int, commission_perc: float, reset_on_close: bool,
                 reward_on_close: bool = False, feat_aug: bool = False):
        assert bars_count > 0
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.feat_aug = feat_aug
        
        # 账户状态
        self.have_position = False
        self.cash = config.env_config["cash"]  # 用现金不好计算，用每次买卖固定股数进行交易
        self.shares = 0.0        
        self.entry_price = 0.0
        self.entry_total_asset = 0.0  # 记录入口总价值（现金 + 持仓市值）
        
         # 新增参数
        self.slippage = config.env_config["slippage"]  # 万1滑点
        self.reward_scale = config.env_config["reward_scale"]  # 奖励缩放因子
        self.idle_penalty = config.env_config["idle_penalty"]  # 空仓惩罚
        
        # 记录连续空仓天数
        self.idle_days = 0
        
        self._prices = None  # 价格数据（稍后设置）
        self._offset = None  # 当前时间偏移量（在价格序列中的位置）

    def reset(self, prices: data.PriceGold, offset: int):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.cash = config.env_config["cash"]
        self.shares = 0.0        
        self.entry_price = 0.0
        self.entry_total_asset = 0.0
        
        self._prices = prices
        self._offset = offset
    
    def _get_execute_price(self, price: float, is_buy: bool) -> float:
        """计算实际成交价（考虑滑点）"""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
        
    def _calculate_commission(self, trade_value: float) -> float:
        """
        计算交易佣金
        万0.5，最低5元
        """
        commission = trade_value * self.commission_perc  # 万0.5
        return max(commission, 5.0)  # 最低5元

    def _get_cur_total_value(self) -> float:
        """计算总资产 = 现金 + 持仓市值"""
        if self.have_position:
            current_price = self._cur_close()
            return self.cash + self.shares * current_price
        else:
            return self.cash

    @property
    def shape(self) -> tt.Tuple[int, ...]:
        # ['Returns', 'Amplitude', 'Volume_Ratio', 'Close_Position'] * bars + position_flag + rel_profit
        if self.feat_aug:
            # + 'RSI_Norm', 'ATR_Ratio', 'EMA_Diff', 'BB_Width', 'Vol20'  # 第二层
            return 9 * self.bars_count + 1 + 1,
        else:
            return 4 * self.bars_count + 1 + 1,
    
    # TODO: 实现特征增强
    def encode(self) -> np.ndarray:
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.Amplitude_Norm[ofs]  # 最高价（相对开盘价的百分比）
            shift += 1
            res[shift] = self._prices.Returns_Norm[ofs]
            shift += 1
            res[shift] = self._prices.Volume_Ratio_Norm[ofs]
            shift += 1
            res[shift] = self._prices.Close_Position_Norm[ofs]
            shift += 1
            
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.entry_price - 1.0
        return res

    def _cur_close(self) -> float:
        """
        Calculate real close price for the current bar
        """
        rel_close = self._prices.Close[self._offset]
        return rel_close

    def step(self, action: Actions) -> tt.Tuple[float, bool]:
        # 当前价格
        current_price = self._cur_close()   
        # 记录动作前的总资产
        prev_total = self._get_cur_total_value()

        done = False
        reward = 0.0
        action_executed = False
        # ===== 执行动作（考虑滑点）=====
        if action == Actions.Buy and not self.have_position:
            # 买入用更高价格
            execute_price = self._get_execute_price(current_price, is_buy=True)            
            max_shares = int(self.cash / execute_price / 100) * 100
            if max_shares >= 100:
                # 预估佣金
                estimated_trade = max_shares * execute_price
                estimated_comm = max(estimated_trade * self.commission_perc, 5)
                
                # 重新计算
                available = self.cash - estimated_comm
                final_shares = int(available / execute_price / 100) * 100
                
                if final_shares >= 100:
                    trade_value = final_shares * execute_price
                    actual_comm = max(trade_value * self.commission_perc, 5)
                    
                    self.shares = final_shares
                    self.cash = self.cash - trade_value - actual_comm
                    self.entry_price = execute_price
                    self.entry_total_asset = self.cash + self.shares * self.entry_price
                    self.have_position = True
                    self.idle_days = 0  # 重置空仓计数
                    action_executed = True
                    
            else: # 无法购买100股以上, 直接结束? 还是学习不购买?
            #     done = True
                print(f"warning: not enough cash to buy 100 shares at {execute_price}")
                                   
        elif action == Actions.Close and self.have_position:
            # 卖出用更低价格
            execute_price = self._get_execute_price(current_price, is_buy=False)
            trade_value = self.shares * execute_price
            commission = max(trade_value * self.commission_perc, 5)
                        
            done |= self.reset_on_close
            if self.reward_on_close:
                final_cash = self.cash + trade_value - commission
                reward += (final_cash - self.entry_total_asset) / self.entry_total_asset
            else:
                # 单步模式，卖出佣金收益，否则后面have_position置false会导致reward为0
                final_cash = self.cash + trade_value - commission  # 
                reward += (final_cash - prev_total) / prev_total
                
            self.cash += trade_value - commission
            self.shares = 0
            self.have_position = False
            self.entry_price = 0.0
            self.entry_total_asset = 0.0
            self.idle_days = 0
            
        else:  # Skip
            self.idle_days += 1
        
        # 移动到下一K线
        self._offset += 1
        new_total = self._get_cur_total_value()
        
        # ===== 计算当前收益率 =====
        if self.have_position and not self.reward_on_close:
            reward += (new_total - prev_total) / prev_total
        
        # 空仓惩罚（连续空仓超过20个K线）
        # if not self.have_position and self.idle_days > 20:
        #     reward -= self.idle_penalty 
        
        # 检查是否结束
        done |= self._offset >= self._prices.Close.shape[0]-1
        reward *= self.reward_scale
        return reward , done

class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self) -> tt.Tuple[int, ...]:
        if self.volumes:
            return 6, self.bars_count
        else:
            return 5, self.bars_count

    def encode(self) -> np.ndarray:
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset-(self.bars_count-1)
        stop = self._offset+1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self._cur_close() / self.entry_price - 1.0
        return res


class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")

    def __init__(
            self, prices: tt.Dict[str, data.PriceGold],
            bars_count: int = DEFAULT_BARS_COUNT,
            commission: float = DEFAULT_COMMISSION_PERC,
            reset_on_close: bool = True, 
            state_1d: bool = False,
            random_ofs_on_reset: bool = True,
            reward_on_close: bool = False, 
            feat_aug=False
    ):
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close,
                                  reward_on_close=reward_on_close, feat_aug=feat_aug)
        else:
            self._state = State(bars_count, commission, reset_on_close,
                                reward_on_close=reward_on_close, feat_aug=feat_aug)
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self, *, seed: int | None = None, options: dict[str, tt.Any] | None = None, start_idx: int = 0):
        # make selection of the instrument and it's offset. Then reset the state
        super().reset(seed=seed, options=options)
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        safe_start = max(start_idx, self._state.bars_count)
        if self.random_ofs_on_reset:
             # -bars为了保证每个 Episode 至少能走 bars_count 步
            offset = self.np_random.choice(prices.Close.shape[0]-bars-safe_start) + safe_start
            
        else:
            offset = safe_start
        self._state.reset(prices, offset)
        return self._state.encode(), {}

    def step(self, action_idx: int) -> tt.Tuple[np.ndarray, float, bool, bool, dict]:
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, False, info

