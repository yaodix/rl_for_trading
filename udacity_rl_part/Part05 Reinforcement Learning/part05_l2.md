## Market Features in Financial Data

![1767064139659](image/part05_l2/1767064139659.png)

**Understanding Market Features**

* **Market features** form the basis for developing trading models in reinforcement learning.
* They offer insights derived from raw financial data to capture market behaviour.

**Types of Market Features**

1. **Price-Based Features:**
   * **Open-High-Low-Close (OHLC):** Core data points indicating price movements within a trading period.
   * **Returns:** Percentage change in price, valuable across various time frames.
2. **Volume-Based Features:**
   * **Trade Volume:** Reflects market activity and sentiment.
   * **VWAP (Volume Weighted Average Price):** Indicates average trading price throughout the day, weighted by trade volume.
3. **Technical Indicators:**
   * **Moving Averages:** Identifies trend directions and includes variations like simple and exponential averages.
   * **Relative Strength Index (RSI):** Highlights overbought or oversold market conditions.
   * **MACD (Moving Average Convergence Divergence):** Shows momentum through moving average relationships.
   * **Bollinger Bands:** Visualize price volatility through deviations from a moving average.

**Feature Engineering**

* Enhances model performance by selecting and creating new features.
* Techniques include lag features and rolling statistics for trend analysis.

Leveraging these features aids in maximizing the predictive potential of trading models.


### Techniques of Normalization

![1767078464499](https://file+.vscode-resource.vscode-cdn.net/home/yao/myproject/learning-AI-trading/3%20Reinforcement%20Learning/image/part05_l2/1767078464499.png)T

wo widely-used normalization methods are **MinMax Scaling** and  **Z Score Normalization** :

* **MinMax Scaling** :
* Transforms feature values to a range between 0 and 1.
* Maintains relationships between data points.
* Works well when feature bounds are known yet sensitive to outliers.
* **Z Score Normalization** :
* Centers data at 0 with a standard deviation of 1.
* Suitable for normally distributed data.
* Less sensitive to outliers compared to MinMax.

### 参考

[Technical Indicator: Definition, Analyst Uses, Types and Examples](https://www.investopedia.com/terms/t/technicalindicator.asp)

[Top Technical Indicators for Rookie Traders](https://www.investopedia.com/articles/active-trading/011815/top-technical-indicators-rookie-traders.asp)

[7 Technical Indicators to Build a Trading Toolkit](https://www.investopedia.com/top-7-technical-analysis-tools-4773275)
