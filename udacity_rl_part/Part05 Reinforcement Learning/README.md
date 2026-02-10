# Reinforcement Learning Trading Agent

## Project Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent for automated stock trading. The agent learns to make buy, sell, and hold decisions based on historical stock price data and technical indicators, with the goal of maximizing trading profits.

## Purpose

The primary objectives of this project are:

- **Automated Trading**: Develop an AI agent that can autonomously make trading decisions
- **Profit Maximization**: Train the agent to identify profitable trading opportunities
- **Risk Management**: Implement strategies to minimize losses while maximizing gains
- **Market Analysis**: Use technical indicators (Bollinger Bands) to inform trading decisions
- **Learning from Experience**: Utilize experience replay to improve decision-making over time

## Features

### Core Components

1. **Deep Q-Network (DQN) Model**
   - 4-layer neural network architecture
   - Input: State representation (price data + technical indicators)
   - Output: Q-values for buy, sell, and hold actions
   - Optimized with Adam optimizer and MSE loss

2. **Reinforcement Learning Agent**
   - Epsilon-greedy exploration strategy
   - Experience replay for stable learning
   - Memory buffer for storing trading experiences
   - Automatic model checkpointing

3. **Technical Analysis**
   - Bollinger Bands calculation
   - Price normalization and scaling
   - Feature engineering for state representation

4. **Trading Environment**
   - Realistic trading simulation
   - Profit/loss tracking
   - Buy/sell signal visualization
   - Performance metrics calculation

### Key Features

- **State Representation**: Combines historical price data with technical indicators
- **Action Space**: Three actions (buy, sell, hold) with intelligent inventory management
- **Reward System**: Profit-based rewards with loss protection
- **Training Visualization**: Real-time plotting of trading decisions and performance
- **Model Persistence**: Automatic saving of trained models for later use

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Required Dependencies

Install the following Python packages:

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tqdm
```

### Alternative: Install via requirements.txt

Create a `requirements.txt` file with the following contents:

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

Then install using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
3 Reinforcement Learning/
├── Building a Reinforcement Learning Trading Model.ipynb  # Main Jupyter notebook with complete implementation
├── GOOG_2009-2010_6m_RAW_1d.csv  # Sample stock data (Google 2009-2010)
├── model_ep0.keras          # Trained model checkpoint (Episode 0)
├── model_ep1.keras          # Trained model checkpoint (Episode 1)
├── model_ep2.keras          # Trained model checkpoint (Episode 2)
└── README.md               # This file
```

## Usage

### Running the Project

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Navigate to the project folder** and open `Building a Reinforcement Learning Trading Model.ipynb`

   > **Note**: The main implementation notebook is named "Building a Reinforcement Learning Trading Model.ipynb" and contains the complete DQN trading agent implementation.

3. **Run the notebook cells sequentially**:
   - Data loading and preprocessing
   - Exploratory data analysis
   - Feature engineering
   - Model training
   - Performance evaluation

### Training Process

The training process consists of multiple episodes:

1. **Data Preparation**: Load and clean historical stock data
2. **Feature Engineering**: Calculate technical indicators (Bollinger Bands)
3. **State Creation**: Combine price data with technical indicators
4. **Agent Training**: Train the DQN agent using experience replay
5. **Model Evaluation**: Test the trained model on unseen data

### Key Parameters

- **Window Size**: Number of historical days to include in state (default: 20)
- **Batch Size**: Number of experiences for training (default: 32)
- **Episodes**: Number of training episodes (default: 3)
- **Epsilon**: Exploration rate (starts at 1.0, decays to 0.01)
- **Gamma**: Discount factor for future rewards (default: 0.95)

## Model Architecture

The DQN model consists of:

- **Input Layer**: Dense layer with 64 units (ReLU activation)
- **Hidden Layer 1**: Dense layer with 32 units (ReLU activation)
- **Hidden Layer 2**: Dense layer with 8 units (ReLU activation)
- **Output Layer**: Dense layer with 3 units (linear activation)

## Trading Strategy

The agent implements the following trading logic:

1. **Buy Signal**: When the agent decides to buy, shares are added to inventory
2. **Sell Signal**: When the agent decides to sell, shares are sold from inventory
3. **Hold Signal**: No action taken, agent continues monitoring
4. **Profit Calculation**: Rewards are based on actual profit/loss from trades

## Performance Metrics

The system tracks:

- **Total Profit**: Cumulative profit/loss from all trades
- **Winning Trades**: Total profit from profitable trades
- **Losing Trades**: Total loss from unprofitable trades
- **Training Loss**: Model learning progress
- **Trading Signals**: Buy/sell decision visualization

## Customization

### Using Different Data

To use your own stock data:

1. Replace `GOOG_2009-2010_6m_RAW_1d.csv` with your dataset
2. Ensure your data has columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
3. Update the data loading section in the notebook

### Modifying Parameters

Adjust training parameters in the notebook:

```python
# Training parameters
episodes = 5          # Increase for more training
batch_size = 64       # Larger batches for faster training
window_size = 30      # More historical data for state
```

### Adding Technical Indicators

To include additional technical indicators:

1. Calculate the indicator in the feature engineering section
2. Add it to the `static_dataset` DataFrame
3. Update the `num_features` parameter accordingly

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch_size or window_size
2. **Training Instability**: Adjust epsilon decay rate
3. **Poor Performance**: Increase number of episodes or adjust reward function
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Tips

- Use GPU acceleration if available (TensorFlow will automatically detect)
- Monitor training loss to ensure convergence
- Experiment with different hyperparameters
- Consider ensemble methods for improved performance

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations when using for actual trading.

## Disclaimer

This project is for educational purposes only. The trading strategies implemented are not financial advice. Always conduct thorough research and consider consulting with financial professionals before making investment decisions. Past performance does not guarantee future results.

## Contact

For questions or contributions, please open an issue in the repository or contact the project maintainers. 