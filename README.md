# Stock Price Forecasting with SimpleRNN and LSTM

## Overview

This project demonstrates how to use Recurrent Neural Networks (RNNs) to forecast stock prices, focusing specifically on predicting the closing price of the ABB stock. Two code implementations are provided:

1. **SimpleRNN-based model**
2. **LSTM-based model**

The rationale for selecting one stock (ABB) is to streamline training and testing. If the approach works effectively on a single stock, it can be extended to other stocks, thereby reducing experimentation time.

Both implementations integrate sentiment analysis from tweets, technical indicators, and data smoothing techniques to improve the models’ ability to capture complex temporal patterns and long-range dependencies in financial data.

## Key Features and Methodologies

- **Data Inputs:**
  - Historical price data (Open, High, Low, Close, Volume) for ABB.
  - Tweet data related to ABB for sentiment analysis.

- **Feature Engineering:**
  - **Technical Indicators:** Moving averages (7-day, 14-day, 21-day), Bollinger Bands, and Momentum.
  - **Data Smoothing (EMA):** An Exponential Moving Average applied to the closing prices to highlight underlying trends.
  - **Sentiment Analysis:** Incorporating average daily sentiment and tweet counts to capture market psychology.

- **Model Architectures:**
  - **SimpleRNN Model (First Code):** A stacked SimpleRNN with dropout layers. Serves as a baseline.
  - **LSTM Model (Second Code):** A stacked LSTM architecture leveraging gating mechanisms to handle longer-term dependencies and reduce vanishing gradient issues.

- **Training and Optimization:**
  - **Normalization:** Scaling data (RobustScaler for SimpleRNN, MinMaxScaler for LSTM) for stable training.
  - **Learning Rate Scheduling:** Dynamically adjusts learning rates to improve convergence.
  - **Early Stopping:** Prevents overfitting by halting training when validation performance stops improving.

- **Evaluation and Visualization:**
  - Models are evaluated using RMSE and MAE metrics on test data.
  - Plots compare predicted vs. actual prices.
  - Both models provide next-day stock price forecasts, visualized alongside recent history.

## How to Run

1. **Data Preparation:**
   - Ensure `stocknet-dataset/price/raw/ABB.csv` is available.
   - Place tweet data in `stocknet-dataset/tweet/preprocessed/ABB/`.
   - If no tweet data is found, the code runs without sentiment features.

2. **Dependencies:**
   - Python 3.x
   - Required libraries: `pandas`, `numpy`, `json`, `sklearn`, `tensorflow`, `matplotlib`, `nltk`
   - Install required packages:
     ```bash
     pip install pandas numpy scikit-learn tensorflow matplotlib nltk
     ```
   - Ensure NLTK VADER lexicon is downloaded:
     ```python
     import nltk
     nltk.download('vader_lexicon')
     ```

3. **Running the Scripts:**
   - For SimpleRNN:
     ```bash
     python simple_rnn_model.py
     ```
   - For LSTM:
     ```bash
     python lstm_model.py
     ```

4. **Outputs:**
   - Training progress and metrics (RMSE, MAE) in the console.
   - Plots showing predicted vs. actual prices, plus next-day prediction.

## Notes and Future Directions

- The current approach focuses on ABB but can be adapted to other stocks.
- Future work could involve more advanced architectures like GRUs or Transformers.
- Integrating additional data sources (economic indicators, fundamental data) might further improve accuracy.

This project provides a starting point for applying deep learning techniques to financial forecasting, illustrating how careful feature engineering, sentiment analysis, and advanced RNN architectures can yield more informative and reliable forecasts.
