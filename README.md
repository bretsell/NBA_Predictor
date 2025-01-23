# NBA Player Performance Predictor

**Status**: Ongoing Development

## Overview

This repository contains a **multi-season NBA player performance predictor** that uses **Python**, **pandas**, and **TensorFlow/Keras** to forecast **points**, **rebounds**, and **assists** for a given player’s next game. The model ingests historical data from the [`nba_api` library](https://pypi.org/project/nba-api/), creates advanced features (e.g., rolling averages, rest days, home/away flags), and trains a **neural network** to produce multi-output predictions.

I created this project to strengthen my **data science** and **machine learning** skills, practice **feature engineering**, and build a **start-to-finish ML pipeline** in Python.

## Tools & Technologies

- **Python**  
- **pandas**, **NumPy** for data manipulation  
- **nba_api** for fetching real NBA data  
- **scikit-learn** for transformations, one-hot encoding, cross-validation  
- **TensorFlow/Keras** for building and training the neural network  
- **Randomized hyperparameter search** for tuning model hyperparameters

## Project Highlights

1. **Multi-Season Data**: Combines logs from multiple NBA seasons, creating a richer training set.  
2. **Advanced Feature Engineering**:  
   - **Lag features** (previous game stats)  
   - **Rolling averages** (3-game rolling stats)  
   - **Home/Away flags**  
   - **Rest days** (days since last game)  
3. **Neural Network with Multi-Output**: Predicts **points**, **rebounds**, and **assists** in a single forward pass, with separate outputs for each stat.

## How It Works

1. **Data Fetching**: Uses `nba_api` to gather a player’s game logs for multiple seasons.  
2. **Preprocessing & Feature Engineering**:  
   - One-hot encoding for team/opponent abbreviations  
   - Scaling numeric features  
   - Variance threshold to remove uninformative columns  
3. **Training & Tuning**:  
   - **Cross-validation** to detect overfitting  
   - **Random search** to find the best hyperparameters (hidden layers, learning rate, etc.)  
   - A final **Keras model** is trained on all data for the best config  
4. **Next Game Prediction**:  
   - The script uses the player’s last game stats (plus rolling averages) as the “most recent form”  
   - User supplies the opponent team abbreviation  
   - The model outputs predicted **PTS**, **REB**, **AST** for that matchup

## Future Plans
**I’m still actively working on this project. Planned improvements include:**

- **More Predictions:** Expand to other statistics (e.g., steals, blocks, turnovers).
- **Better User Interface:** Possibly a web app using Streamlit or Flask for interactive predictions.
- **Enhanced Model Architecture:** Experiment with LSTM or Transformer-based models if time-series structure becomes more relevant.
- **Incorporate More Data:**
  - Additional advanced metrics (expected field goals, player efficiency rating, team win, etc.)
  - Opponent defensive stats (e.g., defensive rating, pace)
  - More seasons or combining data from multiple players.
