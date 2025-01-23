# NBA_Predictor
NBA Player Performance Predictor
Status: Ongoing Development

Overview
This repository contains a multi-season NBA player performance predictor that uses Python, pandas, and TensorFlow/Keras to forecast points, rebounds, and assists for a given player’s next game. The model ingests historical data from the nba_api library, creates advanced features (e.g., rolling averages, rest days, home/away flags), and trains a neural network to produce multi-output predictions.

I created this project to strengthen my data science and machine learning skills, practice feature engineering, and build a start-to-finish ML pipeline in Python.

Tools & Technologies
Python
pandas, NumPy for data manipulation
nba_api for fetching real NBA data
scikit-learn for transformations, one-hot encoding, cross-validation
TensorFlow/Keras for building and training the neural network
Randomized hyperparameter search for tuning model hyperparameters
Project Highlights
Multi-Season Data: Combines logs from multiple NBA seasons, creating a richer training set.
Advanced Feature Engineering:
Lag features (previous game stats)
Rolling averages (3-game rolling stats)
Home/Away flags
Rest days (days since last game)
Neural Network with Multi-Output: Predicts points, rebounds, and assists in a single forward pass, with separate outputs for each stat.
How It Works
Data Fetching: Uses nba_api to gather a player’s game logs for multiple seasons.
Preprocessing & Feature Engineering:
One-hot encoding for team/opponent abbreviations
Scaling numeric features
Variance threshold to remove uninformative columns
Training & Tuning:
Cross-validation to detect overfitting
Random search to find the best hyperparameters (hidden layers, learning rate, etc.)
A final Keras model is trained on all data for the best config
Next Game Prediction:
The script uses the player’s last game stats (plus rolling averages) as the “most recent form”
User supplies the opponent team abbreviation
The model outputs predicted PTS, REB, AST for that matchup
Usage
Install Requirements

bash
Copy
pip install -r requirements.txt
or individually:

bash
Copy
pip install nba_api pandas numpy scikit-learn tensorflow
Run the Script

bash
Copy
python nba_predictor.py
Enter a player’s full name (e.g., LeBron James)
Enter the opponent abbreviation (e.g., GSW)
View the Results

The script performs hyperparameter tuning with cross-validation
Trains a final neural network model
Prints the predicted stats for the player’s next game
Future Plans
I’m still actively working on this project. Planned improvements include:

More Predictions: Expand to other statistics (e.g., steals, blocks, turnovers).
Better User Interface: Possibly a web app using Streamlit or Flask for interactive predictions.
Enhanced Model Architecture: Experiment with LSTM or Transformer-based models if time-series structure becomes more relevant.
Incorporate More Data:
Additional advanced metrics (usage rate, player efficiency rating, real plus-minus, etc.)
Opponent defensive stats (e.g., defensive rating, pace)
More seasons or combining data from multiple players.
Early Stopping & Ensemble Methods: Add advanced training callbacks and possibly ensemble predictions from multiple neural networks.
