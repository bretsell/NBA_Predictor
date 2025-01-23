import pandas as pd
import numpy as np
import random
import os
import warnings

# Suppress TensorFlow's repeated retracing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# NBA API
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# Scikit-Learn
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# Use the past 3 seasons for training
SEASONS = ["2021-22", "2022-23", "2023-24"]

def get_player_id(player_name: str):
    """
    Returns the NBA player_id for a given player name, or None if not found.
    """
    nba_players = players.get_players()
    player = next((p for p in nba_players if p["full_name"].lower() == player_name.lower()), None)
    if player:
        return player["id"]
    else:
        print(f"[ERROR] Could not find '{player_name}' in nba_api.")
        return None

def fetch_player_logs_multi_seasons(player_id: int, seasons: list) -> pd.DataFrame:
    """
    Fetch logs for the specified seasons, concatenate, sort by date.
    """
    all_logs = []
    for season in seasons:
        try:
            logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            # Typically "MM/DD/YYYY" in nba_api
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], format="%m/%d/%Y", errors="coerce")
            logs["SEASON"] = season
            all_logs.append(logs)
        except Exception as e:
            print(f"[ERROR] Unable to fetch logs for {season}: {e}")

    if not all_logs:
        return pd.DataFrame()

    df = pd.concat(all_logs, ignore_index=True)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df

def parse_team_and_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'TEAM_ABBREV' from MATCHUP if needed, 'OPP_ABBREV' from MATCHUP last token.
    """
    if "TEAM_ABBREVIATION" not in df.columns:
        df["TEAM_ABBREV"] = df["MATCHUP"].apply(lambda x: str(x).split(" ")[0])
    else:
        df.rename(columns={"TEAM_ABBREVIATION": "TEAM_ABBREV"}, inplace=True)

    # Opponent is the last token in MATCHUP, e.g. "LAL vs. GSW" => "GSW"
    df["OPP_ABBREV"] = df["MATCHUP"].apply(lambda x: str(x).split(" ")[-1])
    return df

def add_home_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'IS_HOME' feature: 1 if match is at home, 0 if away.
    NBA stats: "vs." => home, "@" => away
    """
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)
    return df

def add_rest_days_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of days since the previous game for each row.
    We'll fill any missing or negative with 1 day.
    """
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days
    # The first game might be NaN or negative if the season changed
    df["REST_DAYS"] = df["REST_DAYS"].fillna(1)
    # If there's a weird scenario with negative days, set them to 1
    df.loc[df["REST_DAYS"] < 1, "REST_DAYS"] = 1
    return df

def create_lag_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lag features for PTS, REB, AST (one-game lag).
    Also creates 3-game rolling average features for PTS, REB, AST,
    shifted by 1 so as not to include the current row's game in the average (no data leakage).
    """
    df["PTS_LAG"] = df["PTS"].shift(1)
    df["REB_LAG"] = df["REB"].shift(1)
    df["AST_LAG"] = df["AST"].shift(1)

    df["PTS_ROLL_3"] = df["PTS"].rolling(window=3).mean().shift(1)
    df["REB_ROLL_3"] = df["REB"].rolling(window=3).mean().shift(1)
    df["AST_ROLL_3"] = df["AST"].rolling(window=3).mean().shift(1)

    df = df.dropna(subset=[
        "PTS_LAG", "REB_LAG", "AST_LAG",
        "PTS_ROLL_3", "REB_ROLL_3", "AST_ROLL_3"
    ])
    return df

def build_dataset(df: pd.DataFrame):
    """
    Build X matrix with:
      - Categorical: TEAM_ABBREV, OPP_ABBREV
      - Numeric: PTS_LAG, REB_LAG, AST_LAG, PTS_ROLL_3, REB_ROLL_3, AST_ROLL_3, IS_HOME, REST_DAYS
    Multi-output Y: PTS, REB, AST.
    """
    req_cols = [
        "TEAM_ABBREV", "OPP_ABBREV", "PTS", "REB", "AST",
        "PTS_LAG", "REB_LAG", "AST_LAG",
        "PTS_ROLL_3", "REB_ROLL_3", "AST_ROLL_3",
        "IS_HOME", "REST_DAYS"
    ]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"[ERROR] Missing column '{c}' in DataFrame after feature engineering.")

    X = df[[
        "TEAM_ABBREV", "OPP_ABBREV",
        "PTS_LAG", "REB_LAG", "AST_LAG",
        "PTS_ROLL_3", "REB_ROLL_3", "AST_ROLL_3",
        "IS_HOME", "REST_DAYS"
    ]].copy()

    Y = df[["PTS", "REB", "AST"]].values
    return X, Y, df

def build_preprocessor():
    """
    OneHotEncode team abbreviations, scale numeric columns, then remove any zero-variance columns.
    """
    cat_cols = ["TEAM_ABBREV", "OPP_ABBREV"]
    num_cols = [
        "PTS_LAG", "REB_LAG", "AST_LAG",
        "PTS_ROLL_3", "REB_ROLL_3", "AST_ROLL_3",
        "IS_HOME", "REST_DAYS"
    ]

    # We'll apply:
    # - OneHotEncoder to cat_cols
    # - StandardScaler to num_cols
    # - Then remove zero-variance columns (VarianceThreshold)
    from sklearn.pipeline import Pipeline

    column_transformer = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    preprocessor = Pipeline([
        ("ct", column_transformer),
        ("vt", VarianceThreshold(threshold=0.0))
    ])
    return preprocessor

def build_keras_model(input_dim, hidden_size=64, hidden_layers=2, learning_rate=0.001):
    """
    Build a Keras feedforward network with variable hidden_size, hidden_layers, and learning_rate.
    Output dimension = 3 (PTS, REB, AST).
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(hidden_size, activation='relu'))
    # final layer with 3 outputs (pts, reb, ast)
    model.add(keras.layers.Dense(3, activation='linear'))

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss='mean_squared_error'
    )
    return model

def random_search_hyperparams(X, Y, n_iter=5, folds=3,
                              hidden_sizes=[32,64,128],
                              hidden_layers_list=[1,2,3],
                              learning_rates=[0.001, 0.0005, 0.0001],
                              epochs_options=[30,40],
                              batch_options=[16,32]):
    """
    Simple random search for hyperparams, using K-fold cross-validation.
    We'll randomly sample from hidden_sizes, hidden_layers, learning_rates, epochs, batch_size.
    Returns the best hyperparam config + best average MAE.
    """
    from sklearn.model_selection import KFold

    best_mae = float('inf')
    best_config = None

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for i in range(n_iter):
        hs = random.choice(hidden_sizes)
        hl = random.choice(hidden_layers_list)
        lr = random.choice(learning_rates)
        ep = random.choice(epochs_options)
        bs = random.choice(batch_options)

        fold_maes = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            model = build_keras_model(
                input_dim=X.shape[1],
                hidden_size=hs,
                hidden_layers=hl,
                learning_rate=lr
            )
            model.fit(
                X_train, Y_train,
                epochs=ep,
                batch_size=bs,
                verbose=0
            )
            preds_val = model.predict(X_val)
            val_mae = mean_absolute_error(Y_val, preds_val)
            fold_maes.append(val_mae)

        avg_mae = np.mean(fold_maes)
        print(f"Config {i+1}/{n_iter}: (hs={hs}, hl={hl}, lr={lr}, ep={ep}, bs={bs}) -> CV MAE={avg_mae:.3f}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_config = (hs, hl, lr, ep, bs)

    print(f"\nBest config: hidden_size={best_config[0]}, hidden_layers={best_config[1]}, lr={best_config[2]}, "
          f"epochs={best_config[3]}, batch_size={best_config[4]} with MAE={best_mae:.3f}")
    return best_config, best_mae

def main():
    print("=== NBA Player Predictor with Rolling Features, Hyperparameter Tuning, Cross-Validation, and Fixes ===\n")

    player_name = input("Enter an NBA player's full name: ").strip()
    if not player_name:
        print("[ERROR] Must enter a player name.")
        return

    opponent_abbrev = input("Enter the opponent's team abbreviation (e.g., 'GSW'): ").strip().upper()
    if not opponent_abbrev:
        print("[ERROR] Must enter an opponent abbreviation.")
        return

    # 1) Get player ID
    pid = get_player_id(player_name)
    if not pid:
        return

    # 2) Fetch logs from multiple seasons
    print(f"Fetching logs for seasons: {SEASONS}")
    logs = fetch_player_logs_multi_seasons(pid, SEASONS)
    if logs.empty:
        print("[ERROR] No logs found. Exiting.")
        return

    # 3) Feature engineering steps:
    logs = parse_team_and_opponent(logs)
    logs = add_home_feature(logs)
    logs = add_rest_days_feature(logs)
    logs = create_lag_and_rolling(logs)
    if logs.empty:
        print("[ERROR] After feature engineering, no data available.")
        return

    # 4) Build dataset
    X_raw, Y, logs = build_dataset(logs)
    print(f"Final data has {len(X_raw)} samples after combining multiple seasons.")

    # 5) Preprocess: OneHot + scale numeric + remove zero-variance
    preprocessor = build_preprocessor()
    X_array = preprocessor.fit_transform(X_raw)
    print(f"Transformed feature shape: {X_array.shape}, Targets shape: {Y.shape}\n")

    # 6) Random hyperparam search with cross-validation
    print("--- Hyperparameter Tuning via Random Search ---\n")
    best_config, best_mae = random_search_hyperparams(X_array, Y, n_iter=5, folds=3)
    hs, hl, lr, ep, bs = best_config

    # 7) Train final model with the best config on ALL data (increase epochs if you want more training)
    print(f"\nTraining final model with best config: hidden_size={hs}, hidden_layers={hl}, lr={lr}, epochs={ep}, batch_size={bs}")
    final_model = build_keras_model(input_dim=X_array.shape[1], hidden_size=hs, hidden_layers=hl, learning_rate=lr)
    final_model.fit(
        X_array, Y,
        epochs=ep,
        batch_size=bs,
        verbose=0
    )

    # 8) Next-game scenario: use last row's stats for "recent performance," but user-chosen OPP_ABBREV
    last_row = logs.iloc[-1]
    X_next_raw = pd.DataFrame([{
        "TEAM_ABBREV": last_row["TEAM_ABBREV"],
        "OPP_ABBREV": opponent_abbrev,
        "PTS_LAG": last_row["PTS_LAG"],
        "REB_LAG": last_row["REB_LAG"],
        "AST_LAG": last_row["AST_LAG"],
        "PTS_ROLL_3": last_row["PTS_ROLL_3"],
        "REB_ROLL_3": last_row["REB_ROLL_3"],
        "AST_ROLL_3": last_row["AST_ROLL_3"],
        "IS_HOME": last_row["IS_HOME"],
        "REST_DAYS": last_row["REST_DAYS"]
    }])

    X_next_array = preprocessor.transform(X_next_raw)
    preds_next = final_model.predict(X_next_array)[0]  # shape: (3,)

    pred_pts, pred_reb, pred_ast = preds_next
    print("\n=== Next Game Prediction ===")
    print(f"Player: {player_name}")
    print(f"Opponent: {opponent_abbrev}")
    print(f"Projected Points:   {pred_pts:.1f}")
    print(f"Projected Rebounds: {pred_reb:.1f}")
    print(f"Projected Assists:  {pred_ast:.1f}")


if __name__ == "__main__":
    main()
