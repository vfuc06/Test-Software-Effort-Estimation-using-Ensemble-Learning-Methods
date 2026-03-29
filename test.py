import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

# ======================
# LOAD DATA
# ======================
data = pd.read_csv("isbsg10.csv")  

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ======================
# SCALE DATA
# ======================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# BASE MODELS
# ======================
knn = KNeighborsRegressor(n_neighbors=3)

extra = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42
)

gboost = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

linear = LinearRegression()

# ======================
# VOTING MODELS
# ======================

#  Voting 1: 2 Extra Trees
voting1 = VotingRegressor([
    ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)),
    ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0))
])

#  Voting 2: 2 Extra Trees + 2 Gradient Boosting
voting2 = VotingRegressor([
    ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)),
    ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)),
    ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0))
])

# Voting 3: 2 Extra Trees + 2 Gradient Boosting + 2 Linear Regression
voting3 = VotingRegressor([
    ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)),
    ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)),

    ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0)),

    ('lr1', LinearRegression()),
    ('lr2', LinearRegression())
])

# ======================
# MODEL LIST
# ======================
models = {
    "KNN": knn,
    "Extra Trees": extra,
    "Voting 1": voting1,
    "Voting 2": voting2,
    "Voting 3": voting3
}

# ======================
# EVALUATE FUNCTION
# ======================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ======================
# TRAIN & TEST
# ======================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, y_pred)

    print(f"\n{name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")