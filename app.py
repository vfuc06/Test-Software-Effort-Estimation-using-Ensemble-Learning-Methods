from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

system_data = {}

def train_dataset(name, filename, target_col, drop_cols=[]):
    print(f"Đang xử lý Dataset: {name.upper()}...")
    if not os.path.exists(filename):
        print(f"  -> Bỏ qua: Không tìm thấy file {filename}")
        return

    data = pd.read_csv(filename)
    if drop_cols:
        data = data.drop(columns=drop_cols, errors='ignore')

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'linear': LinearRegression(),
        'extra': ExtraTreesRegressor(n_estimators=200, random_state=42),
        'voting1': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0))]),
        'voting2': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0))]),
        'voting3': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0)), ('lr1', LinearRegression()), ('lr2', LinearRegression())])
    }

    metrics = {}
    for m_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        yp = model.predict(X_test_scaled)
        
        # FIX LỖI "LỎ": Ép kiểu về float nguyên thủy của Python
        metrics[m_name] = {
            "mae": float(round(mean_absolute_error(y_test, yp), 2)),
            "rmse": float(round(np.sqrt(mean_squared_error(y_test, yp)), 2)),
            "r2": float(round(r2_score(y_test, yp), 4))
        }

    samples = []
    # FIX LỖI "LỎ": Đảm bảo dữ liệu mẫu bốc ra không dính kiểu int64/float64
    for idx in X_test.sample(3, random_state=42).index:
        raw_feat = X_test.loc[idx].to_dict()
        clean_feat = {str(k): float(v) for k, v in raw_feat.items()}
        
        samples.append({
            "features": clean_feat,
            "actual": float(round(y_test.loc[idx], 2))
        })

    system_data[name] = {
        "models": models, "scaler": scaler, "metrics": metrics,
        "feature_names": list(X.columns), "samples": samples
    }
    print(f"  -> Thành công! Đã train 5 mô hình cho {name.upper()}")

dataset_configs = [
    ('isbsg10', 'isbsg10.csv', 'effort', []),
    ('finnish', 'finnish.csv', 'effort', []),
    ('desharnais', 'Desharnais.csv', 'Effort', ['id', 'Project'])
]

print("=== KHỞI TẠO HỆ THỐNG ĐA TẬP DỮ LIỆU ===")
for conf in dataset_configs:
    train_dataset(*conf)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_meta_data', methods=['GET'])
def get_meta_data():
    meta = {k: {"feature_names": v["feature_names"], "metrics": v["metrics"], "samples": v["samples"]} for k, v in system_data.items()}
    return jsonify(meta)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.json
        ds_name = req.get('dataset')
        features = req.get('features') 
        
        ds = system_data[ds_name]
        input_df = pd.DataFrame([features], columns=ds['feature_names'])
        input_scaled = ds['scaler'].transform(input_df)

        
        predictions = {m_name: float(round(model.predict(input_scaled)[0], 2)) for m_name, model in ds['models'].items()}
        
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)