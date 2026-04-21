from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# ==========================================
# PHẦN 1: HỆ THỐNG ĐA TẬP DỮ LIỆU (ACADEMIC)
# ==========================================
system_data = {}

def train_dataset(name, filename, target_col, drop_cols=[]):
    print(f"[ACADEMIC] Đang xử lý Dataset: {name.upper()}...")
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
        'Linear Regression': LinearRegression(),
        'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=42),
        'Voting (2 ETs)': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0))]),
        'Voting (ET+GB)': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0))]),
        'Voting (Full)': VotingRegressor([('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0)), ('lr1', LinearRegression()), ('lr2', LinearRegression())])
    }

    metrics = {}
    for m_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        yp = model.predict(X_test_scaled)
        metrics[m_name] = {
            "mae": float(round(mean_absolute_error(y_test, yp), 2)),
            "rmse": float(round(np.sqrt(mean_squared_error(y_test, yp)), 2)),
            "r2": float(round(r2_score(y_test, yp), 4))
        }

    samples = []
    for idx in X_test.sample(min(3, len(X_test)), random_state=42).index:
        raw_feat = X_test.loc[idx].to_dict()
        clean_feat = {str(k): float(v) for k, v in raw_feat.items()}
        samples.append({"features": clean_feat, "actual": float(round(y_test.loc[idx], 2))})

    system_data[name] = {
        "models": models, "scaler": scaler, "metrics": metrics,
        "feature_names": list(X.columns), "samples": samples
    }
    print(f"  -> Thành công! Đã train {len(models)} mô hình cho {name.upper()}")

dataset_configs = [
    ('isbsg10', 'isbsg10.csv', 'effort', []),
    ('finnish', 'finnish.csv', 'effort', []),
    ('desharnais', 'Desharnais.csv', 'Effort', ['id', 'Project'])
]
for conf in dataset_configs: train_dataset(*conf)


# ==========================================
# PHẦN 2: HỆ THỐNG TỰ HỌC (CONTINUOUS LEARNING)
# ==========================================
company_models = {}
company_metrics = {}
best_company_model = ""
DATA_FILE = "du_lieu_cong_ty.csv"

def prepare_company_data():
    global company_models, company_metrics, best_company_model
    print("\n[CONTINUOUS] Đang đào tạo hệ thống AI Công ty...")
    
    if not os.path.exists(DATA_FILE):
        np.random.seed(42)
        n = 150
        Size = np.random.randint(100, 1000, n)
        Team = np.random.randint(5, 25, n)
        Complexity = np.random.randint(1, 10, n)
        Effort = (Size * 15) + (Team * 100) + (Complexity * 200) + np.random.normal(0, 300, n)
        pd.DataFrame({'Size': Size, 'Team': Team, 'Complexity': Complexity, 'Effort': Effort}).to_csv(DATA_FILE, index=False)
    
    df = pd.read_csv(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(df[['Size', 'Team', 'Complexity']], df['Effort'], test_size=0.2, random_state=42)

    et1, et2 = ExtraTreesRegressor(random_state=42), ExtraTreesRegressor(random_state=123)
    gb1, gb2 = GradientBoostingRegressor(random_state=42), GradientBoostingRegressor(random_state=123)
    lr1, lr2 = LinearRegression(), LinearRegression()

    models = {
        'Voting 1 (2 ETs)': VotingRegressor([('et1', et1), ('et2', et2)]),
        'Voting 2 (ET + GB)': VotingRegressor([('et1', et1), ('et2', et2), ('gb1', gb1), ('gb2', gb2)]),
        'Voting 3 (Full)': VotingRegressor([('et1', et1), ('et2', et2), ('gb1', gb1), ('gb2', gb2), ('lr1', lr1), ('lr2', lr2)])
    }

    lowest_rmse = float('inf')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        company_models[name] = model
        company_metrics[name] = {'RMSE': rmse, 'MAE': mean_absolute_error(y_test, y_pred), 'R2': r2_score(y_test, y_pred) * 100}
        
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_company_model = name
            
    print(f"✅ Đã Train xong AI Công ty! Vô địch: {best_company_model}\n")

prepare_company_data()


# ==========================================
# PHẦN 3: API ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/academic/meta', methods=['GET'])
def get_meta_data():
    meta = {k: {"feature_names": v["feature_names"], "metrics": v["metrics"], "samples": v["samples"]} for k, v in system_data.items()}
    return jsonify(meta)

@app.route('/api/academic/predict', methods=['POST'])
def predict_academic():
    try:
        req = request.json
        ds_name = req.get('dataset')
        features = req.get('features') 
        
        ds = system_data[ds_name]
        input_df = pd.DataFrame([features], columns=ds['feature_names'])
        input_scaled = ds['scaler'].transform(input_df)
        
        predictions = []
        for m_name, model in ds['models'].items():
            predictions.append({
                "model_name": m_name,
                "effort": float(round(model.predict(input_scaled)[0], 2)),
                "rmse": ds['metrics'][m_name]['rmse'],
                "r2": ds['metrics'][m_name]['r2'] * 100
            })
            
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/company/predict', methods=['POST'])
def predict_company():
    data = request.json
    new_project = pd.DataFrame({'Size': [data.get('size', 0)], 'Team': [data.get('team', 0)], 'Complexity': [data.get('complexity', 0)]})
    
    results = []
    for name, model in company_models.items():
        results.append({
            "model_name": name,
            "effort": model.predict(new_project)[0],
            "mae": company_metrics[name]['MAE'],
            "rmse": company_metrics[name]['RMSE'],
            "r2": company_metrics[name]['R2'],
            "is_winner": (name == best_company_model)
        })
    return jsonify(results)

@app.route('/api/company/save_and_retrain', methods=['POST'])
def save_and_retrain():
    data = request.json
    new_row = pd.DataFrame([{'Size': data['size'], 'Team': data['team'], 'Complexity': data['complexity'], 'Effort': data['actual_effort']}])
    new_row.to_csv(DATA_FILE, mode='a', header=False, index=False)
    
    prepare_company_data()
    return jsonify({"status": "success", "message": "Đã lưu vào CSV và đào tạo lại AI thành công!", "new_best_model": best_company_model})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)