import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

def evaluate_datasets():
    dataset_configs = [
        ('ISBSG10', 'isbsg10.csv', 'effort', []),
        ('FINNISH', 'finnish.csv', 'effort', []),
        ('DESHARNAIS', 'Desharnais.csv', 'Effort', ['id', 'Project'])
    ]

    for name, filename, target_col, drop_cols in dataset_configs:
        print(f"\n{'='*50}")
        print(f"ĐANG ĐÁNH GIÁ TẬP DỮ LIỆU: {name}") 
        print(f"{'='*50}")
        
        if not os.path.exists(filename):
            print(f"Lỗi: Không tìm thấy file '{filename}' trong thư mục hiện tại.")
            continue

        # Load và tiền xử lý data
        data = pd.read_csv(filename)
        if drop_cols:
            data = data.drop(columns=drop_cols, errors='ignore')

        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Định nghĩa Models
        models = {
            'Linear Regression': LinearRegression(),
            'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=42),
            'Voting (2 ETs)': VotingRegressor([
                ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), 
                ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0))
            ]),
            'Voting (ET+GB)': VotingRegressor([
                ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), 
                ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), 
                ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), 
                ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0))
            ]),
            'Voting (Full)': VotingRegressor([
                ('et1', ExtraTreesRegressor(n_estimators=200, random_state=42)), 
                ('et2', ExtraTreesRegressor(n_estimators=200, random_state=0)), 
                ('gb1', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)), 
                ('gb2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0)), 
                ('lr1', LinearRegression()), 
                ('lr2', LinearRegression())
            ])
        }

        # Train và In Kết quả dạng Bảng
        print(f"{'Mô hình':<25} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
        print("-" * 62)
        
        for m_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            yp = model.predict(X_test_scaled)
            
            mae = round(mean_absolute_error(y_test, yp), 2)
            rmse = round(np.sqrt(mean_squared_error(y_test, yp)), 2)
            r2 = round(r2_score(y_test, yp), 4)
            
            print(f"{m_name:<25} | {rmse:<10} | {mae:<10} | {r2:<10}")

if __name__ == '__main__':
    evaluate_datasets()