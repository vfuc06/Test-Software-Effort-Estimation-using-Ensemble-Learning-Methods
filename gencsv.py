import pandas as pd
import numpy as np

np.random.seed(42)

# =========================
# 1. FINNISH (mức trung bình)
# =========================
n = 200

size = np.random.randint(50, 500, n)
team = np.random.randint(2, 20, n)
complexity = np.random.randint(1, 10, n)

# ÉP FLOAT để tránh lỗi
effort = (size * 15 + team * 60 + complexity * 120).astype(float)

# Thêm noise
effort += np.random.normal(0, 1200, n)

# Thêm outlier
for i in np.random.choice(n, 10):
    effort[i] *= np.random.uniform(1.3, 1.8)

finnish = pd.DataFrame({
    "size": size,
    "team": team,
    "complexity": complexity,
    "effort": effort
})

finnish.to_csv("finnish.csv", index=False)


# =========================
# 2. ISBSG10 (KHÓ – realistic)
# =========================
n = 300

size = np.random.randint(100, 1000, n)
team = np.random.randint(3, 50, n)
complexity = np.random.randint(1, 15, n)

# ÉP FLOAT
effort = ((size ** 1.2) * 10 + team * 80 + complexity * 150).astype(float)

# noise mạnh
effort += np.random.normal(0, 3500, n)

# outlier mạnh
for i in np.random.choice(n, 30):
    effort[i] *= np.random.uniform(1.5, 3)

# bias ngẫu nhiên
bias = np.random.randint(-2000, 2000, n)
effort += bias

isbsg = pd.DataFrame({
    "size": size,
    "team": team,
    "complexity": complexity,
    "effort": effort
})

isbsg.to_csv("isbsg10.csv", index=False)


print("Đã tạo finnish.csv và isbsg10.csv (realistic version)")