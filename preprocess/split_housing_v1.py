import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# 0. 参数
# -----------------------
PATH = '../data/HousingData.csv'
TARGET_COL = 'MEDV'
TEST_SIZE = 0.2
RANDOM_STATE = 42

DO_SCALE = False

# outlier 比例：例如删除 5% 残差最大的样本
OUTLIER_CONTAMINATION = 0.05


# -----------------------
# 1. 随机森林去除 outlier 的函数
# -----------------------
def remove_outliers_with_random_forest(
    X,
    y,
    contamination=0.05,
    random_state=42
):
    """
    使用 RandomForestRegressor 根据残差大小删除 outlier。
    只在训练集上使用，避免信息泄漏。
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)

    y_pred = rf.predict(X)
    residuals = np.abs(y - y_pred)

    # 以残差分位数为阈值，例如删除 top 5% 最大残差
    threshold = np.quantile(residuals, 1 - contamination)

    mask = residuals <= threshold
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"原始训练集样本数: {len(y)}, 去除 outlier 后样本数: {len(y_clean)}")
    print(f"残差阈值 = {threshold:.4f}")

    return X_clean, y_clean, rf, residuals, threshold


# -----------------------
# 2. 读取数据
# -----------------------
df = pd.read_csv(PATH)
print("Original shape:", df.shape)
print(df.head())

# -----------------------
# 3. 特征 & 目标
# -----------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------
# 4. 对连续目标做分箱，用于分层抽样
# -----------------------
y_bins = pd.cut(
    y,
    bins=10,            # 分成 10 桶，可按需要调整
    labels=False
)

# -----------------------
# 5. 分层划分训练集 / 测试集
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_bins
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# -----------------------
# 6. 缺失值填充（先填充再做随机森林）
# -----------------------
num_imputer = SimpleImputer(strategy='median')

X_train_imputed = pd.DataFrame(
    num_imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    num_imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# -----------------------
# 7. 在训练集上用随机森林去除 outlier
# -----------------------
X_train_clean, y_train_clean, _, _, _ = remove_outliers_with_random_forest(
                                            X_train_imputed,
                                            y_train,
                                            contamination=OUTLIER_CONTAMINATION,
                                            random_state=RANDOM_STATE
                                        )

# 注意：测试集不做 outlier 删除，只做评估用

# -----------------------
# 8. 特征缩放（用“去 outlier 后的训练集”拟合 scaler）
# -----------------------
if DO_SCALE:
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test_imputed.columns,
        index=X_test_imputed.index
    )

# -----------------------
# 9. 重新拼接 target
# -----------------------
train_df = X_train_scaled.copy() if DO_SCALE else X_train_clean.copy()
train_df[TARGET_COL] = y_train_clean

test_df = X_test_scaled.copy() if DO_SCALE else X_test_imputed.copy()
test_df[TARGET_COL] = y_test  # 测试集保留原始 target

# -----------------------
# 10. 保存到文件
# -----------------------
train_path = '../data/housing_train_v1.csv'
test_path = '../data/housing_test_v1.csv'

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Saved train file to: {train_path}")
print(f"Saved test file to: {test_path}")

print("Train MEDV stats:\n", y_train.describe())
print("\nTest MEDV stats:\n", y_test.describe())


