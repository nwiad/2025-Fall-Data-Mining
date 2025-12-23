import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================
# 0. 参数配置
# ============================
# 原始数据路径（改成你自己的路径）
PATH = 'data/HousingData.csv'

# 目标列名
TARGET_COL = 'MEDV'

# train / test 划分
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 是否对特征做标准化
DO_SCALE = True

# 是否用随机森林做一次简单的 outlier 删除（只影响训练集）
REMOVE_OUTLIERS = True
OUTLIER_CONTAMINATION = 0.05  # 删除残差最大的 5% 样本


def remove_outliers_with_random_forest(X, y, contamination=0.05, random_state=42):
    """
    使用 RandomForestRegressor 根据残差大小删除训练集中的 outlier。
    只在训练集上使用，避免信息泄漏。
    """
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)

    y_pred = rf.predict(X)
    residuals = np.abs(y - y_pred)

    # 取残差分布的高分位数作为阈值
    threshold = np.quantile(residuals, 1 - contamination)
    mask = residuals <= threshold

    X_clean = X[mask]
    y_clean = y[mask]

    print(f"[Outlier] 原始训练集样本数: {len(y)}, 去除 outlier 后样本数: {len(y_clean)}")
    print(f"[Outlier] 残差阈值 = {threshold:.4f}")

    return X_clean, y_clean


def main():
    # ============================
    # 1. 读取数据
    # ============================
    df = pd.read_csv(PATH)
    print("[Info] 原始数据形状:", df.shape)
    print(df.head())

    # ============================
    # 2. 拆分特征 & 目标
    # ============================
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # ============================
    # 3. 分层划分 train / test（根据 MEDV 分箱）
    # ============================
    y_bins = pd.cut(y, bins=10, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_bins
    )

    print("[Split] Train shape:", X_train.shape, y_train.shape)
    print("[Split] Test  shape:", X_test.shape, y_test.shape)

    # ============================
    # 4. 缺失值填充（median）
    # ============================
    num_imputer = SimpleImputer(strategy='median')

    X_train_imputed = pd.DataFrame(
        num_imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        num_imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # ============================
    # 5. 可选：删除训练集 outlier
    # ============================
    if REMOVE_OUTLIERS:
        X_train_used, y_train_used = remove_outliers_with_random_forest(
            X_train_imputed.values,
            y_train,
            contamination=OUTLIER_CONTAMINATION,
            random_state=RANDOM_STATE,
        )
        # 注意：X_train_used 现在是 ndarray
        X_train_used = pd.DataFrame(
            X_train_used,
            columns=X_train_imputed.columns,
        )
    else:
        X_train_used = X_train_imputed.copy()
        y_train_used = y_train.copy()

    # ============================
    # 6. 可选：特征标准化
    # ============================
    if DO_SCALE:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_used)
        X_test_scaled = scaler.transform(X_test_imputed.values)

        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train_used.values
        X_test_final = X_test_imputed.values

    # ============================
    # 7. 定义一个小工具函数打印指标
    # ============================
    def eval_model(name, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        mse_tr = mean_squared_error(y_tr, y_pred_tr)
        mae_tr = mean_absolute_error(y_tr, y_pred_tr)
        r2_tr = r2_score(y_tr, y_pred_tr)

        mse_te = mean_squared_error(y_te, y_pred_te)
        mae_te = mean_absolute_error(y_te, y_pred_te)
        r2_te = r2_score(y_te, y_pred_te)

        print(f"\n====== {name} ======")
        print(f"[Train] MSE={mse_tr:.4f}, MAE={mae_tr:.4f}, R2={r2_tr:.4f}")
        print(f"[Test ] MSE={mse_te:.4f}, MAE={mae_te:.4f}, R2={r2_te:.4f}")

    # ============================
    # 8. 跑几个 baseline 模型
    # ============================

    # 线性回归
    linreg = LinearRegression()
    eval_model("LinearRegression", linreg, X_train_final, y_train_used, X_test_final, y_test)

    # L2 正则的 Ridge 回归
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    eval_model("Ridge(alpha=1.0)", ridge, X_train_final, y_train_used, X_test_final, y_test)

    # 随机森林回归
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    eval_model("RandomForestRegressor", rf, X_train_final, y_train_used, X_test_final, y_test)


if __name__ == "__main__":
    main()
