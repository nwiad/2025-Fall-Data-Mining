# 目标：在 housing 回归数据 上，
# 除了 RRL，再做 2 个 baseline 模型；
# 其中 至少 1 个要手动实现（不能调包算法）。

# 模型 A（手动实现）：从零实现一个 线性回归 Linear Regression（闭式解）
# 模型 B（可以调包）：用 sklearn 的 RandomForestRegressor

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===========================
# 1. 手动实现线性回归（闭式解）
# ===========================

class ManualLinearRegression:
    """
    一个从零实现的线性回归:
        y ≈ X w + b

    使用闭式解求参数：
        w* = (X^T X + λ I)^(-1) X^T y
    这里 λ 是 L2 正则项（可选）。

    我们不用 sklearn 的 LinearRegression，完全自己用 numpy 写。
    """

    def __init__(self, fit_intercept=True, l2_reg=0.0):
        self.fit_intercept = fit_intercept
        self.l2_reg = l2_reg
        self.coef_ = None  # 权重向量 w
        self.intercept_ = None  # 截距 b

    def _add_intercept(self, X):
        """如果需要截距，就在 X 左边加一列全 1。"""
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X, y):
        """
        拟合线性回归模型。

        参数:
            X: (n_samples, n_features)
            y: (n_samples,) 或 (n_samples, 1)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        X_design = self._add_intercept(X)  # 设计矩阵
        n_features_total = X_design.shape[1]

        # 计算 (X^T X + λ I) 和 X^T y
        xtx = X_design.T @ X_design
        if self.l2_reg > 0:
            # 对非截距项做 L2 正则（第 0 维是截距，对其不做正则）
            reg_matrix = np.eye(n_features_total)
            if self.fit_intercept:
                reg_matrix[0, 0] = 0.0
            xtx = xtx + self.l2_reg * reg_matrix

        xty = X_design.T @ y

        # 闭式解：w_all = (X^T X + λ I)^(-1) X^T y
        w_all = np.linalg.solve(xtx, xty)  # 形状 (n_features_total, 1)

        if self.fit_intercept:
            self.intercept_ = float(w_all[0, 0])
            self.coef_ = w_all[1:, 0]  # (n_features,)
        else:
            self.intercept_ = 0.0
            self.coef_ = w_all[:, 0]

    def predict(self, X):
        """
        预测：
            y_pred = X w + b
        """
        if self.coef_ is None:
            raise RuntimeError("模型还没有 fit 过！")

        X = np.asarray(X, dtype=np.float64)
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred


# ===========================
# 2. 评估函数
# ===========================

def evaluate_regression(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n====== {prefix} ======")
    print(f"[Test ] MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    return mse, mae, r2


# ===========================
# 3. 主流程
# ===========================

def main():
    # ---- 3.1 读取 train / test 数据 ----
    base_dir = os.path.join("..", "data")
    train_path = os.path.join(base_dir, "housing_train_v1.csv")
    test_path = os.path.join(base_dir, "housing_test_v1.csv")

    print("[Info] Train file:", train_path)
    print("[Info] Test  file:", test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "MEDV"

    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values

    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values

    print("[Info] Train shape:", X_train.shape, y_train.shape)
    print("[Info] Test  shape:", X_test.shape, y_test.shape)

    # ---- 3.2 特征缩放（可选，但一般对线性模型有好处）----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===============================
    # 4. 手动实现的线性回归 baseline
    # ===============================

    manual_lr = ManualLinearRegression(
        fit_intercept=True,
        l2_reg=0.0,   # 可以改成 1e-3 看看效果
    )
    manual_lr.fit(X_train_scaled, y_train)

    # 训练集上的表现（可以看看过拟合情况）
    y_train_pred_lr = manual_lr.predict(X_train_scaled)
    train_mse, train_mae, train_r2 = evaluate_regression(
        y_train, y_train_pred_lr,
        prefix="ManualLinearRegression (Train)"
    )

    # 测试集上的表现（用于与 RRL 对比）
    y_test_pred_lr = manual_lr.predict(X_test_scaled)
    test_mse, test_mae, test_r2 = evaluate_regression(
        y_test, y_test_pred_lr,
        prefix="ManualLinearRegression (Test)"
    )

    # ===============================
    # 5. 调包的 Random Forest baseline
    # ===============================

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)  # RF 对特征缩放不敏感，这里用未缩放的特征

    y_train_pred_rf = rf.predict(X_train)
    evaluate_regression(y_train, y_train_pred_rf, prefix="RandomForestRegressor (Train)")

    y_test_pred_rf = rf.predict(X_test)
    evaluate_regression(y_test, y_test_pred_rf, prefix="RandomForestRegressor (Test)")


if __name__ == "__main__":
    main()
