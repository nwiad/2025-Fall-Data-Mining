from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



# baselines.py
import os
import numpy as np
from sklearn.model_selection import KFold

from rrl.utils import read_csv, DBEncoder

DATA_DIR = './dataset'

def prepare_y(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        # y: (N, C) -> (N,)
        return np.argmax(y, axis=1)
    else:
        return y.reshape(-1)


def get_numpy_data(dataset, k=0, val_ratio=0.05):
    """
    返回:
    (X_train, y_train), (X_val, y_val), (X_test, y_test), db_enc
    其中划分方式和 RRL 的 get_data_loader 保持一致:
      - 先 5 折交叉验证: 第 k 折做 test
      - 剩余 train 部分再切出一小部分做 validation
    """
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')

    # 读取原始 csv
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    # 编码 & 归一化
    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    # K 折划分（和你原来的代码一样）
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[k]

    X_train_full = X[train_index]
    y_train_full = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # 在训练集上再切出一部分作为验证集（比例和原代码一致 0.95/0.05）
    n_train = int(len(X_train_full) * (1 - val_ratio))
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_full[n_train:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), db_enc


# baselines.py 再往下写
class MyLogisticRegression:
    """
    纯 numpy 实现的二分类逻辑回归:
      - 手动初始化参数
      - 手动计算 sigmoid / loss / gradient
      - 使用 batch gradient descent 训练
    """
    def __init__(self, lr=0.1, n_iter=1000, l2=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.verbose = verbose
        self.w = None  # 包含 bias 的权重向量

    @staticmethod
    def _sigmoid(z):
        # 数值稳定一点
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X):
        # 在最左边加一列 1 作为偏置
        b = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([b, X], axis=1)

    def fit(self, X, y):
        X = self._add_bias(X)
        y = y.reshape(-1)

        n_samples, n_features = X.shape
        # 参数初始化为 0
        self.w = np.zeros(n_features, dtype=X.dtype)

        for i in range(self.n_iter):
            # 线性部分
            logits = X @ self.w
            # sigmoid 输出
            y_hat = self._sigmoid(logits)

            # 交叉熵损失的梯度
            error = y_hat - y  # (n,)
            grad = (X.T @ error) / n_samples  # (n_features,)

            # L2 正则
            if self.l2 > 0:
                grad += self.l2 * self.w

            # 梯度下降更新
            self.w -= self.lr * grad

            if self.verbose and (i % 100 == 0 or i == self.n_iter - 1):
                # 计算当前 loss 输出一下
                eps = 1e-8
                loss = -np.mean(
                    y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
                ) + 0.5 * self.l2 * np.sum(self.w ** 2)
                print(f"Iter {i:4d}  loss={loss:.5f}")

    def predict_proba(self, X):
        X = self._add_bias(X)
        logits = X @ self.w
        proba_pos = self._sigmoid(logits)
        # 返回 [P(y=0), P(y=1)] 两列
        proba = np.stack([1 - proba_pos, proba_pos], axis=1)
        return proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def run_manual_logreg(dataset, k=0, lr=0.1, n_iter=2000, l2=0.0):
    """
    使用手写 MyLogisticRegression 训练并在 test 集上评估。
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), db_enc = get_numpy_data(dataset, k)

    y_train = prepare_y(y_train)
    y_val = prepare_y(y_val)
    y_test = prepare_y(y_test)

    model = MyLogisticRegression(lr=lr, n_iter=n_iter, l2=l2, verbose=False)
    model.fit(X_train, y_train)


    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    # 主指标：F1
    f1 = f1_score(y_test, pred_test, average='macro')

    # 其他指标可选
    acc = accuracy_score(y_test, pred_test)
    try:
        auc = roc_auc_score(y_test, proba_test)
    except ValueError:
        auc = None

    print('==== Manual Logistic Regression (fold {}) ===='.format(k))
    print('Test F1: {:.4f}'.format(f1))
    print('Test Accuracy: {:.4f}'.format(acc))
    if auc is not None:
        print('Test AUC: {:.4f}'.format(auc))

    # 建议返回 F1 为第一个
    return f1, acc, auc



# 调包SVM
def run_svm(dataset, k=0):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), db_enc = get_numpy_data(dataset, k)

    # y_train = y_train.ravel()
    # y_test = y_test.ravel()
    y_train = prepare_y(y_train)
    y_test = prepare_y(y_test)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    f1 = f1_score(y_test, pred_test, average='macro')

    acc = accuracy_score(y_test, pred_test)
    auc = roc_auc_score(y_test, prob_test)

    print("==== SVM Result ====")
    print("F1  =", f1)
    print("Acc =", acc)
    print("AUC =", auc)

    return f1, acc, auc


if __name__ == '__main__':
    dataset_name = "bank_train_v1"
    kfold_id = 0

    print('\n\n================ Baseline Models ================')
    run_svm(dataset_name, kfold_id)
    run_manual_logreg(dataset_name, kfold_id)
