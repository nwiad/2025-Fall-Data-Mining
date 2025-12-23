# Datamining


## Task1 bank



## Task2 housing

### 2.1 Baseline
目标：验证数据处理没问题，
1. 普通线性回归能跑到 0.57 R²
    - 数据格式 OK
    - 预处理 OK
    - label 正常
2. 随机森林模型
    - 数据有明显非线性结构
    - 依然是可学习问题

```
$ python scripts/baseline_housing_regression.py 
[Info] 原始数据形状: (506, 14)
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622    3  222     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622    3  222     18.7  396.90    NaN  36.2
[Split] Train shape: (404, 13) (404,)
[Split] Test  shape: (102, 13) (102,)
[Outlier] 原始训练集样本数: 404, 去除 outlier 后样本数: 383
[Outlier] 残差阈值 = 2.6204
/data-share/miniconda3/envs/lijiaqi_pytorch/lib/python3.10/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(

====== LinearRegression ======
[Train] MSE=15.0043, MAE=2.7709, R2=0.8092
[Test ] MSE=36.1924, MAE=3.6226, R2=0.5693

====== Ridge(alpha=1.0) ======
[Train] MSE=15.0048, MAE=2.7697, R2=0.8092
[Test ] MSE=36.2342, MAE=3.6226, R2=0.5688

====== RandomForestRegressor ======
[Train] MSE=0.7373, MAE=0.6699, R2=0.9906
[Test ] MSE=29.9895, MAE=3.1595, R2=0.6431

```

### 2.2 
目前R2还是不正常


[Test Regression] MSE=426.6449, MAE=19.0196, R2=-5.5610
[Structure] #Edges = 48, Log(#Edges+eps) = 3.8712