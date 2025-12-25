# Datamining


## Task1 bank



## Task2 Housing 回归任务

### 2.1 Baseline

1. 手写 Linear Regression（ManualLinearRegression） （满足“不能调包”的要求）
2. 调包 RandomForestRegressor（sklearn）

```
$ python baseline_housing_manual_lr_and_rf.py 
[Info] Train file: ../data/housing_train_v1.csv
[Info] Test  file: ../data/housing_test_v1.csv
[Info] Train shape: (383, 13) (383,)
[Info] Test  shape: (102, 13) (102,)

====== ManualLinearRegression (Train) ======
[Test ] MSE=15.0043, MAE=2.7709, R2=0.8092

====== ManualLinearRegression (Test) ======
[Test ] MSE=36.1924, MAE=3.6226, R2=0.5693

====== RandomForestRegressor (Train) ======
[Test ] MSE=0.7390, MAE=0.6708, R2=0.9906

====== RandomForestRegressor (Test) ======
[Test ] MSE=29.9875, MAE=3.1589, R2=0.6432

```

### 2.2 调了一些参数，写在脚本里面

目前最好的参数是

```
[Test Regression] MSE=11.6421, MAE=2.7883, R2=0.8210
[Structure] #Edges = 549, Log(#Edges+eps) = 6.3081
python3 experiment.py \
    -d housing_train_v1 \
    -bs 4 \
    -s 5@64 \
    -e 100 \
    -lrde 200 \
    -lr 0.005 \
    -ki 0 \
    -i ${WORLD_SIZE_2} \
    -wd 0.0001 \
    --task_type "regression"\
    --print_rule \
    --master_address "127.0.0.1" \
    --master_port 12345 \
    --save_best \
    --use_not
```


### 2.3 实验结果分析
要求：不同模型的优劣和背后的原理，适用什么样的任务类型。可以从以下角度分析
1. 选择合适的评价指标分析
2. 可解释性：模型解释的可理解性，以及这些解释对模型真实决策过程的忠实度
3. 复杂度：RRL以及基于决策树的模型可以计算连边数量之和的自然对数，rrl中有log（edge）指标

# 模型对比与分析

##  评价指标选择与结果分析

本任务属于**回归问题**，目标变量 MEDV 为连续数值，因此选择以下指标评价模型效果：

| 指标      | 含义                | 解释       |
| ------- | ----------------- | -------- |
| **MSE** | 均方误差，对大误差更加敏感     | 越小越好     |
| **MAE** | 平均绝对误差，鲁棒性较强      | 越小越好     |
| **R²**  | 拟合优度，刻画模型解释数据方差能力 | 越接近 1 越好 |

R² 尤其重要：

* R²=1 表示完美拟合
* R²=0 表示模型还不如直接输出均值
* R²<0 表示模型比“随机猜”还差

---

### 实验结果

#### （1）手动实现线性回归

| Train       | Test      |
| ----------- | --------- |
| MSE = 15.00 | **36.19** |
| MAE = 2.77  | **3.62**  |
| R² = 0.81   | **0.57**  |

现象：

* 训练集表现较好，说明数据中确实存在一定线性关系
* 测试集 R²≈0.57，效果一般
* 说明线性模型表达能力有限，无法充分捕捉 Housing 数据的非线性结构

---

#### （2）调用Random Forest

| Train      | Test      |
| ---------- | --------- |
| MSE = 0.74 | **29.99** |
| MAE = 0.67 | **3.16**  |
| R² = 0.99  | **0.64**  |

现象：

* Train R²≈0.99，几乎完美拟合 → 明显**过拟合**
* Test R²≈0.64，优于线性模型，但是提升有限
* 样本量较小（506），噪声较大时随机森林容易捕捉伪规律

---

#### （3）RRL（Rule-based Representation Learning）

| Metric          | Test 结果    |
| --------------- | ---------- |
| **MSE**         | **11.64**  |
| **MAE**         | **2.79**   |
| **R²**          | **0.8210** |
| **log(#Edges)** | **6.31**   |

现象：

* **Test R² 达到 0.82，远高于两种 baseline**
* MAE、MSE 也均最优
* 说明 RRL 既具备较强的非线性建模能力，又没有过度拟合

---

## 2️可解释性分析

可解释性从两个维度考虑：

* **是否可以给出人类可理解的解释**
* **解释是否忠实于模型真实决策机制（Fidelity）**

---

### （1）Linear Regression —— 最简单透明的模型

解释方式：

* 每个特征对应一个权重
* 权重大 → 影响房价越大
* 权重正 / 负 → 房价上升 / 下降

例如：

* RM（房间数）权重大 → 房子越大越贵
* LSTAT（低收入人口比例）负权重大 → 区域越贫困房价越低
* PTRATIO（师生比）负相关 → 教育资源差 → 房价下降

结论：

* **解释性极好**
* **解释 = 模型本身，忠实度最高**
* 但模型只能表达线性关系 → 限制性能

---

### （2）Random Forest 

理论上：

* 单棵决策树具有 IF-THEN 规则解释
* 但 Random Forest：

  * 数百棵树
  * 每棵树几十~上百个 split
  * 难以整体理解

常见解释方式：

* Feature Importance
* Partial Dependence Plot

问题：

* Feature Importance 只是统计量
* 并不能真实反映模型如何逐步做出决策
* 属于**弱可解释**

---

### （3）RRL —— “结构性、规则级可解释”

RRL 输出清晰规则：

```
IF condition1 AND condition2 THEN predict ...
```

优势：

* 规则是明确的人类语言形式
* 这些规则**直接参与预测**
* 而不是像 RF 那样只是后验分析
* 因此：

  * 解释性强
  * 忠实度高
  * 可读性好

结论：

RRL 在**可解释性 + 表达能力**之间取得了一个很好平衡。

---

## 模型复杂度分析

复杂度影响：

* 过低 → 表达能力不够（如线性模型）
* 过高 → 过拟合风险大（如 Random Forest）
* 因此需要一个“结构化复杂度”指标

---

### （1）Linear Regression

* 参数数量 ≈ 特征数量
* Housing 约 13 个特征
* 结构极简
* 但表达能力弱

---

### （2）Random Forest

* 真实结构非常巨大
* 难以统一量化
* 难以解释其真实复杂度

---

### （3）RRL 复杂度（Edge + log Edge）

RRL 的复杂度由图结构决定：

* Edge = 规则图中的连边
* Edge 越多 → 结构越复杂
* 取 log(#Edges)：

  * 稳定尺度
  * 便于比较
  * 避免数量级过大

本实验结果：

```
#Edges = 549
log(#Edges) = 6.31
```

📌 说明：

* RRL 的结构具有一定复杂度，足以表达非线性模式
* 但复杂度是**可度量、可控制**的
* 相比 Random Forest 混乱的结构，RRL 更“优雅”

---

## 最终总结

* **Linear Regression**

  * 优点：解释简单清晰、忠实度高
  * 缺点：表达能力弱，Test R²≈0.57
  * 适合：线性任务 + 高解释要求

---

* **Random Forest**

  * 优点：表达能力强
  * 缺点：过拟合明显，可解释性弱
  * Test R²≈0.64
  * 适合：不要求解释，只追求效果

---

* **RRL（本实验最佳）**

  * Test R²≈0.82，显著优于 baseline
  * MAE/MSE 最优
  * 规则级可解释
  * 复杂度可度量（log(#Edges)=6.31）
  * **同时兼顾性能与可解释性**