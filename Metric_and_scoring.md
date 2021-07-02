本文档主要积累各种模型评估方法，包括公式解析、使用sklearn实现中使用方法、优缺点等。


```python
from sklearn import metrics 
```

# 分类模型


```python

```

# 回归模型

## Explained variance score

Explained variance score也叫解释方差，解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。公式如下所示：
<center>$explained\_variance(y, \hat{y})=1-\frac{Var(y-\hat{y})}{Var(y)}$</center>
使用sklearn模块实现上述算法的实例如下所示：


```python
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
metrics.explained_variance_score(y_true, y_pred)
```




    0.9571734475374732



当使用sklearn中的网格搜索(GridSearchCV()）、交叉验证(cross_val_score())等函数时，表示Explained variance score的字符串为**'explained_variance'**。

## Max error

max_error函数计算真实数据与预测数据之间的最大残差，这是一个衡量预测值和真实值之间最坏情况误差的指标。在完美拟合的单输出回归模型中，max_error为0，尽管这在现实世界中极不可能发生，但该指标显示了模型在拟合时的误差程度，公式如下所示。
<center>$Max Error(y, \hat{y})=max(|y_i-\hat{y}_i|)$</center>
使用sklearn模块实现上述算法如下所示，当使用sklearn中的网格搜索(GridSearchCV()）、交叉验证(cross_val_score())等函数时，表示Max Error的字符串为'max_error'。


```python
metrics.max_error(y_true, y_pred)
```




    1.0



## Mean absolute error

Mean absolute error函数计算真实数据与预测数据的平均绝对损失，一个对用与绝对误差的期望的风险度量($l1$范数)，公式如下所示：
<center>$MAE(y, \hat{y})=\frac{1}{n_{s}}\sum\limits_{i=1}^{n_{s}-1}|y_i-\hat{y}_i|$</center>
上式中，$y_i$表示真实样本，$\hat{y}_i$表示预测样本，$n_s$表示样本数量。
使用sklearn模块实现上述算法如下所示，当使用sklearn中的网格搜索(GridSearchCV()）、交叉验证(cross_val_score())等函数时，表示Mean absolute error的字符串为'neg_mean_absolute_error'。


```python
metrics.mean_absolute_error(y_true, y_pred)
```




    0.5



## Mean squared error

Mean squared error函数计算真实数据与预测数据的均方误差，这是一个风险度量，对应于平方（二次）误差或损失的预期值, 公式如下所示：
<center>$MSE(y, \hat{y})=\frac{1}{n_{s}}\sum\limits_{i=1}^{n_{s}-1}(y_i-\hat{y}_i)^2$</center>
上式中，$y_i$表示真实样本，$\hat{y}_i$表示预测样本，$n_s$表示样本数量。
使用sklearn模块实现上述算法如下所示，当使用sklearn中的网格搜索(GridSearchCV()）、交叉验证(cross_val_score())等函数时，表示Mean squared error的字符串为'neg_mean_squared_error'。


```python
metrics.mean_squared_error(y_true, y_pred)
```




    0.375



## Mean squared logarithmic error


```python

```

## Mean absolute percentage error

<center>$MAPE(y, \hat{y})=\frac{1}{n_{s}}\sum\limits_{i=0}^{n_{s}-1}\frac{|y_i-\hat{y_i}|}{max(\epsilon, |y_i|)}$</center>



## Median absolute error

<center>$MedAE(y,\hat{y})=median(|y_1-\hat{y_1}|,...,|y_n-\hat{y_n}|)$</center>


```python

```

## $R^2$score

<center>$R^2(y,\hat{y})=1-\frac{\sum\limits_{i=0}^{n_{s}-1}(y_i-\hat{y_i})^2}{\sum\limits_{i=0}^{n_{s}-1}(y_i-\bar{y_i})^2}$


```python

```

# 聚类模型


```python

```
