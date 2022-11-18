---
title: 분류모델 평가지표(정밀도, 재현율), ROC곡선, AUC
author: yun
date: 2022-11-18
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, accuracy, roc, auc, classification, precision, recall]
---

# 목표
* Precision, Recall의 차이점이 뭘까?
* 암 진단을 위해서는 어떤 지표가 더 중요할까? 예시를 들어 생각해 보자.



# 키워드
- [x] 혼동행렬
- [x] Precision, Recall
- [x] 위음성(false negatives)
- [x] 위양성(false positives)
- [ ] 조화평균(harmonic mean)
- [ ] AUC  ROC curve


# 용어설명
## 정밀도(Precision)
* 바른 긍정적 예측의 수를 정량화하는 메트릭. 정확하게 예측된 긍정적 사례의 비율을 예측된 긍정적 사례의 총 수로 나눈 값으로 계산된다.
* **Precision = TruePositives / (TruePositives + FalsePositives)**
* The result is a value between `0.0` for no precision and `1.0` for full or perfect precision.
* precision is that it is not concerned with false negatives and it **minimizes false positives.**

### from sklearn.metrics import precision_score
```python
from sklearn.metrics import precision_score

# no precision
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
score = precision_score(y_true, y_pred)
print('No Precision: %.3f' % score)

# some false positives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Some False Positives: %.3f' % score)

# some false negatives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Some False Negatives: %.3f' % score)

# perfect precision
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Perfect Precision: %.3f' % score)
```
```python
== 결과값 ==
No Precision: 0.000
Some False Positives: 0.714
Some False Negatives: 1.000
Perfect Precision: 1.000
```
> 일부 가양성을 예측하는 예는 정밀도가 떨어지는 것을 보여주며 측정값이 가양성을 최소화하는 것과 관련이 있음을 강조한다. <br/>
> 일부 위음성을 예측하는 예는 측정이 위음성과 관련이 없음을 강조하면서 완벽한 정밀도를 보여준다.
<br/>



## 재현율(Recall)
* 만들 수 있었던 모든 긍정적인 예측에서 만들어진 올바른 긍정적인 예측의 수를 정량화하는 메트릭.
* 정확하게 예측된 긍정 예제의 비율을 예측할 수 있는 긍정 예제의 총 수로 나눈 값으로 계산된다.
* **Recall = TruePositives / (TruePositives + FalseNegatives)**
* The result is a value between `0.0` for no recall and `1.0` for full or perfect recall.
* recall is that it is not concerned with false positives and it **minimizes false negative.**

### from sklearn.metrics import recall_score
```python
from sklearn.metrics import recall_score
# no recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
score = recall_score(y_true, y_pred)
print('No Recall: %.3f' % score)

# some false positives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Some False Positives: %.3f' % score)

# some false negatives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Some False Negatives: %.3f' % score)

# perfect recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Perfect Recall: %.3f' % score)
```
```python
== 결과값 ==
No Recall: 0.000
Some False Positives: 1.000
Some False Negatives: 0.600
Perfect Recall: 1.000
```
> 일부 가양성(false positives)을 예측하는 예는 측정이 가양성과 관련이 없음을 강조하면서 완벽한 재현율을 보여준다. <br/>
> 일부 위음성(false negatives)을 예측하는 예는 측정값이 위음성을 최소화하는 것과 관련이 있음을 강조하면서 재현율 감소를 보여준다.
<br/>



## F-Measure
* **F-Measure = (2 * Precision * Recall) / (Precision + Recall)**
* This is the `harmonic mean` of the two fractions.
* The result is a value between `0.0` for the worst F-measure and `1.0` for a perfect F-measure.
* F-측정값은 두 측정값의 중요성이 균형을 이루고 있으며 우수한 정밀도와 우수한 재현율만이 좋은 F-측정값을 가져온다는 것.

### 최악의 경우
1. **모든 예가 완벽하게 잘못 예측**되면 정밀도와 재현율이 0이 되어 F-측정값이 0이 된다.
```python
# worst case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# no precision or recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('No Precision or Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```
```python
== 결과값 ==
No Precision or Recall: p=0.000, r=0.000, f=0.000
```
> 정밀도나 재현율이 없으면 최악의 F-측정값이 된다. `F-Measure=0`
<br/>

2.
```python
# another worst case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# no precision and recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('No Precision or Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```
```python
== 결과값 ==
No Precision or Recall: p=0.000, r=0.000, f=0.000
```
> 긍정적인 사례가 예측되지 않았으므로 정밀도와 재현율이 0으로 출력되고 결과적으로 F-측정값이 출력된다.
<br/>

### 최상의 경우
* 반대로 완벽한 예측은 완벽한 정밀도와 재현율, 그리고 완벽한 F-측정 결과를 보여준다. 
<br/>

```python
# best case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# perfect precision and recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('Perfect Precision and Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```
```python
== 결과값 ==
Perfect Precision and Recall: p=1.000, r=1.000, f=1.000
```




# 나만의 언어로 설명해보기
* TP(True Positive)
  : 양성(Positive)으로 예측했는데 그게 맞음!(True) 딩동댕!

* TN(True Negative)
  : 음성(Negative)으로 예측했는데 그게 맞음!(True) 딩동댕!
  
* FP(False Positive)
  : 양성(Positive)으로 예측했는데 그게 틀림!(False) 땡! (음성을 양성으로 예측)

* FN(False Negative)
  : 음성(Negative)으로 예측했는데 게 틀림!(False) 땡! (양성을 음성으로 예측)

* 임계점
  : 사람 마음의 깐깐도 점수같다. `0~1` 
  : ex) 깐깐도가 0.9인 사람이 맛있다고 한 음식은 정말 맛있다. 정밀도 상승, 재현율(민감도) 하락.
  : 반면에 깐깐도가 낮은 사람은 뭐든 맛있다고 함. 내가 먹어보면 맛 없을 수도 있음.. 정밀도 하락, 재현율(민감도) 상승.
  





# 레퍼런스
* [Introduction to the Confusion Matrix in Classification](https://youtu.be/wpp3VfzgNcI)
* [A Gentle Introduction to the Fbeta-Measure for Machine Learning](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)









# NOTE
|              | Positive Prediction | Negative Prediction|
|:-------------|:--------------------|:--------------------
|Positive Class | True Positive (TP)  | False Negative (FN)|
|Negative Class | False Positive (FP) | True Negative (TN)|








# Code

## sklearn.metrics.plot_confusion_matrix
## from sklearn.metrics import roc_curve

```
```
