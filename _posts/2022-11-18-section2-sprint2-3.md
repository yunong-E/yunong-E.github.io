---
title: 분류모델 평가지표(정밀도, 재현율), ROC곡선, AUC
author: yun
date: 2022-11-18
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, accuracy, roc, auc, classification, precision, recall]
---

# 키워드
- [ ] 혼동행렬
- [ ] Precision, Recall


# 용어설명
* 정밀도
  : 바른 긍정적 예측의 수를 정량화하는 메트릭. 정확하게 예측된 긍정적 사례의 비율을 예측된 긍정적 사례의 총 수로 나눈 값으로 계산된다.
  : 정밀도 = TruePositives / (TruePositives + FalsePositives)
  : 결과는 정밀도가 없는 경우 0.0, 전체 또는 완벽한 정밀도인 경우 1.0 사이의 값

# 나만의 언어로 설명해보기


# 레퍼런스
* [Introduction to the Confusion Matrix in Classification](https://youtu.be/wpp3VfzgNcI)
* [A Gentle Introduction to the Fbeta-Measure for Machine Learning](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)



# NOTE
               | Positive Prediction | Negative Prediction
Positive Class | True Positive (TP)  | False Negative (FN)
Negative Class | False Positive (FP) | True Negative (TN)


정밀도 = TruePositives / (TruePositives + FalsePositives)




# Code
## from sklearn.metrics import precision_score
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
결과값:
```python
No Precision: 0.000
Some False Positives: 0.714
Some False Negatives: 1.000
Perfect Precision: 1.000
```
> 일부 가양성을 예측하는 예는 정밀도가 떨어지는 것을 보여주며 측정값이 가양성을 최소화하는 것과 관련이 있음을 강조한다.
> 일부 위음성을 예측하는 예는 측정이 위음성과 관련이 없음을 강조하면서 완벽한 정밀도를 보여준다.
