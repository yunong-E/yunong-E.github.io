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

<br/>

# 키워드
- [x] [혼동행렬](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)
- [x] Precision, Recall
- [x] 위음성(false negatives)
- [x] 위양성(false positives)
- [ ] 조화평균(harmonic mean)
- [x] ROC curve
- [x] AUC

<br/>

# 용어설명
## 정밀도(Precision)
* 바른 긍정적 예측의 수를 정량화하는 메트릭. 정확하게 예측된 긍정적 사례의 비율을 예측된 긍정적 사례의 총 수로 나눈 값으로 계산된다.
* **Precision = TruePositives / (TruePositives + FalsePositives)**
* The result is a value between `0.0` for no precision and `1.0` for full or perfect precision.
* precision is that it is not concerned with false negatives and it **minimizes false positives.**
<br/>

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
> 일부 `false positives`를 예측하는 예는 정밀도가 떨어지는 것을 보여주며 측정값이 `false positives`를 최소화하는 것과 관련이 있음을 강조한다. <br/>
> 일부 `false negatives`를 예측하는 예는 측정이 `false negatives`와 관련이 없음을 강조하면서 완벽한 정밀도를 보여준다.

<br/>

## 재현율(Recall)
* 만들 수 있었던 모든 긍정적인 예측에서 만들어진 올바른 긍정적인 예측의 수를 정량화하는 메트릭.
* 정확하게 예측된 긍정 예제의 비율을 예측할 수 있는 긍정 예제의 총 수로 나눈 값으로 계산된다.
* **Recall = TruePositives / (TruePositives + FalseNegatives)**
* The result is a value between `0.0` for no recall and `1.0` for full or perfect recall.
* recall is that it is not concerned with false positives and it **minimizes false negative.**
<br/>

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
> 일부 `false positives`를 예측하는 예는 측정이 `false positives`와 관련이 없음을 강조하면서 완벽한 재현율을 보여준다. <br/>
> 일부 `false negatives`를 예측하는 예는 측정값이 `false negatives` 최소화하는 것과 관련이 있음을 강조하면서 재현율 감소를 보여준다.
<br/>


## F-Measure
* **F-Measure = (2 * Precision * Recall) / (Precision + Recall)**
* This is the `harmonic mean` of the two fractions.
* The result is a value between `0.0` for the worst F-measure and `1.0` for a perfect F-measure.
* F-측정값은 두 측정값의 중요성이 균형을 이루고 있으며 우수한 정밀도와 우수한 재현율만이 좋은 F-측정값을 가져온다는 것.
<br/>

### 최악의 경우
1. **모든 예가 완벽하게 잘못 예측**되면 정밀도와 재현율이 0이 되어 F-측정값이 0이 된다.
<br/>

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
<br/>

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

<br/>

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
  
<br/>

# 레퍼런스
* [Introduction to the Confusion Matrix in Classification](https://youtu.be/wpp3VfzgNcI)
* [A Gentle Introduction to the Fbeta-Measure for Machine Learning](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)

<br/>

# NOTE
|              | Positive Prediction | Negative Prediction |
|:-------------|:--------------------|:--------------------|
|Positive Class | True Positive (TP)  | False Negative (FN)|
|Negative Class | False Positive (FP) | True Negative (TN) |
<br/>

* 정확도(Accuracy)는 전체 범주를 모두 바르게 맞춘 경우를 전체 수로 나눈 값: $$\large \frac{TP + TN}{Total}$$
* 정밀도(Precision)는 **Positive로 예측**한 경우 중 올바르게 Positive를 맞춘 비율: $$\large \frac{TP}{TP + FP}$$
* 재현율(Recall, Sensitivity)은 **실제 Positive**인 것 중 올바르게 Positive를 맞춘 것의 비율: $$\large \frac{TP}{TP + FN}$$
* F1점수(F1 score)는 정밀도와 재현율의 `조화평균(harmonic mean`): $$ 2\cdot\large\frac{precision\cdot recall}{precision + recall}$$

<br/>

# Code
## sklearn.metrics.plot_confusion_matrix
* sklearn.metrics.plot_confusion_matrix 는 가로, 세로축이 바뀔 수 있으니 외부자료를 참조할 때 주의할 것.
* 색깔을 통해서 샘플의 비율(?)을 알 수 있다.
<br/>

```python
# 렉처노트 n223 예시

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
pcm = plot_confusion_matrix(pipe, X_val, y_val,
                            cmap=plt.cm.Blues,
                            ax=ax)
plt.title(f'Confusion matrix, n = {len(y_val)}', fontsize=15)
plt.show()
```
![confusion_matrix](https://github.com/yunong-E/utterances_only/blob/main/assets/img/confusion_matrix.png)
<br/>

### plot_confusion_matrix
* plot_confusion_matrix 에서 테이블 데이터만 가져와 총 정확도(accuracy)를 구할 수 있다.
<br/>

```python
# 렉처노트 n223 예시

# 혼동행렬에서 2차원 행렬값 중요. 
cm = pcm.confusion_matrix
cm

# TP예측 (예측: functional, 실제: functional)
cm[1][1]

# TP + TN
correct_predictions = np.diag(cm).sum()
correct_predictions

# 총 예측한 수
total_predictions = cm.sum()
total_predictions


# 분류 정확도(classification accuracy) 1과 2의 결과값은 같다.
# 1 
correct_predictions/total_predictions

# 2
accuracy_score(y_val, y_pred))
```

<br/>

## sklearn.metrics.classification_report
* sklearn.metrics.classification_report를 사용하면 정밀도, 재현율을 확인할 수 있다.
<br/>

```python
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
```
```python
==결과==
              precision    recall  f1-score   support

           0       0.76      0.80      0.78      7680
           1       0.75      0.70      0.72      6372

    accuracy                           0.75     14052
   macro avg       0.75      0.75      0.75     14052
weighted avg       0.75      0.75      0.75     14052
```

<br/>

## from sklearn.metrics import roc_curve
* 모든 임계값을 한 눈에 보고 모델을 평가할 수 있는 방법? `ROC curve` 사용!
* ***이진분류문제***에서 사용할 수 있다. 다중분류문제에서는 각 클래스를 이진클래스 분류문제로 변환(One Vs All)하여 구할 수 있다.
  * 3-class(A, B, C) 문제 -> A vs (B,C), B vs (A,C), C vs (A,B) 로 나누어 수행
* $$재현율 = 민감도 = TPR = 1-FNR$$
* $$Fall-out = FPR = 1-TNR(특이도)$$
* [재현율은 최대화 하고 위양성률은 최소화 하는 임계값이 최적의 임계값 (서로 Trade-off 관계)](http://www.navan.name/roc/)
* AUC 는 ROC curve의 아래 면적을 말한다.
<br/>

```python
# 렉쳐노트 n223 예시

# ROC curve 그리기
plt.scatter(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('FPR(Fall-out)')
plt.ylabel('TPR(Recall)');
```
![roc_curve](https://github.com/yunong-E/utterances_only/blob/main/assets/img/roc_curve.png)
<br/>

```python
# 렉쳐노트 n223 예시

# 최적의 threshold(임계값) 찾기.
# threshold 최대값의 인덱스, np.argmax()
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print('idx:', optimal_idx, ', threshold:', optimal_threshold)
```
```python
==결과==
# 임계값 0.5(default)와 비교했을때 큰 차이는 나지 않지만, 문제와 상황에 따라서 더 좋은 결과를 낼 수도 있다.
idx: 257 , threshold: 0.4633333333333334
```

<br/>

## from sklearn.metrics import roc_auc_score
* AUC를 계산할 수 있다.
* AUC는 값이 클 수록 좋다. `0.5~1`
<br/>

```python
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_val, y_pred_proba)
auc_score
```
