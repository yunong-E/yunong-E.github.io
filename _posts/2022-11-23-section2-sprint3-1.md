---
title: 정보누수(Leakage), 교차검증
author: yun
date: 2022-11-23
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, k-fold, loocv]
---

# 목표
* 교차검증을 사용하는 이유가 무엇일까?
  1. 내가 가지고 있는 데이터셋에 적합한 모델을 찾기 위해?
  2. 어떤 하이퍼파라미터를 사용할 것인지 알기 위해?


* 왜 accuracy 만 사용하면 모델 성능에 대해 잘못된 판단을 내릴 수 있을까?

<br/>

# 키워드
- [x] 정보누수(Data Leakage)
- [x] 불균형클래스
- [ ] 오버샘플링
- [ ] 언더샘플링
- [ ] TransformedTargetRegressor

<br/>

# NOTE
* 지도학습을 통해 문제를 풀고자 하면 `타겟`을 명확하게 해야한다.
* 지도학습은 회귀, 분류문제로 풀 수 있다.
* 회귀문제로 보이는 타겟특성도 분류문제로 바꿀 수 있고, 그 반대의 경우도 있다.
* 정보의 누수는 과적합을 일이크기 실제 테스트 데이터에서 성능이 급격하게 떨어지는 결과를 초래한다.
* 분류문제에서 타겟 클래스비율이 `70% 이상` 차이날 경우에는 ***정확도만 사용하면*** 판단을 정확히 할 수 없다.
  * ***정밀도, 재현율, ROC curve, AUC 등***을 **같이 사용**해야한다.
* 회귀분석에서는 ***타겟의 분포***를 주의깊게 살펴야 한다. 타겟 분포가 **비대칭 형태**인지 확인 필수!
  * 선형 회귀 모델은
    a. 일반적으로 특성과 타겟간에 `선형관계`를 가정한다. <br/>
    b. 그리고 특성 변수들과 타겟변수의 분포가 `정규분포` 형태일때 좋은 성능을 보인다. <br/>
    c. 특히 타겟변수가 ***왜곡된 형태의 분포(skewed)***일 경우 예측 성능에 부정적인 영향을 미친다. <br/>
    <img width="608" alt="스크린샷 2022-11-29 오전 12 11 28" src="https://user-images.githubusercontent.com/81222323/204314915-80fce41f-81a0-4954-93f8-50644d07bb4a.png">
    
    
    d. 이상치가 존재할 경우 ***이상치 제거 필수.***
    e. 타겟이 `right-skewed` 상태라면 로그변환(Log-Transform)을 사용. 비대칭 분포형태를 정규분포형태로 변환해줌. (무조건은 아님!)
    
    
<img width="599" alt="스크린샷 2022-11-29 오전 12 12 26" src="https://user-images.githubusercontent.com/81222323/204313840-d4068d05-954c-433a-b9de-5b95acad5c03.png">

<br/>

## 참고자로
1. [How (and why) to create a good validation set](https://www.fast.ai/posts/2017-11-13-validation-sets.html)
2. [Handling Imbalanced Datasets in Deep Learning](https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758)

<br/>

## 타겟설정
* 문제는 무엇이고 목적은 무엇인지? -> `Target`을 결정 -> `회귀/분류` 어떤 것으로 풀 것인지 선택 -> 문제에 적합한 `평가지표` 선택.
* 기승전결 중 `기`와 `결`도 무엇보다 중요합니다. 어필할 수 있어야 합니다.

<br/>

## 데이터과학자 실무 프로세스
1. 비즈니스 문제
    - 실무자들과 대화를 통해 문제를 발견
2. 데이터 문제
    - 문제와 관련된 데이터를 발견
3. 데이터 문제 해결
    - 데이터 처리, 시각화
    - 머신러닝/통계
4. 비즈니스 문제 해결
    - 데이터 문제 해결을 통해 실무자들과 함께 해결

<br/>

## 정보누수(Data Leakage, 데이터 리키지)
* 데이터 리키지가 발생하는 두 가지 경로
  1. 타겟 리키지.<br/>
    a. 타켓정보가 학습에 포함이 되어있음
      * rating 이라는 타겟정보가 학습에 포함이 되어있었음. (렉쳐노트 n231)
      * 수료여부예측 모델 -> 조기하차 feature = 수료 못함.


    b. 학습 데이터가 예측 시 ***못 쓰는 feature를 반영***하는 경우
      * 암진단 여부를 확인해야 하는데 항암치료 여부 feature가 이미 암을 진단받았다는 사실(암진단 이후)를 명시함.  

<br/>

  2. Train - Test contamination
    * 전처리

<br/>

## 평가지표
* 회귀: mae, $$r^2$$, mse, rmse ...

* 분류: f1-score, accuracy, precision, recall, auc ...

* 불균형 타겟
  * 타겟이 불균형할 때 정확도만 사용하면 안되는 이유?
    * 예측에 대한 catch를 제대로 하지 못함. 정확도가 0.99로 나올 수 있음.
    * 희귀값에 대한 예측을 제대로 하지 못함.


참고자로
1. [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
2. [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
3. [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

<br/>

## 타겟 분포 불균형 (불균형 클래스)
* 분류 (바이너리, 이산형, 0-1)
  * **가중치**
    * 가중치를 줘서 1:99 비율의 데이터를 5:5로 ***보이게하여*** 비율 1의 데이터를 잘 catch 할 수 있게 함.
    * `class_weight = 'balanced'`
  * **샘플링**
    * 오버샘플링: 1:99 -> 99:99 ('현실에 없는 데이터를 사용하기 때문에 로버스트하지 않다' 라는 평가가 있다.)
    * 언더샘플링  1:99 -> 1:1

* 회귀
  * 기사의 댓글 수, 연봉 등.. 현실세계에서는 `Positively skewed` 형태의 분포가 많다. 로그변환을 해주자.
  * `TransformedTargetRegressor`
  * 로그변환을 한다고 ***무조건 정규분포 형태를 띄는 것은 아니다***. 왜도에 따라서 다름.
  * 리니어모델의 경우 왜도가 심하면 `log transform` 하면 좋다.

<br/>

> Negatively Skewed의 경우는 어떻게 변환을 하면 될까? 
<br/>

-> 로그 변환

<br/>

# Code
## 1. 결측치가 있는 특성 확인하기 `for문`, `.any()` 사용
```python
# 렉쳐노트 n231 예시
[(x, df[x].isnull().sum()) for x in df.columns if df[x].isnull().any()]

==결과==
# 형태만 확인할 것.
[('beanType', 1), ('broadOrigin', 1)]
```

<br/>

## 2. `describe(include='all').T`
```python
df.describe(include='all').T
```
> describe() 메서드로 인해 제외되었던 누락데이터(NaN)가 표시된다.


참고자료: [pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)

<br/>

## 3. numpy.logical_or
* 요소별로 요소별로 x1 OR x2의 진리값을 계산한다. (x1, x2는 인수)
```python
# 1
np.logical_or(True, False)

==결과==
True


# 2
np.logical_or([True, False], [False, False])

==결과==
array([ True, False])
```

<br/>


## 4. 데이터 세트 나누기 (train_test_split)
```python
from sklearn.model_selection import train_test_split

# 훈련, 검증세트로만 나눔.
# test_size에 0.2값을 넣어주면 검증(테스트)데이터 20:훈련데이터 80 으로 분배됨.
train, val = train_test_split(df, test_size=0.2, random_state=2)
```

<br/>

## 5. 결정트리(분류) 그리기
* `filled`: 기본값 `False`, `True`로 설정하면 노드를 페인트하여 분류를 위한 다수 클래스, 회귀를 위한 값의 극단 또는 다중 출력을 위한 노드 순도를 나타낸다.
* `proportion`: 기본값 `False`, `True`로 설정하면 '값' 혹은 '샘플'의 표시를 각각 비율 및 백분율로 변경.


```python
import graphviz
from sklearn.tree import export_graphviz

# 결정트리(분류)
tree = pipe.named_steps['decisiontreeclassifier']

dot_data = export_graphviz(
    tree,
    feature_names=X_train.columns, 
    class_names=y_train.unique().astype(str), 
    filled=True, 
    proportion=True
)

graphviz.Source(dot_data)
```


<img width="375" alt="스크린샷 2022-11-28 오후 11 28 35" src="https://user-images.githubusercontent.com/81222323/204310757-6fff9fe2-1e5c-415f-8027-604702108110.png">

<br/>

## 6. 혼동행렬(Confusion matrix) 시각화
```python
# 렉쳐노트 n231 예시
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# 참고로 pipe는 이 값이었음.
pipe = make_pipeline(
     OrdinalEncoder(), 
     DecisionTreeClassifier(max_depth=5, random_state=2)
)

fig, ax = plt.subplots()
pcm = plot_confusion_matrix(pipe, X_val, y_val,
                            cmap=plt.cm.Blues,
                            ax=ax);
plt.title(f'Confusion matrix, n = {len(y_val)}', fontsize=15)
```

<br/>

## 7. classification_report (리포트)
```python
# 렉쳐노트 n231 예시
from sklearn.metrics import classification_report

y_pred = pipe.predict(X_val)
print(classification_report(y_val, y_pred))

==결과==
              precision    recall  f1-score   support

       False       0.84      0.98      0.91       302
        True       0.17      0.02      0.03        57

    accuracy                           0.83       359
   macro avg       0.50      0.50      0.47       359
weighted avg       0.73      0.83      0.77       359
```

<br/>

## 8. AUC score (roc_auc_score)
```python
# 렉쳐노트 n231 예시
from sklearn.metrics import roc_auc_score

# 예측 확률
y_pred_proba = pipe.predict_proba(X_val)[:, -1]
print('AUC score: ', roc_auc_score(y_val, y_pred_proba))

==결과==
# 형태확인
AUC score:  0.5991634715928895
```

<br/>

## 9. ROC curve 그리기 (roc_curve)
```python
# 렉쳐노트 n231 예시
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

plt.scatter(fpr, tpr, color='blue')
plt.plot(fpr, tpr, color='green')
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
```


<img width="382" alt="스크린샷 2022-11-28 오후 11 49 55" src="https://user-images.githubusercontent.com/81222323/204310449-df0f7c12-7837-457c-ac6a-7df97e790430.png">

<br/>

## 10. 불균형클래스 (class_weight)
a. 데이터가 적은 범주 데이터의 손실을 계산할 때 가중치를 더 곱하여 데이터의 균형을 맞추거나 <br/>
b. 적은 범주 데이터를 `오버샘플링(oversampling)`하거나 반대로 많은 범주 데이터를 `언더샘플링(undersampling)`하는 방법이 있다.


```python
# 렉쳐노트 n231 예시

# 1 범주의 비율 확인
y_train.value_counts(normalize=True)

==결과==
False    0.824268
True     0.175732
Name: recommend, dtype: float64



# 2 파이프라인 만들기
pipe = make_pipeline(
    OrdinalEncoder(), 
    # DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=2)
    DecisionTreeClassifier(max_depth=5, class_weight={False:custom[0],True:custom[1]}, random_state=2)
)

pipe.fit(X_train, y_train)
print('검증 정확도: ', pipe.score(X_val, y_val))

==결과==
검증 정확도:  0.584958217270195



# 혼동행렬을 그려서 확인
fig, ax = plt.subplots()
pcm = plot_confusion_matrix(pipe, X_val, y_val,
                            cmap=plt.cm.Blues,
                            ax=ax);
plt.title(f'Confusion matrix, n = {len(y_val)}', fontsize=15)
```


<img width="332" alt="스크린샷 2022-11-29 오전 12 02 56" src="https://user-images.githubusercontent.com/81222323/204310649-3a77b9ea-391e-4a1c-803b-9dd8b5fae090.png">


```python
# 렉쳐노트 n231 예시

# True 범주의 수치 확인(비교)
y_pred = pipe.predict(X_val)
print(classification_report(y_val, y_pred))

==결과==

y_pred = pipe.predict(X_val)
print(classification_report(y_val, y_pred))
              precision    recall  f1-score   support

       False       0.86      0.60      0.71       302
        True       0.19      0.49      0.27        57

    accuracy                           0.58       359
   macro avg       0.53      0.55      0.49       359
weighted avg       0.76      0.58      0.64       359



# AUC score 및 ROC curve 확인
y_pred_proba = pipe.predict_proba(X_val)[:, -1]
print('AUC score: ', roc_auc_score(y_val, y_pred_proba))
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

plt.scatter(fpr, tpr, color='blue')
plt.plot(fpr, tpr, color='green')
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')

==결과==
AUC score:  0.624056000929476
```


<img width="386" alt="스크린샷 2022-11-29 오전 12 04 56" src="https://user-images.githubusercontent.com/81222323/204311268-69449588-94ab-475e-9f9c-30ce1485cba6.png">

<br/>

## 11. 로그변환(Log-Transform)
```python
# 렉쳐노트 n231 예시

plots=pd.DataFrame()
plots['original']=target
plots['transformed']=np.log1p(target)
plots['backToOriginal']=np.expm1(np.log1p(target))

fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.histplot(plots['original'], ax=ax[0]);
sns.histplot(plots['transformed'], ax=ax[1]);
sns.histplot(plots['backToOriginal'], ax=ax[2]);
```


<img width="897" alt="스크린샷 2022-11-29 오전 12 19 02" src="https://user-images.githubusercontent.com/81222323/204314345-0e6e1e3f-ff66-47d0-a4ab-bda06fdaf232.png">

<br/>
 
## 12. TransformedTargetRegressor
```python
# 렉쳐노트 n231 예시

target = 'SalePrice'
from sklearn.model_selection import train_test_split

df = df[df[target].notna()]

train, val = train_test_split(df, test_size=260, random_state=2)

features = train.drop(columns=[target]).columns

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]



# 타겟 변환 전
from sklearn.linear_model import Ridge
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor

pipe = make_pipeline(
    OneHotEncoder(), 
    SimpleImputer(),  
    Ridge(alpha=0.01)
)
pipe.fit(X_train, y_train)
pipe.score(X_val, y_val)

==결과==
0.7223389015612416



# 타겟 변환 후
pipe = make_pipeline(
    OneHotEncoder(), 
    SimpleImputer(),  
    Ridge(alpha=0.01)
)

tt = TransformedTargetRegressor(regressor=pipe,
                                func=np.log1p, inverse_func=np.expm1)

tt.fit(X_train, y_train)
tt.score(X_val, y_val)

==결과==
0.8539727902315182
```
