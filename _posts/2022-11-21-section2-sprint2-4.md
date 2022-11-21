---
title: 교차검증 (k-fold, LOOCV)
author: yun
date: 2022-11-21
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, k-fold, loocv]
---

# 목표
* 교차검증을 사용하는 이유가 무엇일까?
  1. 내가 가지고 있는 데이터셋에 적합한 모델을 찾기 위해?
  3. 어떤 하이퍼파라미터를 사용할 것인지 알기 위해?

<br/>

# 키워드
- [x] Hold-out Cross-validation
- [x] k-fold cross-validation
- [ ] Leave One Out Cross Validation
- [x] 모델선택(Model selection)
- [x] 하이퍼파라미터 튜닝
- [x] 최적화(optimization)
- [x] 일반화(generalization)
- [x] 검증곡선(Validation curve)
- [ ] 훈련곡선(learning curve)

<br/>

# 용어설명
* 모델선택(Model selection)
  : 주어진 문제를 풀기 위해 어떤 모델을 사용할 것인지?
  : 어떤 하이퍼파라미터를 사용할 것인지?
  
*  k-fold cross-validation(CV)
  : k개의 집합에서 k-1개의 부분집합을 **훈련**에 사용하고 나머지 부분집합을 테스트 데이트로 검증하는 것.
  : 예를들어, 데이터를 3등분으로 나누고 검증(1/3)과 훈련세트(2/3)를 총 3번 바꾸어가며 검증하는 것은 3-fold CV, 총 10번 검증하는 것은 10-fold CV

* 최적화(optimization)
  : 훈련 데이터로 더 좋은 성능을 얻기 위해 모델을 조정하는 과정.
  
* 일반화(generalization)
  : 학습된 모델이 처음 본 데이터에서 얼마나 좋은 성능을 내는 지를 말한다.

* 검증곡선(Validation curve)
  : 훈련/검증데이터에 대해 y축: 스코어,  x축: 하이퍼파라미터 로 그린 그래프
  
* 훈련곡선(learning curve)
  : 훈련곡선(learning curve)이라는 용어는 x축이 훈련데이터 수(# of training samples)에 대해 그린 것. 검증곡선과 혼동하지 말 것!
  
* [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
  : 검증하고 싶은 하이퍼파라미터들의 수치를 정해주고 그 조합을 ***모두*** 검증합니다.

* [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
  : 검증하려는 하이퍼파라미터들의 값 범위를 지정해주면 ***무작위로*** 값을 지정해 그 조합을 모두 검증한다.

<br/>

# 나만의 언어로 설명해보기
* Hold-out Cross-validation
  : k-fold 교차검증을 알기 전까지 내가 했던 데이터세트 분리 방법. (훈련/검증/테스트)

<br/>
  
# 학습이 더 필요한 부분
- [ ] 선형회귀, 랜덤포레스트 모델들의 튜닝 추천 하이퍼파라미터. 설명할 수 있을 정도로 ..
  **Random Forest**
    * `class_weight`: 불균형(imbalanced) 클래스인 경우
    * `max_depth`: 너무 깊어지면 과적합
    * `n_estimators`: 적을경우 과소적합, 높을경우 긴 학습시간 (나무들의 갯수임)
    * `min_samples_leaf`: 과적합일 경우 높임
    * `max_features`: 줄일 수록 다양한 트리생성

  **Logistic Regression**
    * `C`: Inverse of regularization strength)
    * `class_weight`: 불균형 클래스인 경우
    * `penalty`

  **Ridge / Lasso Regression**
    * `alpha`

<br/>

# NOTE
* 교차검증은 시계열(time series) 데이터에는 적합하지 않음.
* Hold-out Cross-validation의 ***단점***이 있다.
  1. 훈련세트의 크기가 모델학습에 충분하지 않을 경우 문제가 될 수 있다. (= 학습할 데이터가 많으면 문제 없다.)
  2. Validation set(검증세트) 크기가 충분히 크지 않다면 예측 성능에 대한 추정이 부정확하다.

* random forest model에서 주요 parameter는
  1. n_estimators : 생성할 tree의 개수와
  2. max_features : 최대 선택할 특성의 수입니다.

* 모델의 복잡도를 높이는 과정에서 훈련/검증 세트의 손실이 함께 감소하는 시점은 `과소적합(underfitting)` 되었다고 한다.
* 훈련데이터의 손실은 계속 감소하는데 검증데이터의 손실은 증가하는 때가 있습니다. 이때 우리는 `과적합(overfitting)` 되었다고 한다.

<br/>

# Code
## from category_encoders import TargetEncoder
* 특성은 특정 범주 값이 주어진 대상의 사후 확률과 모든 훈련 데이터에 대한 대상의 사전 확률의 혼합으로 대체됩니다.
* 이게 무슨말이냐? 하고 이해가 안된다면 [이 문서](https://medium.com/analytics-vidhya/target-encoding-vs-one-hot-encoding-with-simple-examples-276a7e7b3e64)를 봐주세요. 
<br/>

```python
# 렉처노트 n224 예시

from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

pipe = make_pipeline(
    # TargetEncoder: 범주형 변수 인코더로, 타겟값을 특성의 범주별로 평균내어 그 값으로 인코딩
    TargetEncoder(min_samples_leaf=1, smoothing=1), 
    SimpleImputer(strategy='median'), 
    RandomForestRegressor(max_depth = 10, n_jobs=-1, random_state=2)
)

k = 3

scores = cross_val_score(pipe, X_train, y_train, cv=k, 
                         scoring='neg_mean_absolute_error')

print(f'MAE for {k} folds:', -scores)
```
```
=결과=
# 형태 확인용
MAE for 3 folds: [16289.34502313 19492.01218055 15273.23000751]
```
* 참고
  * 생각보다 성능이 좋아서 현재 많이 사용되고 있는 방법임. 여러분들 많이 사용해보세요.
<br/>

```python
# min_samples_leaf=1 범주별 타겟의 평균을 계산하기 위한 최소한의 범주의 샘플 수.
# smoothing=1000 범주별 타겟평균을 그대로 쓰지 않고 정규화를 해주는 속성. 그래서 값이 높아질 수록 평균값과 많이 달라지게 된다.
enc = TargetEncoder(min_samples_leaf=1, smoothing=1000) 

# LotShape 특성을 타겟 인코더로 인코딩 해 봄.
enc.fit_transform(X_train,y_train)['LotShape'].value_counts()

=결과=
# 형태 확인용
166368.017847    759
190133.621187    391
194093.059489     28
196089.739213     10
Name: LotShape, dtype: int64


X_train['LotShape'].value_counts()

=결과=
Reg    759
IR1    391
IR2     28
IR3     10
Name: LotShape, dtype: int64


# 'Reg'에 해당하는 범주에 타겟값의 평균.
# 위에서 구한 166368.017847    759 값과 완전히 똑같지 않음을 알 수 있다.
y_train[X_train[X_train.LotShape == 'Reg'].index].sum() / 759

=결과=
161871.9486166008
```

<br/>

## from sklearn.model_selection import cross_val_score
* [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn-model-selection-cross-val-score)


## warning 제거를 위한 코드
```python
# 1
np.seterr(divide='ignore', invalid='ignore')

# 2
import warnings
warnings.filterwarnings('ignore')
```

<br/>

## from sklearn.model_selection import validation_curve
* [from sklearn.model_selection import validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn-model-selection-validation-curve)
```python
# 렉처노트 n224 예시

import matplotlib.pyplot as plt
from category_encoders import OrdinalEncoder
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor

pipe = make_pipeline(
    OrdinalEncoder(), 
    SimpleImputer(), 
    DecisionTreeRegressor()
)


# 랜덤서치나 그리드서치는 내가 한 설정에 따라서 시간이 오래 걸릴 수도 있으므로
# 꼭 n_iter(이터레이션) 값을 작게 시작하고 필요한 만큼 늘려가며 사용하는 것을 추천.
depth = range(1, 30, 2)
ts, vs = validation_curve(
    pipe, X_train, y_train
    , param_name='decisiontreeregressor__max_depth'
    , param_range=depth, scoring='neg_mean_absolute_error'
    , cv=3
    , n_jobs=-1
)

train_scores_mean = np.mean(-ts, axis=1)
validation_scores_mean = np.mean(-vs, axis=1)

fig, ax = plt.subplots()

# 훈련세트 검증곡선
ax.plot(depth, train_scores_mean, label='training error')

# 검증세트 검증곡선
ax.plot(depth, validation_scores_mean, label='validation error')

# 이상적인 max_depth
ax.vlines(5,0, train_scores_mean.max(), color='blue')

# 그래프 셋팅
ax.set(title='Validation Curve'
      , xlabel='Model Complexity(max_depth)', ylabel='MAE')
ax.legend()
fig.dpi = 100
```
> 이대로 트리의 깊이를 정해야 한다면, <br/>
> 'max_depth = 5 부근에서 설정해 주어야 과적합을 막고 일반화 성능을 지킬 것 같다'는 결론을 도출할 수 있다.
<br/>
![validation_curve](https://github.com/yunong-E/utterances_only/blob/main/assets/img/validation_curve.png)

<br/>

## from sklearn.model_selection import RandomizedSearchCV
* 하이퍼파라미터의 최적값을 찾으러 가보죠. 
* 참고로 하이퍼파라미터는 사용자가 직접 지정을 해줘야 하는 값 이란다.
* 최고로 적합한 하이퍼파라미터를 수작업으로 계산(알아내는 것)하는 것은 `노가다`임. 반박불가.
* 그러니 도구(`GridSearchCV`, `RandomizedSearchCV`)를 사용하여 구해보자.
* `best_estimator_` 는 CV가 끝난 후 찾은 best parameter를 사용해 모든 학습데이터(all the training data)를 가지고 다시 학습(refit)한 상태.
  * 따라서, hold-out교차검증을 수행한 경우 (훈련 + 검증)데이터셋에서 최적화된 하이퍼파라미터로 최종 모델을 재학습(refit) 해야 한다. 
<br/>

a. 릿지회귀모델에 적용
<br/>

```python
# 렉처노트 n224 예시

from sklearn.model_selection import RandomizedSearchCV

# 릿지회귀모델로 가보자고
pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True)
    , SimpleImputer()
    , StandardScaler()
    , SelectKBest(f_regression)
    , Ridge()
)

# 튜닝할 하이퍼파라미터의 범위를 지정해 주는 부분
dists = {
    'simpleimputer__strategy': ['mean', 'median'], 
    'selectkbest__k': range(1, len(X_train.columns)+1), 
    'ridge__alpha': [0.1, 1, 10], 
}

# 랜덤서치나 그리드서치는 내가 한 설정에 따라서 시간이 오래 걸릴 수도 있으므로
# 꼭 n_iter(이터레이션) 값을 작게 시작하고 필요한 만큼 늘려가며 사용하는 것을 추천.
clf = RandomizedSearchCV(
    pipe, 
    param_distributions=dists, 
    # 각 이터레이션 당 교차검증(cv) 3회씩 시행 50*3= 150
    n_iter=50, 
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train, y_train);



# 위에서 작업한 파라미터를 clf.best_params_ 속성을 통해 확인할 수있다.
print('최적 하이퍼파라미터: ', clf.best_params_)

# 검증스코어도 확인 가능
print('MAE: ', -clf.best_score_)

==결과==
최적 하이퍼파라미터:  {'simpleimputer__strategy': 'median', 'selectkbest__k': 55, 'ridge__alpha': 10}
MAE:  18414.633797820472
```

<br/>

b. 랜덤포레스트에 적용
<br/>

```python
# 렉처노트 n224 예시

from scipy.stats import randint, uniform

# 랜덤포레스트 가보자고
pipe = make_pipeline(
    TargetEncoder(), 
    SimpleImputer(), 
    RandomForestRegressor(random_state=2)
)

dists = {
    'targetencoder__smoothing': [2.,20.,50.,60.,100.,500.,1000.], # int로 넣으면 error(bug)
    'targetencoder__min_samples_leaf': randint(1, 10),     
    'simpleimputer__strategy': ['mean', 'median'], 
    'randomforestregressor__n_estimators': randint(50, 500), 
    'randomforestregressor__max_depth': [5, 10, 15, 20, None], 
    'randomforestregressor__max_features': uniform(0, 1) # max_features
}

clf = RandomizedSearchCV(
    pipe, 
    param_distributions=dists, 
    n_iter=50, 
    cv=3, 
    scoring='neg_mean_absolute_error',  
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train, y_train);

print('최적 하이퍼파라미터: ', clf.best_params_)
print('MAE: ', -clf.best_score_)

==결과==
최적 하이퍼파라미터:  {'randomforestregressor__max_depth': 20, 'randomforestregressor__max_features': 0.22612308958451122, 'randomforestregressor__n_estimators': 498, 'simpleimputer__strategy': 'mean', 'targetencoder__min_samples_leaf': 8, 'targetencoder__smoothing': 1000.0}
MAE:  15741.360087309344



# 만들어진 모델에서 가장 성능이 좋은 모델을 불러온다.
# clf.best_estimator_ 오호.
pipe = clf.best_estimator_

from sklearn.metrics import mean_absolute_error

y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'테스트세트 MAE: ${mae:,.0f}')

==결과==
테스트세트 MAE: $15,778
```
