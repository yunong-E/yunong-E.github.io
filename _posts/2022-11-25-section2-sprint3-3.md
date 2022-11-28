---
title: Xgboost, Gradient boosting
author: yun
date: 2022-11-25
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, Xgboost]
---

# 목표
- [ ] 특성 중요도 계산 방법들을 이해하고 사용하여 모델을 해석하고 특성 선택시 활용할 수 있다.
- [x] Gradient Boosting을 이해하고 Xgboost를 사용하여 모델을 만들 수 있다.

<br/>

# 키워드
- [ ] [XAI](https://youtu.be/6xePkn3-LME)
- [ ] AdaBoost
- [x] Xgboost
- [ ] LightGBM
- [ ] CatBoost
- [x] Early Stopping

<br/>

# NOTE
## Feature importance
a. **MDI** 
  * 트리베이스 모델에서 사용  `rf.feature_importance_`
  * 높은 카디널리티, 분기가 높으면 많이 사용된다, 그러면 특성 중요도가 올라간다. (왜곡될 수 있다.)
  * 먼저 사용되고 자주 사용될 수록 불순도 감소량이 높아진다. 따라서 High Cardinality의 특성이 특성 중요도가 높게 나온다.
  * 빠르기는 하지만 항상 중요성에 대한 정확한 그림을 제공하지는 않는다는 것.
  * 잠재적인 예측 변수가 측정 규모나 범주 수가 다른 상황에서 신뢰할 수 없다.
  * 불순물 기반 중요도는 높은 카디널리티 기능으로 편향된다.
  * 불순도 기반 중요도는 교육 세트 통계에서 계산되므로 테스트 세트로 일반화되는 예측을 만드는 데 유용한 기능의 기능을 반영하지 않는다(모델의 용량이 충분한 경우).

<br/>

b. **Drop Column**
  * 속도가 느리다는 단점이 있다.
  * 모델을 재훈련하는 ***계산 비용을 무시하면*** 무차별 강제 삭제 열 중요도 메커니즘 을 사용하여 가장 정확한 기능 중요도를 얻을 수 있다.
  * Drop Column importance는 반복되는 모델 교육으로 인해 계산하는 데 여전히 매우 비싸다.

<br/>

c. **순열 중요도**
  * drop을 하지 않고 noise를 준다.
  * 성능이 하락할 경우 중요한 특성임을 알 수 있다.
  * 각각의 특성을 무작위로 배열(shuffle)하여 제 기능을 하지 못하게 만든 뒤 결과를 비교해, 각 특성의 중요도를 계산한다. 즉, 특성을 아예 제외(Drop)하는 것이 아니기 때문에 각각의 모델을 학습할 필요가 없어 시간이 훨씬 단축된다는 장점이 있다.
  * It is model agnostic (모델에 구애받지 않는다는 뉘앙스)
  * 순열 중요도 이용시 사전에 ***특성간 상관관계*** 가 있는지 없는지 파악할 것!


```python
For each feature column:
  1. shuffle feature column
  2. observe performance and compare to original
```

<br/>

참고자료
1. [Beware Default Random Forest Importances](https://explained.ai/rf-importance/)
2. [Tree’s Feature Importance from Mean Decrease in Impurity (MDI)](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#tree-s-feature-importance-from-mean-decrease-in-impurity-mdi)
3. [데이터 분석 초보자를 위한 순열 중요도와 PDP](https://velog.io/@gayeon/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D-%EC%B4%88%EB%B3%B4%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-%EC%88%9C%EC%97%B4-%EC%A4%91%EC%9A%94%EB%8F%84%EC%99%80-PDP)

<br/>

## Bagging VS Boosting
a. **Bagging**
  * 서로 독립적으로 병렬로(동시에) 학습하고 모델 평균을 결정하기 위해 결합하는 동종 약한 학습자 모델.
  * `Bootstrap Aggregation`의 약어이며 예측 모델의 분산을 줄이는 데 사용된다.
  * 분산을 줄이고 모델의 과적합 문제를 해결한다.
  * 일반적으로 분류기가 불안정하고 분산이 큰 경우에 적용한다.

<br/>

b. **Boosting**
  * 마지막 분류에 따라 관찰 가중치를 반복적으로 조정하는 순차적 앙상블 방법.
  * 관측치가 잘못 분류되면 해당 관측치의 가중치가 증가한다.
  * '부스팅'이라는 용어는 약한 학습자를 더 강한 학습자로 변환하는 알고리즘을 나타낸다.
  * 편향 오류를 줄이고 강력한 예측 모델을 구축한다.
  * 일반적으로 분류기가 안정적이고 단순하며 편향이 높은 경우에 적용한다.

<br/>

c. Gradient Boosted Trees
  * 특이하게도 잔차를 계산한다.

<br/>

d. AdaBoost <br/>
e. LightGBM <br/>
f. CatBoost <br/>

<br/>

## g. **XgBoost**
* (R에서) `subsample` 설정에 따라 비복원 추출로 샘플링 한다는 점만 빼면 부스팅은 마치 랜덤 포레스트와 같이 동작한다고 한다.
* (R에서) `eta`는 가중치의 변화량을 낮추어 오버피팅(과적합)을 방지하는 효과가 있다고 한다. 파이썬에서는 `learning_rate`

<br/>

a. xgboost 함수를 무작정 사용할 경우 과적합이 될 수 있으며 과적합은 다음과 같은 두 가지의 문제를 야기한다.
  1. train 데이터에 없는 새로운 데이터에 대한 모델의 정확도가 떨어짐.
  2. 모델의 예측 결과에 변동이 매우 심하고 불안정한 결과를 보임. 

<br/>
b. `learning_rate` 같은 파라미터를 통해 과적합을 방지할 수 있다.

<br/>
c. **정규화(regularization)** 방법도 있다.
  1.  정규화를 위한 두 파라미터 `reg_alpha`, `reg_lambda`가 있다.
  2.  이 파라미터들을 크게하면, 모델이 복잡해질수록 더 많은 벌점을 부여하고 결과적으로 얻는 트리의 크기가 작아진다.


> XGB에는 reg_lambda와 같은 정규항을 넣어줄 수 있다. <br/>
> 왜 XGB와 그라디언트 부스팅에는 이런 과적합 방지 기술(?)들이 많을까? 생각해보자.
{: .prompt-tip }

<br/>

## eval_set의 형태에 대한 궁금증
```python
# 1
# (x_test, y_test)를 검증 데이터 세트를 지정하고 
eval_set = [(x_train, y_train), (x_test, y_test)] 

# (x_test, y_test)를 검증 데이터로 지정합니다.
eval_set = (x_test, y_test)
```


참고자료: [eval_set 관련 헷갈리는 부분 질문드립니다.](https://www.inflearn.com/questions/193982)

<br/>

## pipeline VS make_pipeline
이제는 알지.
