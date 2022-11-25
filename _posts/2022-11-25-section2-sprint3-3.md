---
title: Xgboost, Gradient boosting
author: yun
date: 2022-11-25
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, Xgboost]
---

# 목표
- [ ] 특성 중요도 계산 방법들을 이해하고 사용하여 모델을 해석하고 특성 선택시 활용할 수 있다.
- [ ] Gradient Boosting을 이해하고 Xgboost를 사용하여 모델을 만들 수 있다.

<br/>

# 키워드
- [ ] XAI
- [ ] AdaBoost
- [ ] Xgboost
- [ ] LightGBM
- [ ] CatBoost
- [ ] Early Stopping

<br/>

# NOTE
## Feature inportance
a. MDI 
  * 트리베이스 모델에서 사용  `rf.feature_importance_`
  * 높은 카디널리티, 분기가 높으면 많이 사용된다, 그러면 특성 중요도가 올라간다. (왜곡될 수 있다.)


b. Drop Column
  * 속도가 느리다는 단점이 있다.


c. 순열 중요도
  * drop을 하지 않고 noise를 준다.
  * 성능이 하락할 경우 중요한 특성임을 알 수 있다.

<br/>

## Boosting
a. 배깅
  * 독립적으로 트리를 만든다.


b. 부스팅
  * 영향을 준다.
  * 이전에 제대로 분류되지 않은 값들에 가중치를 준 후, 다음에 더욱 잘 분류될 수 있도록 함.


c. Gradient Boosted Trees
  * 특이하게도 잔차를 계산한다.


d. AdaBoost
e. LightGBM
f. CatBoost

g. XgBoost


> XGB에는 reg_lambda와 같은 정규항을 넣어줄 수 있다. <br/>
> 왜 XGB와 그라디언트 부스팅에는 이런 과적합 방지 기술(?)들이 많을까? 생각해보자.
{: .prompt-tip }
