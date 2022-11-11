---
title: 릿지회귀모델(Ridge Regression)
author: yun
date: 2022-11-11
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, ridge regression, lambda]
---

람다는 0부터 무한대의 양의 값을 가질 수 있다. <br/>
$$\lambda$$ can be any value from $$0$$ to $$positive infinity$$ <br/>
When $$\lambda=0$$, then the **Ridge Regression Penalty is also 0.** <br/>

람다 값이 커질 수록 기울기가 점점낮아진다. 크기에 대한 예측이 가중치에 덜 민감해진다. <br/>
람다 값이 0이되면 보통 다중회귀모델이 된다. <br/>
Ridge regression can also be applied to **Logistic Regression**. <br/>

어떻게 적절한 람다의 값을 구할 수 있을까? -> 검증실험 (RidgeCV를 활용.) <br/>


- [ ] 원핫인코딩
- [ ] 다중공선성


실무에서는 특성공학에 많은 시간을 할애한다. <br/>

**좋은 특성을 추출하는 방법**
  : 특성들끼리는 상관성이 적으면서 타겟특성과는 상관성이 큰 특성을 추출한다.

**SelectKBest**
  : 타겟 특성 A과 가장 상관관계가 높은 특성 k개를 추출할 수 있다. 각 특성들이 독립적으로 평가가 되므로 계산이 매우 빠르다. 모델과 상관없이 전처리에서 사용할 수 있다. k값을 너무 작게 넣어서 중요한 특성(변수)들이 제외되지 않도록 하는 것이 중요하다.
  
