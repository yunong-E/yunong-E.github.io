---
title: 릿지회귀모델(Ridge Regression)
author: yun
date: 2022-11-11
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, ridge regression, lambda]
---

## 키워드
- [ ] 릿지회귀
- [ ] 원핫인코딩
- [ ] 다중공선성

 <br/> <br/>

## 정리필요
람다는 0부터 무한대의 양의 값을 가질 수 있다. <br/>
$$\lambda$$ can be any value from `$$0$$` to `$$positive infinity$$` <br/>
When $$\lambda=0$$, then the **Ridge Regression Penalty is also 0.** <br/>
Ridge regression can also be applied to **Logistic Regression**. <br/>
실무에서는 특성공학에 많은 시간을 할애한다. <br/> <br/>


## 용어설명
**좋은 특성을 추출하는 방법**
  : 특성들끼리는 상관성이 적으면서 타겟특성과는 상관성이 큰 특성을 추출한다. <br/>

**SelectKBest**
  : 타겟 특성 A과 가장 상관관계가 높은 특성 k개를 추출할 수 있다. 각 특성들이 독립적으로 평가가 되므로 계산이 매우 빠르다. 모델과 상관없이 전처리에서 사용할 수 있다. k값을 너무 작게 넣어서 중요한 특성(변수)들이 제외되지 않도록 하는 것이 중요하다. <br/>
  
**RidgeCV**
 : 데이터를 교차검증하여 최적의 패널티 값을 찾아준다.  <br/>
 
**가중치감소**
  : 학습 과정에서 큰 가중치(회귀계수, 기울기)에 대해서는 큰 패널티를 부과하여 가중치의 절대값을 가능한 작게 만든다. **규제**란 과대적합이 되지 않도록 모델을 강제로 제한하는 의미로 L1규제, L2규제가 있다. <br/>
  
**릿지회귀(능형회귀, Ridge Regression)**  
  : * 손실함수에 가중치에 대한 L2노름(norm)의 제곱을 더한 패널티를 부여하여 가중치 값을 비용함수 모델에 비해 작게 만들어 낸다.
  : * 손실함수가 최소가 되는 가중치값인 중심점을 찾아 큰 가중치를 제한하는데 $$lambda$$(람다)로 규제의 강도를 크게 하면 가중치(기울기)는 $$0$$에 가까워진다.
  : * 모델의 복잡도에 따라 벌점을 부여하는 방식.
  : * RSS(잔차제곱합)에 회귀계수의 개수와 크기에 대한 함수인 벌점을 추가한 값을 최소화한다.  $$\lambda$$는 계수에 대해 어느 정도 벌점을 부여할 것인가를 결정한다. 이 값이 클 수록 모델이 데이터에 오버피팅(과적합)할 가능성이 낮아진다.
  : * 람다 값이 커질 수록 기울기가 점점 낮아진다. 크기에 대한 예측이 가중치에 덜 민감해진다. <br/><br/>
  
  
* 람다는 튜닝파라미터(초매개변수, 하이퍼파라미터)로 값이 커질수록, 회귀계수(기울기, 가중치)를 0으로 수렴시킵니다. (0으로 만들지는 않음) <br/>
* 람다 값이 0이되면 보통 다중회귀모델이 된다. <br/> 따라서, 적절한 람다값을 찾아내서 일반화가 잘 되는 지점(적절한 패널티값)을 찾아야 한다. <br/>
* 우리는 이것을 정규화모델 이라고 한다. <br/><br/>

어떻게 적절한 람다의 값을 구할 수 있을까? -> 검증실험 (RidgeCV를 활용.) <br/>

