---
title: 단순선형회귀모델, 회귀계수
author: yun
date: 2022-11-09
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, simple regression, tabular data, supervised learning, regression, classification]
---

## 회귀선을 그리기 위한 계수를 구하는 공식?

- [x] 회귀계수
- [x] 잔차

<br/>
<br/>

회귀계수(regression coefficient)
  : 회귀직선의 기울기 (유의어: $기울기^slope$, $b_1$, $\beta_1$, 모수 추정값, 가중치) <br/>

잔차(residual)
  : 관측값과 적합값의 차이, 관측값 - 예측값 (유의어: 오차)
  
절편(intercept)
  : 회귀직선의 절편, 즉, $X = 0$일 때 예측값  


## Tabular Data는 세 가지 주요 구성으로 나눌 수 있다.
1. **Observations(관찰)**:  테이블의 행
2. **variables(변수)**: 테이블의 열에 위치
3. **Relationships(관계)**: 한 테이블의 데이터를 다른 테이블에 있는 데이터와 연결. 


## 분류와 회귀의 비교
* “How Much / How Many?” (회귀)
* “Is this A or B?” (분류) <br/>

a. 알고리즘의 출력유형
* Supervised calssification(지도 분류): discrete (calss labels) - 클래스레이블의 형식으로 이산적임.
* regresstion(회귀): continuous(number) - 연속적 <br/><br/>

b. what are you trying to find? (분류 또는 회귀를 수행할 때 실제로 무엇을 찾으려고 합니까?)
* Supervised calssification(지도 분류): dicision boundary(결정경계) - 결정경계를 기준으로 점이 따라오는 위치에 따라 클래스 레이블을 할당할 수 있다.
* regresstion(회귀): best fit line(최적선): 테이터를 설명하는 경계(X) **데이터에 맞는 선(O)** <br/><br/>

c. evaluation(평가방법)
* Supervised calssification(지도 분류): Accuracy(정확도)
* regresstion(회귀): Sum of squared error(오차제곱의 합), $r^2$ (r squared)

<br/>
<br/>


| 항목                          | Supervised calssification(지도 분류) | regresstion(회귀)    |
|:-----------------------------|:-----------------------------------|--------------------:|
| 알고리즘의 출력유형               | discrete(class labels)             | continuous(number)  |
| what are you trying to find? | dicision boundary(결정경계)          | best fit line(최적선) |
| evaluation(평가방법)           | Accuracy(정확도)                     | SSE, $r^2$          |


<br/>
회귀가 지도 분류와 정확히 동일하지 않다. <br/>
지도 분류에 대해 이미 알고 있는 것들이 회귀 분류에 직접적인 유사점을 가지고 있다. <br/>
따라서, 회귀를 다른 유형의 지도 학습으로 생각해야 한다. 처음부터 배워야하는 완전히 새로운 주제가 아니다.
