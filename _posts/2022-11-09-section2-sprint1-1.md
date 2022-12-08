---
title: 단순선형회귀모델, 회귀계수
author: yun
date: 2022-11-09
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, simple regression, tabular data, supervised learning, regression, classification]
---

# 목표
- 선형회귀모델을 이해하고 설명할 수 있다.
- 지도학습(Supervised Learning) 이해하고 설명할 수 있다.
- 회귀모델에 기준모델을 설정할 수 있다.
- Scikit-learn을 이용해 선형회귀 모델을 만들어 사용하고 해석할 수 있다.

<br/>
# 키워드
- [x] 회귀계수(Regression Coefficient)
- [x] 예측값
- [x] 잔차(Residual) 
- [x] MAE(평균절대오차, Mean Absolute Error)
- [X] RSS(Residual Sum of Squares)
- [x] OLS(Ordinary least squares), 최소제곱회귀
- [x] 보간(Interpolate)

<br/>

# 용어설명
* 회귀계수(Regression Coefficient)
  : 회귀직선의 기울기 (유의어: $기울기^slope$, $b_1$, $$\beta_1$$, 모수 추정값, 가중치)

* 예측값
  : 만들어진 모델이 추정하는 값.

* 잔차(residual)
  : 예측값과 관측값의 차이. (유의어: 오차)
  
* 오차(Error)
  : **모집단에서의** 예측값과 관측값의 차이.
  
* 절편(intercept)
  : 회귀직선의 절편, 즉, $$X = 0$$일 때 예측값.
  
* RSS(Residual Sum of Squares), SSE(Sum of Square Error)
  : 잔차 제곱들의 합(이를 최소화 하는 것이 회귀선이다.), 회귀모델의 `비용함수(Cost function)`다. (비용함수를 최소화 하는 모델을 찾는 과정을 학습이라고 한다.)

* OLS(Ordinary least squares), 최소제곱회귀
  : 잔차제곱합을 최소화하는 방법. 이를 통해 선형 회구계수를 쉽게 구할 수 있다.

<br/>

# NOTE
## Tabular Data는 세 가지 주요 구성으로 나눌 수 있다.
1. **Observations(관찰)**:  테이블의 행
2. **variables(변수)**: 테이블의 열에 위치
3. **Relationships(관계)**: 한 테이블의 데이터를 다른 테이블에 있는 데이터와 연결. 

## 모델을 만들고 분석하기 위한 데이터구조
<img width="431" alt="스크린샷 2022-12-07 오후 10 50 24" src="https://user-images.githubusercontent.com/81222323/206196525-edb98ad9-f0d8-433e-b8ff-53468dffbf22.png">


* 특성 데이터(Features)와 타겟 데이터를 나누어 준다.

* 특성행렬은 주로 X로 표현하며 보통 2-차원 행렬(`[n_samples, n_features]`). 주로 `NumPy 행령`이나 `Pandas 데이터프레임`으로 표현.

* 타겟배열은 주로 y로 표현하고 보통 1차원 형태(`n_samples`). 주로 `Numpy 배열`이나 `Pandas Series`로 표현.


참고자료: [Scikit-Learn 소개](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API)
<br/>

## 분류와 회귀의 비교
* “How Much / How Many?” (회귀)
* “Is this A or B?” (분류) <br/>

<br/>

  **a. 알고리즘의 출력유형**
  * Supervised calssification(지도 분류): discrete(calss labels, 이산적)
  * regresstion(회귀): continuous(number, 연속적)

<br/>

  **b. what are you trying to find? (분류 또는 회귀를 수행할 때 실제로 무엇을 찾으려고 합니까?)**
  * Supervised calssification(지도 분류): dicision boundary(결정경계를 기준으로 점이 따라오는 위치에 따라 클래스 레이블을 할당할 수 있다.)
  * regresstion(회귀): best fit line(= 테이터를 설명하는 경계(X) **데이터에 맞는 선(O)**)

<br/>

  **c. evaluation(평가방법)**
  * Supervised calssification(지도 분류): Accuracy(정확도)
  * regresstion(회귀): Sum of squared error(오차제곱의 합), $$ r^2 $$ (r squared)

<br/>

  | 항목                          | Supervised calssification(지도 분류) | regresstion(회귀)    |
  |:-----------------------------|:-----------------------------------|:--------------------|
  | 알고리즘의 출력유형               | discrete(class labels)             | continuous(number)  |
  | what are you trying to find? | dicision boundary(결정경계)          | best fit line(최적선) |
  | evaluation(평가방법)           | Accuracy(정확도)                     | SSE, $$r^2$$        |

<br/>

  * 회귀가 지도 분류와 정확히 동일하지 않다.
  * 지도 분류에 대해 이미 알고 있는 것들이 회귀 분류에 직접적인 유사점을 가지고 있다. 
  * 따라서, 회귀를 다른 유형의 지도 학습으로 생각해야 한다. 처음부터 배워야하는 완전히 새로운 주제가 아니다.


## 선형회귀모델(Simple Linear Regression)
* [Scatterplot(산점도)](https://plotly.com/python-api-reference/plotly.express.html)에 가장 잘 맞는(Best Fit) 직선을 그려주면 그것이 ***회귀예측모델***이 된다.
* 그렇다면 회귀직선은 어떻게 만들 수 있을까? RSS(혹은 SSE)를 최소화하는 직선을 찾기.

<br/>

## Mean Absolute Error(MAE, 평균절대오차
* 예측 error의 절대갑 평균을 의미한다.
* $Error = (price - guess)$

<br/>

# Code
## 수동으로(Scikit-learn없이) MAE 구하기
```python
# error에 절대값을 취한 후 평균을 계산하면 된다.
mean_absolute_error = errors.abs().mean()

# 렉쳐노트 n211예제
x = df['GrLivArea'] # 상관관계를 보고자 하는 피쳐
y = df['SalePrice'] # 타겟값

predict = df['SalePrice'].mean() # 평균값을 기준모델로 사용
errors = predict - df['SalePrice'] # 기준모델 - 타겟
mean_absolute_error = errors.abs().mean() # MAE

print(f'예측한 주택 가격이 ${predict:,.0f}이며 절대평균에러가 ${mean_absolute_error:,.0f}임을 확인할 수 있습니다.')

==결과==
예측한 주택 가격이 $180,921이며 절대평균에러가 $57,435임을 확인할 수 있습니다.


sns.lineplot(x=x, y=predict, color='red')
sns.scatterplot(x=x, y=y, color='blue'); # 기준모델 
```

<img width="408" alt="스크린샷 2022-12-07 오후 10 35 34" src="https://user-images.githubusercontent.com/81222323/206193244-7eb0ee54-fbc6-43a2-9b28-4f5f960becee.png">

<br/>

## Pair plot
* 몇 개의 특성끼리의 상관관계를 눈으로 확인하기 좋은 방법

```python
# 렉쳐노트 n211예시
import seaborn as sns

sns.set(style='whitegrid', context='notebook')
cols = ['GrLivArea', 'LotArea','SalePrice']
sns.pairplot(df[cols], height=2);
```

<img width="421" alt="스크린샷 2022-12-07 오후 10 38 25" src="https://user-images.githubusercontent.com/81222323/206193991-47cf698a-87f3-47ef-9794-2573ff93c618.png">

<br/>

## [Simple Linear Regression (단순 선형 회귀)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
```python
# 렉쳐노트 n211예시

# 1. Scikit-Learn 라이브러리에서 사용할 예측모델 클래스 Import.
from sklearn.linear_model import LinearRegression

# 2. 예측모델 인스턴스를 만들기.
model = LinearRegression()

# 3. 데이터 분리 (특성과 타겟)
feature = ['GrLivArea']
target = ['SalePrice']

X_train = df[feature]
y_train = df[target]
model = LinearRegression()
model = LinearRegression()

# 4. 모델을 학습(fit)
model.fit(X_train, y_train)

# 5. 또 다른 데이터(하나의 샘플)을 선택해 학습한 모델을 통해 예측.
X_test = [[4000]]
y_pred = model.predict(X_test)

print(f'{X_test[0][0]} sqft GrLivArea를 가지는 주택의 예상 가격은 ${int(y_pred)} 입니다.')

==결과==
4000 sqft GrLivArea를 가지는 주택의 예상 가격은 $447090 입니다.
```
