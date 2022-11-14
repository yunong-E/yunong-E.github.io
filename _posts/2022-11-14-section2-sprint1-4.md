---
title: 로지스틱 회귀, 훈련/검증/테스트 세트
author: yun
date: 2022-11-14
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, logistic regression, classification]
---

## 키워드
- [ ] 로지스틱 회귀(Logistic regression)
- [ ] 검증 데이터세트
- [ ] totes useless


#### 검증데이터의 용도와 테스트 데이터의 용도?:
* 테스트는 성능이 충분히 좋은지 확인하는 것에 관한 모든 것.
* 검증은 보상을 허용하는 것. 기계학습 프로젝트에서 둘 다 필요한 최종 시험에 도달하기 전에 성능을 향상시키는 방법.
* validation is allowing for redemption it is a way to imporve performance before you get to that final exam you do need both in machine learning.
* [Machine Learning: Validation vs Testing](https://www.youtube.com/watch?v=pGlQLMPI46g)


#### 테스트 데이터로 예측한 후 모델을 수정해 또 테스트 데이터로 예측하는 행동을 피해야하는 이유?:
* Traning set: construct classifier
  * NB: count frequencies, DT: pick ttributes to split on
* Validation set: pick algorithm + knob settings
  * pick best-performing algorithm (NB vs. DT vs. ...)
  * fine-tune knobs (tree depth, k in KNN, c in SVM ...) 
* Testing set: estimate future error rate
  * never report best of many runs
  * run only once, or report results of every run
* Sprit ***randomly*** to avoid bias
* [Overfitting 4: training, validation, testing](https://www.youtube.com/watch?v=4wGquWG-vGw)


#### 로지스틱회귀와 선형회귀의 차이점?:
* Logistic regression predicts whether someting is ***True*** or ***False***, instead of predicting something continuous like ***size***.
* also, instead of fitting a line to the data, logistic resgression fits an "S" shaped "logistic function"
* It's usually used for ***classification.***
* Just like linear regression, logistic regression can work with continuous data(like **weight** and **age**) and discrete data(like **genotype** and **astrological sign**).
* ***totes useless***: 도움이 되지 않는다는 통계적 전문용어. 죽, 연구에서 시간과 공간을 절약할 수 있다는 의미.
* 로지스틱 회귀는 확률을 제공하고 연속 및 이산 측정을 사용하여 새 샘플을 분류하는 기능을 사용하여 사용되는 기계학습 방법.
* 선형회귀와 로지스틱회귀의 한 하지 큰 차이점은 ***선이 데이터에 적합하는 방식***.
  * 선형회귀를 사용하면 최소 제곡법을 사용하여 선에 맞춘다. 잔차의 제곱합을 최소화하는 선을 찾는다. 잔차를 사용해 $$r^2$$를 계산한다.
  * 로지스틱 회귀에는 잔차와 동일한 개념이 없으므로, 최소 제곱법을 사용할 수 없고 $$r^2$$을 계산할 수도 없다. 대신 ***최대 가능성(우도, Maximum likelihood)***라는 것을 사용한다.
* [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)


#### 시계열데이터(time series)를 훈련/검증/테스트 세트로 나눌 때 주의해야할 점?:
* 기본 아이디어는 다음과 같습니다.
  * 훈련 세트는 주어진 모델을 훈련하는 데 사용됩니다.
  * 검증 세트는 모델 중에서 선택하는 데 사용됩니다.
  * 테스트 세트는 수행한 방법을 알려줍니다. 많은 다른 모델을 시도했다면 우연히 검증 세트에서 잘 작동하는 모델을 얻을 수 있으며 테스트 세트가 있으면 그렇지 않은지 확인하는 데 도움이 됩니다.
* 검증 및 테스트 세트의 주요 속성은 ***미래에 보게 될 새로운 데이터를*** 대표해야 한다는 것
* [좋은 유효성 검사 세트를 만드는 방법(및 이유)](https://www.fast.ai/posts/2017-11-13-validation-sets.html)


***
## 학습 시작
### 로지스틱 회귀: 
  이름은 회귀이지만 실제로는 분류모델을 푸는 지도학습 모델. 로지스틱 회귀모델은 샘플이 특정한 범주에 속할 확률을 추정하는데 많이 사용된다.
