---
title: 랜덤포레스트(Random Forests), 배깅(Bagging)
author: yun
date: 2022-11-17
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, random forests, accuracy, Bagging, Bootstrapping]
---

# 개념
## 키워드
- [ ] 부트스트랩
- [ ] 부트스트래핑(Bootstrapping)
- [x] 복원추출법
- [x] 배깅(Bagging)
- [x] Out-Of-Bag Dataset(OOB)
- [x] Out-Of-Bag Error
- [ ] 앙상블
- [ ] 
<br/>

## 용어설명
* 배깅(Bagging)
  : **B**ootstrapping the data plus using the **agg**regate to make a decision is called **"Bagging"**
  : 회귀의 경우 `평균`으로, 분류의 경우 `최빈값`으로 예측.
  
* 부트스트랩 세트 
  : 모델을 여러개 만들기 위해 원본 데이터에서 여러개의 데이터 세트를 복원 추출하여 학습을 진행하는데, 이 때 복원 추출하여 만들어지는 데이터 세트.
  
* 부트스트랩 샘플링
  : 위의 과정.  
  
* Out-Of-Bag Dataset
  : 부트스트랩 데이터 세트에 포함되지 않은 항목.
  
* 앙상블
  : 한 종류의 데이터로 여러 머신러닝 학습모델(weak base learner, 기본모델)을 만들어 그 모델들의 예측결과를 다수결이나 평균을 내어 예측하는 방법을 말헌다.
  이론적으로 기본모델 몇가지 조건을 충족하는 여러 종류의 모델을 사용할 수 있습니다. 
 
* 데이터 세트의 희소성이 증가한다.
  : '의미없는 데이터가 많아진다' 는 의미. 이로 인하여 모델의 학습이 제대로 이루어지지 않는다.
  
<br/>

## 나만의 언어로 설명



  
<br/>

## 레퍼런스
* [StatQuest: Random Forests Part 1 - Building, Using and Evaluating]([https://www.youtube.com/watch?v=7VeUPuFGJHk](https://youtu.be/J4Wdy0Wc_xQ))
  * 부트스트랩 만들기
  1. 부트스트랩 데이터 세트 생성 (원본과 같은 크기, 원래 데이터 세트에서 샘플을 무작위로 선택 `복원추출법`으로 선택!
  2. 부트스트랩 데이터 세트를 사용하여 결정트리를 생성 But, only use a random subset of variables (or colums) at each step.
  (평소와 같이 트리를 구축하지만 각 단계의 이중경계에서 변수의 임의 하위 집합만 고려)
  3. Now go back to Step 1 and repeat: Make a new bootstrapped dataset and build a tree considering a subset of variables at each step.
  4. 랜덤 포레스트의 모든 트리 아래로 데이터를 실행한 후 만든 모든 트리에 대해 반복. 그리고 어떤 옵션이 더 많은 표를 받았는지 확인합니다. 
  (See which option received **more votes**)
  
<br/>

## NOTE
* The `variety` is what makes random forests more effective than `individual decision trees`.
* 랜덤포레스트는 결정트리를 기본모델로 사용하는 앙상블 방법이라 할 수 있다.
* 기본모델인 결정트리들은 **독립적**으로 만들어진다. 
* 각각 기본모델이 랜덤으로 예측하는 성능보다 좋을 경우, 이 기본모델을 **합치는 과정에서 에러가 상쇄**되어 랜덤포레스트는 **더 정확한 예측**을 할 수 있습니다.
* $$기준모델 \ne 기본모델$$

<br/>

## 학습이 더 필요한 부분
- [ ] 
- [ ] 
- [ ] 

<br/>

# Code
## from sklearn.ensemble import RandomForestClassifier
[from sklearn.ensemble import RandomForestClassifier](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
```python
### 1 ###
# 렉쳐노트 n222 예시

# 전처리
pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True), 
    SimpleImputer(), 
    # oob_score를 true로 하면 훈련 종료 후 oob 샘플을 기반으로 평가를 수행한다.
    RandomForestClassifier(n_jobs=-1, random_state=10, oob_score=True)
)


### 2 ###
# 모델 학습
model = RandomForestClassifier(n_estimators=10, random_state=0,
                              max_features=4, oob_score=True)
model.fit(X_train, y_train)

# 평가
print("훈련 세트 정확도: {:.3f}".format(model.score(X_train, y_train)) )
print("테스트 세트 정확도: {:.3f}".format(model.score(X_test, y_test)) )
print("OOB 샘플의 정확도: {:.3f}".format(model.oob_score_) )
```
```python
==결과==
훈련 세트 정확도: 0.992
테스트 세트 정확도: 0.933
OOB 샘플의 정확도: 0.958
```
> 만약 make_pipeline을 사용하였을때 oob_score을 보고 싶으면
```python
print('oob socre :', pipe.named_steps['randomforestclassifier'].oob_score_)
```

<br/>

## 특성공학시 알아두면 좋은 코드들.

