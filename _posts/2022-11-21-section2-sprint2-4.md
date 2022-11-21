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
  2. 과적합 방지를 위해?

# 개념
# 키워드
- [ ] Hold-out Cross-validation
- [ ] k-fold cross-validation
- [ ] Leave One Out Cross Validation
- [ ] 최적화
- [ ] 검증곡선(Validation curve)


# 용어설명



# 나만의 언어로 설명해보기
* Hold-out Cross-validation
  : k-fold 교차검증일 알기 전까지 내가 했던 데이터세트 분리 방법. (훈련/검증/테스트)
  
  
  
# 학습이 더 필요한 부분

# NOTE
* 교차검증은 시계열(time series) 데이터에는 적합하지 않음.
* Hold-out Cross-validation의 단점도 있다.
  1. 훈련세트의 크기가 모델학습에 충분하지 않을 경우 문제가 될 수 있다. (= 학습할 데이터가 많으면 문제 없다.)
  2. Validation set(검증세트) 크기가 충분히 크지 않다면 예측 성능에 대한 추정이 부정확하다.
* random forest model에서 주요 parameter는
  1. n_estimators : 생성할 tree의 개수와
  2. max_features : 최대 선택할 특성의 수입니다.



# Code
## from category_encoders import TargetEncoder
