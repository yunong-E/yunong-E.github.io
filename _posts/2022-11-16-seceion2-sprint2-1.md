---
title: Decision Trees(의사결정나무, 결정트리)
author: yun
date: 2022-11-16
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, decision trees, classification]
---

# 키워드
- [ ] Decision Trees(의사결정나무, 결정트리)
- [ ] 지니지수(Gini)
- [ ] 정보획득(Information Gain)
- [ ] 
- [ ]

# 용어설명
* Root Node(The Root)
  : 트리의 맨위
* Internal Nodes(Nodes)
* Leaf Nodes(Leaves)
* Impure
* Impurity(불순물)
* Gini(지니지수)
  : 지니지수는 낮을 수록 좋다. 불순물 이니까.

# 레퍼런스
* [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
  * 환자를 나누기에 가장 좋은 체중은?
  1. 체중별로 환자 정렬 (오름차순)
  2. 인접한 모든 환자의 평균 체중 계산
  3. 각 평균 중량에 대한 불순물 값 계산
  4. 가장 작은 불순물 값을 가진 체중을 기준으로 나누기.

* [Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://youtu.be/LDRbO9a6XPU)
  * 효과적인 트리를 구축하는 요령
  a. 언제, 어떤 질문을 해야하는지 이해하는 것.
  b. 그러기 위해서는 질문이 레이블을 분리하는데 얼마나 도움이 되는지 ***정량화*** 해야한다.
  c. `지니지수`와 `정보획득(Information Gain)` 개념을 시용하자. 
  
  * 더이상 물어볼 질문이 없을 때까지 데이터를 분할하고 ***마지막 지점에서 잎사귀(Leaf)를 추가***한다. 이를 구현하기 위해서는,
  a. 데이터에 대해 어떤 유형의 질문을 할 수 있는지 이해할 것.
  b. 언제, 어떤 질문을 할 것인지 결정하는 방법을 이해할 것.
  c. ***가장 좋은 질문***은 ***불확실성을 가장 많이 줄이는 질문.***


# NOTE
* classification can be categories or numeric (정렬 혹으 숫자는 범주에 있을 수 있습니다.)
