---
title: 데이터 랭글링(Data Wrangling)
author: yun
date: 2022-11-24
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, data wrangling]
---

# 목표


# 키워드
- [ ] 데이터 랭글링(Data Wrangling)
- [ ] 분할정복

<br/>

1. 예측문제 정의 - 예측하고자 하는 문제를 보다 간단히 만들어 볼 것.
2. 1에서 나온 간단한 질문들에 대한 답변을 하기 위해 Data Wrangling을 해볼 것.




# Code
## Merge, Join

<br/>

## [Group by](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#computations-descriptive-stats)
* `.any()`: Return `True` if ***any value*** in the group is truthful, else `False`.
* `.all()`: Return `True` if ***all values*** in the group are truthful, else `False`.

<br/>

## apply
* `.apply(list)`

<br/>

## 상위 5개 항목 추출하기 value_counts() 활용
```python
# 상위 5개의 구매 제품 확인하기
# 이제는 잘 할 수 있죠?
top5_items = prior['product_id'].value_counts()[:5]
```
<br/>
* 복습: `value_counts(normalize=True)` 상대빈도를 표시해주는 파라미터. 이제 알죠?

<br/>

## set_index()
* `index`를 재지정하는 코드.
* 이런식으로도 활용할 수 있구나.


```python

products.set_index('product_id').loc[top5_index]

==결과==
	           product_name	aisle_id	department_id
product_id			
24852	Banana	24	4
13176	Bag of Organic Bananas	24	4
21137	Organic Strawberries	24	4
21903	Organic Baby Spinach	123	4
47209	Organic Hass Avocado	24	4
```

<br/>

## set1.isdisjoint(set2)
* 두 집합이 서로소인지 확인. Boolean값으로 반환을 한다.
* 서로소인 경우: `True`, 아닌 경우: `False`
* **서로소란?**: 서로소(disjoint)는 공통으로 포함하는 원소가 없는 두 집합의 관계다.
* [참고 사이트](https://hongsusoo.github.io/python%20basic/grammar12/)
<br/>

```python
# 렉처노트 n232 예시

# 테스트 데이터에 있는 고객이 훈련데이터에 포함되어있는지 확인을 함.
set(orders[orders['eval_set']=='test']['user_id'])\
    .isdisjoint(set(orders[orders['eval_set']=='train']['user_id']))
    
==결과==
True
```


- [ ] (sol 3).부터 복습요망..
                         
