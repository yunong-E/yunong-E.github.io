---
title: \[AIB] 선형대수(Linear Algebra), 벡터, Span, Basis, Rank
author: yun
date: 2022-10-27
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, linear algebra, vector, span, basis, rank]
---

# 자료구조
## List
## Array


# 기본 선형대수(Linear Algebra)
***
## Scalar
* 하나의 숫자(실수)를 나타냅니다.
* 변수에 저장하여 표기할 수 있습니다.
* 양수, 음수 모두 가능합니다.

## Vector
* 순서를 갖는 1차원 형태의 배열로 List 또는 Array로 나타냅니다.
* 성분의 개수는 **벡터의 차원**을 의미합니다.
> 성분의 개수로 표현하는 벡터의 차원은 list나 array를 나타낼 때의 1차원 배열과는 *다른 의미* 입니다. <br/>

a. **벡터의 크기**
* Norm 혹은 length, Magnitude라고 합니다.
* 벡터의 길이를 나타냅니다. 따라서 음수가 될 수 없습니다.
* 벡터의 모든 성분이 0이면 벡터의 크기도 0입니다.
* $||v||$와 같이 표기합니다.
* 피타고라스 정리를 사용하여 구할 수 있습니다.
> $v = [a, b, c, \cdots]$ <br/>
>
> $||v|| = \sqrt{a^2 + b^2 + c^2 + \cdots}$ <br/>

b. **벡터의 내적**
* `Dot Pruduct`라고 합니다.
* 두 벡터에 대해서 서로 대응하는 각각의 성분을 ***곱한뒤 모두 합하여*** 구합니다. 이때 두 벡터의 차원이 ***같아야*** 합니다.
* `np.dot()`을 사용해 구할 수 있습니다.
* 벡터를 내적한 값은 *스칼라*입니다.
> $v_1 = [a_1, a_2, a_3, \cdots]$ <br/>
>
> $v_2 = [b_1, b_2, b_3, \cdots]$ <br/>
>
> $v_1 \cdot v_2 = a_1b_1 + a_2b_2 + a_3b_3 + \cdots$ <br/> 

c. **벡터의 직교(Orthogonality)와 그 조건**
두 벡터의 내적이 $0$이면 두 벡터는 서로 수직입니다.
```python
vec_x = [1,3]
vec_y = [-3,1]

np.dot(vec_x, vec_y)

# 결과값: 0
```
<br/>

d. **단위 벡터(Unit Vector)**
* 길이가 $1$인 벡터입니다.
* 모든 벡터는 단위 벡터의 선형 결합으로 표기할 수 있습니다.
> $v = [2,5] = [2,0] + [0,5] = 2[1,0] + 5[0,1] = 2\hat{i} + 5\hat{j}$
<br/>

## Matrix
* 수 또는 변수를 ()안에 행과 열로 배열한 것입니다.
* 2차원 형태의 array 또는 list로 나타냅니다.
* 행과 열의 개수는 **매트릭스의 차원**을 의미합니다. 이는 `.shape`을 통해 확인할 수 있습니다.
* 두 매트릭스가 일치하려면 ***차원과 성분이 동일***해야 합니다.
```python
# 1차원 array : 벡터

array_1d = np.array([1,2,3,4,5])
array_1d # 결과값: array([1, 2, 3, 4, 5])

# .ndim을 사용하여 배열의 차원 확인
array_1d.ndim # 결과값: 1

# .shape을 사용하여 벡터의 차원 확인
array_1d.shape #결과값: (5, )
```
<br/>

> `array_1d`는 *1차원 벡터*로 행과 열로 구분되지 않습니다. <br>
> 따라서 콤마 뒤가 공란이며 콤마 앞의 수는 벡터의 차원, 즉 성분의 개수를 나타냅니다. 
<br>







## Span
## Basis
## Rank
