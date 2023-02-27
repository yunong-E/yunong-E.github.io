---
title: Hash Table, Hash
author: yun
date: 2023-02-27
categories: [Blogging, Study, Python, Programming]
tags: [study, python, Computer Science, Python Programming, Hash Table, Hash]
---


# **키워드**
---
- [ ] 해시테이블
- [ ] 해시충돌
- [ ] 체이닝
- [ ] 오픈 어드레스
- [ ] 로드팩터
- [ ] SHA

<br/>

# **개념**
---
## **해시(Hash)란?**
<img width="402" alt="hash" src="https://user-images.githubusercontent.com/81222323/221544011-8b0cc4dd-0ffa-4c3c-ab0d-971dcfb637c2.png">
- `해시(Hash)`는 딕셔너리 코드와 관련되어 활용되는 개념이다.
- `해시(Hash)`는 *해시함수*를 통해 나온 `값(value)`이다.

<br/>

## **해시테이블이란?**
- `해시테이블`은 `key`를 빠르게 **저장 및 검색**할 수 있는 **테이블 형태**의 자료구조다.
- 해시테이블은 **`key`를 활용하여 `value`에 직접 접근이 가능**한 구조다.
- 해시테이블은 **검색**을 위한 역할도 하고, **딕셔너리를 위한 자료구조**의 역할도 한다.
- 해시테이블 `dict의 값`을 읽어오는데 사용되는 문자열이 `key`다.
- 해시테이블에 `저장된 데이터(value)`에 접근하려면 `key`를 알아야한다.
- 파이썬의 `딕셔너리`는 내부적으로 **해시테이블 구조**로 구현되어 있다.

<br/>

## **해시함수란?**
![다운로드 (1)](https://user-images.githubusercontent.com/81222323/221544289-f2fba144-fcc3-4fb9-9221-e84881a98b1d.png)
- 위의 그림처럼 해시함수는 **`key`를 해시테이블 내의 `버킷(=hashes =hash값)`으로 매핑**시킨다.
- 입력값의 형태는 **`다양`**하고, 출력값의 형태는 **`숫자`**다.
- 해시함수의 요구조건
  * 해시함수는 **입력값이 같다면, 동일한 출력값**을 받아야 한다.
  * 입출력값이 일정하지 않다면 적절한 해시함수가 아니다.
    * 입력값 '뮤즈'가 4를 반환한다면, 입력값 '아쿠아'는 4를 반환할 수 없다.
    * 해시함수는 특정범위 안에 있는 숫자를 반환해야 한다.
- 하나의 `해시함수`가 입력 데이터별로 다른 숫자와 매핑된다면, 그것은 **완벽한 해시함수**다.
  * 해시함수가 입력데이터에 따라 다른 숫자를 반환하게 되면 **`해시충돌`을 최소화**하는 것이기 때문.
- `해싱(Hashing)`의 목적은 **검색**이다. (앞서 배운 정렬 알고리즘과는 다르다.)
- `해싱(Hashing)`의 장점은 데이터의 양에 영향을 덜 받으며 성능이 빠르다. (`key`를 통해 `value`를 검색하기 때문.)
- `해싱(Hashing)`은 쉽게 말해서 다 흩뜨려 놓고, `key`와 매칭되는 `value`을 검색하는 과정이다.
- `해시함수`는 여러 키를 분할하기 위해 `key`를 `hash값(정수값)`으로 매칭시키는 역할을 한다.

<br/>

## **해시성능**
- 해시테이블 자체는 충돌을 해결해주지 않는다.
- **`O(1)` 시간복잡도 안에** 검색, 삽입, 삭제를 할 수 있다.
  - `상수시간(O(1))`은 **해시테이블의 사이즈에 관계없이 동일한 양의 계산**을 다룬다.
  - 해시충돌로 인해 모든 `bucket`의 `value`를 찾아야 하는 경우(반복문)도 있다.
- 만약 해시테이블이 하나의 요소를 갖고 있다면, 해시테이블 인덱스 갯수에 관계없이 프로그램 수행시간이 비슷하다.
  - 왜그럴까? 바로 해시함수 때문이다
- 검색/삽입/삭제 무엇을 하든지 해시함수는 `key`를 통해 저장된 `value`에 연관된 `index`를 반환한다. (즉, `key`와 `index`가 매칭되어야 함.)

<br/>

# **Code**
---
- 정수형에서 문자열로 변환하기 위해 해시함수는 문자열에 해당하는 개별적인 단어를 활용한다.
- 아래 예시는 파이썬에서 `.encode()`메소드를 활용해 문자열에서 바이트코드로 인코드하는 것이다.
- 인코딩 후, 정수형(104, 101 ...)은 각 단어('h', 'o' ...)를 나타낸다.

```python
# 인코딩 예제
bytes_representation = "hello".encode()

for byte in bytes_representation:
    print(byte)
    
 == 결과 == 
104
101
108
108
111
```

- 여러개의 정수들을 하나의 문자열로 변환하기.

```python
# 정수값의 합 반환
bytes_representation = "hello".encode()

sum = 0
for byte in bytes_representation:
    sum += byte

print(sum)

== 결과 == 
532
```

- 해시함수를 만들고 활용하기.

```python
def my_hashing_func(str, list_size):
    bytes_representation = str.encode()    
    sum = 0
    
    for byte in bytes_representation:
        sum += byte

    print('sum:', sum)
    print('list_size', list_size)
    print('sum % list_size:', sum % list_size)
    return sum % list_size
    
    
# 위의 my_hashing_func이라는 해시함수를 활용하여 아래처럼 값을 확인할 수 있다.
my_list = [None] * 5 # 5개의 빈 슬롯이 들어가는 리스트 초기화.

# 해시테이블 값을 입력
my_list[my_hashing_func("aqua", len(my_list))] = "#00FFFF"

# 해시테이블 있는 값을 출력
print(my_list[my_hashing_func("aqua", len(my_list))])

# 전체 해시테이블 출력
print(my_list)

== 결과 ==
sum: 424
list_size 5
sum % list_size: 4
sum: 424
list_size 5
sum % list_size: 4
#00FFFF
[None, None, None, None, '#00FFFF']
```
<br/>

# **레퍼런스**
---
* [왜 hash(-1) == hash(-2) 일까?](https://omairmajid.com/posts/2021-07-16-why-is-hash-in-python/)
* [PEP 456 - 안전하고 교환 가능한 해시 알고리즘](https://peps.python.org/pep-0456/)
