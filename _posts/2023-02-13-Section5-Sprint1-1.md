---
title: 컴퓨터 공학 기본
author: yun
date: 2023-02-13
categories: [Blogging, Study, Python, Programming]
tags: [study, python, Computer Science, Python Programming]
---


# **키워드**
- [ ] 자료구조
- [ ] 알고리즘
- [ ] 메타문자
- [ ] 무한루프

<br/>

# **메서드**
`re.compile('정규식')` : 패턴을 지정해준다.

`.match` : 시작부터 일치하는지 검사한다.


```python
p = re.compile('[a-z]+')
m = p.match('3 python')
```


`.search` : 어디에든 있는지 찾아준다.

`.group()` : 정규표현식과 일치하는 문자열 변환

`.span()` :  정규표현식과 일치하는 문자열의 (시작위치, 끝위치) 튜플반환


`rjust(width, [fillchar])` : 오른쪽으로 정렬
`ljust(width, [fillchar])` : 왼쪽으로 정렬
`.split` : 문자 나누기

<br/>

# **반복문과 조건문**
## **for문**
```python
for 변수 in


```
## **while문**
```python
while(조건문)
```

## 이중반복문



## **if-else문**
```python
if 조건식:
  조건식의 결과가 참(True)일 때만 실행되는 명령문
else:
  조건식의 결과가 거짓(False)일 때만 실행되는 명령문
```

## if-elif-else문
```
elif 조건식2 : 
  
```
