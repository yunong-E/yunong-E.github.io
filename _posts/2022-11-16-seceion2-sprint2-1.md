---
title: Decision Trees(의사결정나무, 결정트리)
author: yun
date: 2022-11-16
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, decision trees, classification]
---

# 개념
## 키워드
- [x] [Decision Trees(의사결정나무, 결정트리)](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [x] 지니불순도(Gini Impurity or Gini Index)
- [x] 엔트로피(Entropy)
- [x] 정보획득(Information Gain)
- [x] [sklearn.pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [ ] 재귀
- [x] SimpleImputer

<br/>

## 용어설명
* Root Node(The Root)
  : 트리의 맨위

* Internal Nodes(Nodes)
  : 질문이나 말단의 정답.

* Leaf Nodes(Leaves)

* Impurity(불순도)
  : 여러 범주가 섞여있는 정도를 말한다.
* Gini(지니지수)
  : 지니지수는 낮을 수록 좋다. 불순물 이니까.
  
* 정보획득(Information Gain)
  : 특정한 특성을 사용해 분할했을 때의 엔트로피의 감소량을 의미한다.<br/>
  $$IG(T,a)= H(T)-H(T|a)$$ $$= 분할 전 노드 불순도 - 분할 후 자식노드 들의 불순도$$
  
<br/>

## 나만의 언어로 설명
* Impurity(불순도)
  : MBTI 중 P와 J의 비율이 49:51이면 불순도가 높은 것. 10:90이면 불순도가 낮은 것 = 순수도(purity)는 높은 것.

* 정보획득(Information Gain)
  : 자식들아... 잘해라(?). 아랫 물이 맑아야 윗 물이 맑지(?)
  
* Decision Trees(의사결정나무, 결정트리)
  : 다양한 크기의 여러가지 입자(raw data)를 서로 크기가 다른 여러개의 뜰채(특성)를 활용하여 분류하는 것으로 이해.
  : ex) 바다모래, 흙, 돌멩이, 자갈, 금 등이 섞여있는 모래주머니(raw data)에서 각각의 그룹끼리(클래스) 걸러내고자 할 때.
  : 올바른 분류를 위해서는 우리가 사용 할 **뜰채의 갯수(깊이)**와 **뜰채의 크기(특성)**가 중요하다고 생각합니다. 많은 뜰채를 사용할 수록 과적합 발생 확률 증가,
  : 개인적인 견해이지만, 결과(모델) 뿐만 아니라 걸러내는 과정 속에서도 중요한 의미 혹은 정보를 습득할 수 있지 않을까? 하는 생각을 해봤습니다.  
  
<br/>

## 레퍼런스
* [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
  * 환자를 나누기에 가장 좋은 체중은?
  1. 체중별로 환자 정렬 (오름차순)
  2. 인접한 모든 환자의 평균 체중 계산
  3. 각 평균 중량에 대한 불순물 값 계산
  4. 가장 작은 불순물 값을 가진 체중을 기준으로 나누기.

<br/>

* [Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://youtu.be/LDRbO9a6XPU)
  * 효과적인 트리를 구축하는 요령 <br/>
  a. 언제, 어떤 질문을 해야하는지 이해하는 것. <br/>
  b. 그러기 위해서는 질문이 레이블을 분리하는데 얼마나 도움이 되는지 ***정량화*** 해야한다. <br/>
  c. `지니지수`와 `정보획득(Information Gain)` 개념을 시용하자.  <br/>
  
  * 더이상 물어볼 질문이 없을 때까지 데이터를 분할하고 ***마지막 지점에서 잎사귀(Leaf)를 추가***한다. 이를 구현하기 위해서는, <br/>
  a. 데이터에 대해 어떤 유형의 질문을 할 수 있는지 이해할 것. <br/>
  b. 언제, 어떤 질문을 할 것인지 결정하는 방법을 이해할 것. <br/>
  c. ***가장 좋은 질문***은 ***불확실성을 가장 많이 줄이는 질문.***

<br/>

## NOTE
* classification can be categories or numeric (정렬 or 숫자 범주에 있을 수 있습니다.)
* 결정트리모델은 선형회귀처럼 특성을 해석하기에 좋다. 다른 모델보다 성능은 조금 떨어지지만 해석하기 좋아서 많이 사용된다.
* 회귀문제, 분류문제 모두 적용 가능하다.
* sample에 민감해서 트리구조가 잘 바뀌는 단점도 있다. 이에 해석도 바뀔 수 있다.
* 추후 배울 앙상블 기법의 기초가 되므로 이론에 대한 이해를 확실히 해둘 것.
* 회귀모델에 비해 전처리 과정에서 덜 신경써도 되는 부분이 있다. 예를들어. 데이터 특성들의 스케일을 굳이 맞춰줄 필요가 없다. ***왜?***
  * 결정트리에서는 StandardScaler()는 도움이 되지 않는다. 중요한 것은 `순서(대소관계)`이기 때문에. ***아하***
* 좋은 질문을 거쳐 분할된 데이터 세트는 지니 불순도 값이 작다는 것을 알 수 있다. -> 좋은 질문을 도대체 어떻게 하는가?

<br/>

# Code
## a. for문 if문 한번에 작성하기 (list comprehension)
```python
# 렉쳐노트 n221 예시

# 1-1
for col in df.columns:
  if 'behavioral' in col:
    behaviorals.append(col)

# 1-2
behaviorals = [col for col in df.columns if 'behavioral' in col] 


# 2-1
# for문과 if문을 각각 작성했을때, 실행시간 2.31ms
mylist = [3, 2, 6, 7]
answer = []
for number in mylist:
  if number % 2 == 0:
    answer.append(number**2) # 들여쓰기를 두 번 함

# 2-1
# list comprehension일 때, 실행시간: 1.76ms
mylist = [3, 2, 6, 7]
answer = [number**2 for number in mylist if number % 2 == 0]
```

<br/>

## b. 특성공학시 알아두면 좋은 코드들.
```python
# 렉쳐노트 n221 예시
###### 높은 카디널리티를 가지는 특성을 제거합니다. ######

    # 1. 숫자형 오브젝트형 컬럼 추출
    selected_cols = df.select_dtypes(include=['number', 'object'])
    
    # 2. nunique() 고윳값들의 '갯수'를 반환 (특성별 카디널리티 리스트)
    labels = selected_cols.nunique()
    
    # 3. index를 list 형태로 반환? (카디널리티가 30보다 작은 특성만 선택)
    selected_features = labels[labels <= 30].index.tolist()
    
    # 4. 저장
    df = df[selected_features]
```

<br/>

## c. 파이프라인(Pipeline)
* [from sklearn.pipeline import make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* 코드가 보다 간결해졌음을 알 수 있다. (기존 코드는 렉쳐노트 참조要) 
<br/>

```python
# 렉쳐노트 n221 예시
pipe = make_pipeline(
    OneHotEncoder(), 
    SimpleImputer(), 
    StandardScaler(), 
    
    # n_jobs=-1 입력시 모든 코어 실행.
    # 코랩에서의 CPU core개수는 2개.
    # n_jobs=-1: Use all available cores on the machine이기 때문에 여기서의 n_jobs=2와 같다.
    LogisticRegression(n_jobs=-1)
)

pipe.fit(X_train, y_train)

print('검증세트 정확도', pipe.score(X_val, y_val))

y_pred = pipe.predict(X_test)
```

<br/>

* `named_steps` 속성을 사용해서 파이프라인의 각 스텝에 접근이 가능하다.


```python
pipe.named_steps
```


결과값:
> {'onehotencoder': OneHotEncoder(cols=['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc', 'agegrp', 'census_msa']),
> 'simpleimputer': SimpleImputer(),
> 'standardscaler': StandardScaler(),
> 'logisticregression': LogisticRegression(n_jobs=-1)}

<br/>

* `named_steps`은 유사 딕셔너리 객체(dictionary-like object)로 파이프라인 내 과정에 접근 가능하도록 한다.


```python
# 모델의 회귀계수 프린트
# 기존에는 밖에서 프린트...?

model_lr = pipe.named_steps['logisticregression']
enc = pipe.named_steps['onehotencoder']
encoded_columns = enc.transform(X_val).columns
coefficients = pd.Series(model_lr.coef_[0], encoded_columns)
```

<br/>

## d. sklearn.tree.DecisionTreeClassifier
* [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* 파이프라인을 사용하면 위에서 본 코드에서 단지 분류기만 바꾸어 주면 된다.
* 하이퍼 파라미터 (더 있으니 API 확인要)
  * `criterion`: 분할 품질을 측정하는 기능 {"gini", "엔트로피", "log_loss"}, default="gini"
  * `min_samples_split`: 내부 노드를 분할하는 데 필요한 최소 샘플 수 [int or float, default=2, 범위값:20~100] ?? (꽃게책)
  * `max_depth`: 트리의 최대 깊이 [int, default=None, 범위값: 5~30]
  * `min_samples_leaf`: 리프 노드에 있어야 하는 최소 샘플 수 [int or float, default=1]
  * `ccp_alpha`: Minimal Cost-Complexity Pruning에 사용되는 복잡성 매개변수. `GridSerchCV`를 사용해 최적의 값을 찾을 수 있다. [음수가 아닌 float, default=0.0]
<br/>

```python
from sklearn.tree import DecisionTreeClassifier

pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),  
    SimpleImputer(), 
    DecisionTreeClassifier(random_state=2, 
                           criterion='entropy', 
                           min_samples_leaf=11, 
                           min_samples_split = 51, # 20~100
                           max_depth=20) # 5~30
)

pipe.fit(X_train, y_train)
print('훈련 정확도: ', pipe.score(X_train, y_train))
print('검증 정확도: ', pipe.score(X_val, y_val))
```
결과값:
> 훈련 정확도:  0.8333185066571775 <br/>
> 검증 정확도:  0.8312181235915075

<br/>

## e. F1 Score(F1 스코어)
```python
from sklearn.metrics import f1_score

# 검증세트에 대한  F1-score
pred = pipe.predict(X_val)
f1 = f1_score(y_val, pred)
f1
```
결과값:
> 0.5750970438936996

<br/>

## f. from sklearn.impute import SimpleImputer
* [from sklearn.impute import SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
* `SimpleImputer` 클래스와 `fit_transform` 클래스 메소드를 활용하여 결측데이터를 채워넣을 수 있다.


```python
from sklearn.impute import SimpleImputer

# most_frequent : 최빈값, mean : 평균값, median : 중앙값
# default='mean'
imputer = SimpleImputer(strategy="most_frequent")

# 데이터 프레임으로 만들기.
df = pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
```
