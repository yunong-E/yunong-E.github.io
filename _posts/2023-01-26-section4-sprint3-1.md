---
title: 합성곱 신경망(Convolutional Neural Network)과 전이 학습(Transfer Learning)
author: yun
date: 2023-01-26
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, convolutional neural network, transfer learning]
---



# 개념
## CNN(합성곱 신경망) 의 장점
* 입력된 데이터의 공간적인 특성을 보존하며 학습할 수 있다.


## 합성곱
* 컨볼루션에서의 가중치 수는 **커널크기의 제곱 × 커널의 개수 × 채널 수**에 비례하게 된다.

```
model = Sequential()
model.add(Conv2D(3, (5, 5), activation='relu', input_shape=(100, 100, 3))) # ---- (A)
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
```

따라서 (A)에서 학습될 가중치의 개수는 225개 이다. ($5^2$ * 3 * 3)


## 풀링(Pooling)
* feature map의 가로, 세로 방향의 공간을 줄일 수 있다.
* 풀링의 방법에는 최대 풀링(Max Pooling), 평균 풀링(Average Pooling)이 있다.
* 풀링에는 학습해야 할 가중치가 **없다**.



# Q&A

풀링은 꼭 진행해야하는 과정인가요? : 아닙니다.

필터는 가중치 이다.

지막에 softmax 10 값은 데이터셋의 특성들이 10개로 정리될 수 있는 `cifar10`이라서 10으로 설정된 건가요? : 네 맞습니다.


** 전이학습

사전학습된 모델을 가져와서 우리가 풀고자 하는 문제에 적용시키는 것.

** vgg

1. 3x3 필터 사용. 계산하는 양이 적다는 이점이 있다.
2. 채널이란? RGB: 3, 흑백: 1
3. 왜 224x224x3 에서  224x224x64가 될까요? 필터링을 거쳐서.
4. 풀링 레이어는 채널이 변하지 않는다.
5. 필터는 채널 수가 됩니다. (디스커션 참조)
 

**

**
