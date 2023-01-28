---
title: 합성곱 신경망(Convolutional Neural Network)과 전이 학습(Transfer Learning)
author: yun
date: 2023-01-26
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, convolutional neural network, transfer learning]
---


# 키워드
- [x] 합성곱
- [ ] 필터 (커널)
- [x] 패딩
- [x] stride
- [ ] Feature map

<br/>

# 개념
## **CNN(Convolutional Neural Network, 합성곱 신경망) 의 장점**
![yVw7una](https://user-images.githubusercontent.com/81222323/214837753-c63b23b6-10b0-4aab-9e1b-8ca91cda65ce.png){: width="500" height="400"}
* 입력된 데이터(이미지)의 공간적인 특성을 보존하며 학습할 수 있다.
* 특징이 추출되는 부분으로 합성곱 층(Convolution Layer)과 풀링 층(Pooling Layer)이 있다.

<br/>

## **합성곱(Convolution)**
![1_MrGSULUtkXc0Ou07QouV8A](https://user-images.githubusercontent.com/81222323/214838182-efec7be0-d19c-4bce-b040-3b6128c22ad9.gif)
* 합성곱 층에서는 `합성곱 필터(Convolution Filter)`가 슬라이딩하며 이미지의 부분적인 특징을 읽어 나간다.
* `Convolution`에서의 가중치 수는 **커널크기의 제곱 × 커널의 개수 × 채널 수**에 비례하게 된다.


```python
model = Sequential()
model.add(Conv2D(3, (5, 5), activation='relu', input_shape=(100, 100, 3))) # ---- (A)
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
```

따라서 (A)에서 학습될 가중치의 개수는 225개 이다. ($5^2$ * 3 * 3)

<br/>

## **패딩(Padding)**
![GRDbmHF](https://user-images.githubusercontent.com/81222323/214838691-263f6a5c-7ba1-46a2-ac7d-ccfb2a955d5d.gif)
* 이미지 외부를 특정한 값으로 둘러싸서 처리해주는 방식.
* Output 즉, `Feature Map`의 크기롤 조절하고 실제 이미지 값을 충분히 활용하기 위해 사용됨.
* 보통은 '0'으로 둘러싸주는 `Zero-Padding`이 가장 많이 사용됨.

<br/>

## **Stride**
* 슬라이딩시에 몇 칸씩 건너뛸지를 나타낸다.
* Defalut 값은 `1`이다.
* $$ N_{\text{out}} = \bigg[\frac{N_{\text{in}} + 2p - k}{s}\bigg] + 1 $$
$N_{\text{in}}$ : 입력되는 이미지의 크기(=피처 수) <br/>
$N_{\text{out}}$ : 출력되는 이미지의 크기(=피처 수) <br/>
$k$ : 합성곱에 사용되는 커널(=필터)의 크기 <br/>
$p$ : 합성곱에 적용한 패딩 값 <br/>
$s$ : 합성곱에 적용한 스트라이드 값

<br/>

## **풀링(Pooling)**
* `feature map`의 가로, 세로 방향의 공간을 줄일 수 있다.
* 풀링의 방법에는 `최대 풀링(Max Pooling)`, `평균 풀링(Average Pooling)`이 있다.
* 풀링에는 학습해야 할 가중치가 **없다**.

<br/>

## **전이학습**
* 사전학습 된 모델을 가져와서 우리가 풀고자 하는 문제에 적용시키는 것.
* 대량의 데이터로 사전 학습한 모델의 **가중치**를 가져와서 사용한다.
* 사전 학습에서 학습된 가중치는 보통 *학습되지 않도록 **고정*** 한다.
* 사전 학습에서 학습된 가중치를 가져오고 고정할 경우, 모델의 학습 속도가 빠르다.

<br/>

## **ResNet**
* 구조적 특징 `Residual Connection(=Skipped Connection)`을 적용했다.
  * Residual Connection(=Skipped Connection)
    * 층을 거친 데이터의 출력(F(x))에 거치지 않은 출력(x)을 더해 준다.
    * 층을 깊게 쌓아 발생하는 `기울기 소실 문제`를 어느정도 해결할 수 있다.
    * (A)를 적용하기 위해서는 층을 거치지 않은 출력(x)과 층을 거친 데이터의 출력(F(x))의 차원이 같아야 한다.


## **이미지 증강(Image Augmentation)**
* 자르기
* 채도 변경
* 회전
* 좌우 





# **Q&A**

1. 풀링은 꼭 진행해야하는 과정인가요? : 아닙니다.

2. 필터는 가중치 이다.

3. 지막에 softmax 10 값은 데이터셋의 특성들이 10개로 정리될 수 있는 `cifar10`이라서 10으로 설정된 건가요? : 네 맞습니다.

4. 필터를 여러개 사용하는 이유? : 각각의 필터가 서로 다른 특징을 잡아내기 때문이다.

5. GlobalAveragePooling2D을 써야하는 경우와 flattne을 써야하는 경우가 다른가요? : 비슷합니다. case by case 입니다.

6. 입력값 (28,28,3)중에 입력 채널 3개마다 커널이 존재하는거죠? : 네 맞습니다.

7. 필터가 랜덤한 초기 가중치를 가지고 있기에 시작점이 달라서 다른 특징을 뽑아내는거죠? : 맞습니다. 정확한 설명입니다.

1x1 convolutional layers 를 사용하면 비선형성의 장점이 있다고 하는데, 파라미터가 줄어드는데 어떻게 비선형성이 더 향상되나요? : Layer도 결국에는...
비선형성이 추가되면 추가될 수록 과적합이 될 우려가 있습니다.




## **전이학습**



## **vgg**

1. 3x3 필터 사용. 계산하는 양이 적다는 이점이 있다.
2. 채널이란? RGB: 3, 흑백: 1
3. 왜 224x224x3 에서  224x224x64가 될까요? 필터링을 거쳐서.
4. 풀링 레이어는 채널이 변하지 않는다.
5. 필터는 채널 수가 됩니다. (디스커션 참조)
 



**
