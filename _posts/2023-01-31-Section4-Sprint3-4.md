---
title: GAN(Generative Adversarial Networks, 생성적 적대 신경망)
author: yun
date: 2023-01-31
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, GAN]
---


# **키워드**
- [ ] GAN(Generative Adversarial Networks, 생성적 적대 신경망)
- [ ] 생성자(Generator)
- [ ] 판별자(Discriminator)
- [ ] CycleGAN
- [ ] StyleGAN
- [ ] Mode Collapse

<br/><br/>

# **개념**
## **1. GAN(Generative Adversarial Networks, 생성적 적대 신경망)**
<img width="721" alt="image" src="https://user-images.githubusercontent.com/81222323/215648718-56481c82-f1cf-4b74-8e6b-017161028e04.png">
* `Noise`에서 이미지를 시작하면서 시작.
* GAN의 목표는 실제와 유사한 이미지 등을 만들어 내는 것이다. (e.g. 딥페이크)
* Generative라는 말은 '생성적인, 생산하는' 이라는 뜻으로, 이 모델을 이용해서 이미지 등을 생성할 수 있기 때문에 해당 표현이 사용되었다.
* Adversarial은 '적대적인'이라는 뜻으로, 이 모델에서 길항 작용을 하는 두 네트워크가 서로 경쟁하면서 발전하기 때문에 해당 표현을 사용한다.
* Generator(생성자)의 목표는 Random noise를 사용해서 Discriminator(판별자)를 속일 수 있는 가짜 이미지를 만들어 내는 것이다.
* GAN의 손실 함수는 Generator(생성자)의 손실과 Discriminator(판별자)의 손실을 모두 고려한다.
* GAN의 경우는 `Checkpoint` 설정을 해줘야 한다. 이미지가 변하기 때문(?)
* `Mode Collapse` 문제가 야기될 수 있다. 


## **1-1. Generator(생성자)**
<img width="742" alt="gan" src="https://user-images.githubusercontent.com/81222323/215648194-ad9a4b8d-1a66-48e4-876c-24c2e210e198.png">
실제 이미지 : 검정 분포
생성 이미지 : 초록 분포
판별자 : 파란 분포


## **1-2. Discriminator(판별자)**
* Discriminator(판별자)의 목표는 입력된 이미지가 진짜인지, 가짜인지 잘 분류하는 것이다. **이진분류**



## **1-3. DCGAN**
* 비지도 학습인가? Test data를 사용하지 않는다고 한다.
* 정규화를 [-1, 1]로 하는 이유?



<br/><br/>

# **Q&A**
1. 그럼 `tanh`를 안쓰고 `sigmoid`를 쓸때는 0, 1로 정규화하면 되나요? -> 네 그렇습니다.
2. 두개의 차이는 성능 때문인가요? -> 네 맞습니다. 논문에서 가장 좋았다고 언급했기 때문에, 경험적인 차이가 있기에 `tanh`을 사용합니다.
3. model.add(Conv2D(32, (3,3), padding='same', activation='relu'))레이어의 의미가, (3, 3) 커널을 갖고 32차원으로 출력하는 의미가 맞지요? -> 채널이 32가 되는 것입니다.

<br/><br/>

# **Code**

<br/><br/>

# **기타**
## **1. Normalization**
`Normalization`은 데이터의 범위를 사용자가 원하는 범위로 제한하는 것이다. 예를 들어 이미지 데이터의 경우 픽셀 정보를 0~255 사이의 값으로 가지는데, 
이를 255로 나누어주면 0.0~1.0 사이의 값을 가지게 될 것이다. 이런 행위를 feature들의 scale을 정규화(Normalization)한다고 한다.

## **2. LeakyReLU**
relu함수는 임계값보다 작으면 0을, 크면 입력값을 출력한다. 미분하면 작으면 0, 크면 1을 출력한다. 때문에 `exploding`, `vanishing`문제가 발생할 확률이 작아진다. 그렇다고 `relu`가 완벽하게 문제를 해결하는 것은 아니다. `knockout`문제가 발생할 수 있기 때문이다. 미분했을 때 0 or 1의 값을 가진다는 것은 어떤 레이어 하나에서 모든 노드의 미분 값이 0이 나온 다면 이후의 레이어에서 어떤 값이 나오건 학습이 이루어지지 않는 것을 의미한다. 이런 문제를 해결하는 활성화 함수는 `Leaky relu`이다. `leaky relu`는 임계치보다 작을 때 0을 출력하는 `relu`와는 달리 **0.01을 곱**한다.

하지만 실제 사용에서는 relu를 많이 사용한다. 그 이유는 무엇일까? `relu`은 연산 비용이 크지 않다.  임계치보다 작으면 0을 크면 그 수를 그대로 반환하기 때문이다. 반면에 `leaky relu는` 임계치보다 작으면 0.01을 곱해야 하기 때문에 연산 비용이 상대적으로 크다. 연산 비용이 크다는 것은 속도가 그만큼 느리다는 것을 의미한다. `relu`가 속도면에서 `leaky relu`보다 좋습니다. 따라서 `relu`를 많이 사용한다.

## **BatchNormalization()**
<img width="672" alt="BatchNormalization" src="https://user-images.githubusercontent.com/81222323/215649824-487ed810-23ca-43ca-aab5-0b1eda6e255d.png">
`BatchNormalization()` 에서는 편향과 비슷한 역할을 하느 파라미터가 있기에 편향을 사용하지 않는다. `use_bias=False`


min maxV(D, G)

G(z) = 생성 이미지.
D(G(z)) = 생성자에 노이즈를 넣는다. 1(Real)값이 나와야 생성자 입장에서 좋다. (판별자 입장에서는 0)
log(1-1) = - 무한대
log(1) + log(1) = 0+0 = 0 (-무한대 보다 큰 값.)

logD($x$) = 1

## **지식**
Upsampling + Conv2D가 Conv2DTranspose를 대체할 수도 있다는 점도 기억해두자.


