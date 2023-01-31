
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

<br/><br/>

# **개념**
## **1. GAN(Generative Adversarial Networks, 생성적 적대 신경망)**
* GAN의 목표는 실제와 유사한 이미지 등을 만들어 내는 것이다. (e.g. 딥페이크)
* Generative라는 말은 '생성적인, 생산하는' 이라는 뜻으로, 이 모델을 이용해서 이미지 등을 생성할 수 있기 때문에 해당 표현이 사용되었다.
* Adversarial은 '적대적인'이라는 뜻으로, 이 모델에서 길항 작용을 하는 두 네트워크가 서로 경쟁하면서 발전하기 때문에 해당 표현을 사용한다.
* Generator(생성자)의 목표는 Random noise를 사용해서 Discriminator(판별자)를 속일 수 있는 가짜 이미지를 만들어 내는 것이다.
* GAN의 손실 함수는 Generator(생성자)의 손실과 Discriminator(판별자)의 손실을 모두 고려한다.
* Discriminator(판별자)의 목표는 입력된 이미지가 진짜인지, 가짜인지 잘 분류하는 것이다. **이진분류**


<br/><br/>


# **Q&A**

<br/><br/>

# **Code**


# 기타
## **Normalization**
`Normalization`은 데이터의 범위를 사용자가 원하는 범위로 제한하는 것이다. 예를 들어 이미지 데이터의 경우 픽셀 정보를 0~255 사이의 값으로 가지는데, 
이를 255로 나누어주면 0.0~1.0 사이의 값을 가지게 될 것이다. 이런 행위를 feature들의 scale을 정규화(Normalization)한다고 한다.

## LeakyReLU
relu함수는 임계값보다 작으면 0을, 크면 입력값을 출력한다. 미분하면 작으면 0, 크면 1을 출력한다. 때문에 `exploding`, `vanishing`문제가 발생할 확률이 작아진다. 그렇다고 `relu`가 완벽하게 문제를 해결하는 것은 아니다. `knockout`문제가 발생할 수 있기 때문이다. 미분했을 때 0 or 1의 값을 가진다는 것은 어떤 레이어 하나에서 모든 노드의 미분 값이 0이 나온 다면 이후의 레이어에서 어떤 값이 나오건 학습이 이루어지지 않는 것을 의미한다. 이런 문제를 해결하는 활성화 함수는 `Leaky relu`이다. `leaky relu`는 임계치보다 작을 때 0을 출력하는 `relu`와는 달리 **0.01을 곱**한다.
