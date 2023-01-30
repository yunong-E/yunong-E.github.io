---
title: 오토인코더(AE, AutoEncoder)
author: yun
date: 2023-01-30
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, AE, AutoEncoder]
---


# **키워드**
- [ ] 오토인코더(AE, AutoEncoder)
- [ ] 매니폴드 학습(Manifold Learning)
- [ ] Convolutional AutoEncoder
- [ ] 

<br/>

# **개념**
## **1. 오토인코더(AE, AutoEncoder)**
![ae](https://user-images.githubusercontent.com/81222323/215396334-542fb7bc-eb22-456d-baea-41f618924b3b.png)
* **`오토인코더`**는 입력 데이터를 저차원의 벡터로 압축한 뒤 원래 크기의 데이터로 복원하는 신경망이다. 궁극적으로 `Latent 벡터`를 잘 얻기위한 방법이라고 할 수 있다.
* 위의 *Code*라고 표시된 가장 저차원의 벡터는 **`Latent(잠재) 벡터`** 라고 한다. **`Latent 벡터`**란, 원본 데이터보다 차원이 작으면서도, 원본 데이터의 특징을 잘 보전하고 있는 벡터를 말한다.
* `비지도학습`이며 X값만 사용한다. (y값은 사용하지 않는다.)
* `Encoder`와 `Decoder`구조로 이루어져 있으며(**대칭**), `Encoder`에서는 입력 데이터를 저차원의 벡터로 **압축**하고 `Decoder`에서는 원래 크기의 데이터로 **복원**한다.
* 입력층의 노드 개수는 출력층의 개수와 **동일**해야 한다.
* `Latent 벡터` 레이어의 수를 가장 먼저 정해주며, `Latent 벡터`의 차원 수는 데이터를 충분히 표현한다면 적을수록 좋다. `입력층 차원: 100, 잠재층 차원: 100` 의 경우 가능은 하지만 의미가 없다.
* 비선형으로 진행이 되며 딥러닝이기에 과적합 오류가 있다.

* 노이즈의 정도는 20%~25%가 적당하다.
* 흑백이미지를 컬러이미지로 바꾸는 Task, 이미지(손상) 복원 Task, 워터마크를 지우는 Task 등에 활용할 수 있다

<br/>

## **1-2. 오토인코더의 쓰임새**
1. **차원 축소(Dimensionality Reduction)와 데이터 압축**
2. **DAE(데이터 노이즈 제거, Denoising AutoEncoder)** 
  * 원본데이터가 100차원이라면 중요한 신호 + 잡음(= 중요하지 않은 부분)도 있을 것.
  * `Encoder/Decoder` 에서 완전 연결 신경망(`Dense`)이 아닌 Convolution 층(`Conv2D`)을 사용하는 AutoEncoder를 **`Convolutional AutoEncoder`**라고 한다.
4. **이상치 탐지(Anomaly Detection)**
  * 정상적인 데이터의 특징을 뽑아내는 것이기 때문에 비 정상적인 인코더를 AE에 통과시킨다면..
  * `Latent 벡터`를 바탕으로 **<u>원본 데이터로 복원할 때 발생하는 오류</u>**, 즉 **`복원오류(Reconstruction Error)`**를 최소화 하도록 훈련된다.
  * 화재탐지도 가능하다. 시간별로 산의 모습을 촬영. `산불이 난 이미지를 데이터가 학습 -> 이상치로 학습. == 산불을 감지`



## **1-2. `MNIST` 이미지를 3차원 벡터로 축소하는 오토인코더(AutoEncoder)를 설계하고자 한다. (A), (B), (C) 에 들어갈 숫자와 코드는?**
```python
latent_dim = ###(A)###

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.###(B)###(),
            layers.Dense(###(C)### , activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)
```


> 3, Flatten, latent_dim


## **1-3. 이상 현상 발견을 위한 오토인코더(AutoEncoder)**
* 정상 데이터만 사용해서 모델을 학습한다.
* 복원을 했을 때의 오차가 임계값을 초과하는 경우, 해당 데이터를 비정상으로 분류한다.




## **2. 매니폴드 학습(Manifold Learning**

<br/>

# **Q&A**

<br/>

#  **Code**

<br/>
