---
title: Beyond Classification
author: yun
date: 2023-01-27
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, Segmentation, U-net]
---


# **키워드**
- [ ] Semantic Segmentation
- [ ] Instance Segmentation
- [ ] FCN(Fully Convolutional Networks)
- [ ] Transpose Convolution
- [ ] 객체 탐지 / 객체 인식 / Object Detection / Object Recognition
- [ ] mAP
- [x] IoU
- [ ] MobileNetV2
- [ ] Pix2Pix
- [ ] One Stage Detector
- [ ] Two Stage Detector

<br/><br/>

# **개념**
## **1. 분할(Segmentation)**
![분할](https://user-images.githubusercontent.com/81222323/214996008-cb4afe67-5284-4b64-9151-a93ba5129729.png){: width="500" height="400"}
* `분할(Segmentation)`은 위의 이미지와 같이 하나의 이미지에서 같은 의미를 가지고 있는 부분을 구분해내는 `Task` 이다.
* 이미지 분류에서는 *이미지를 하나의 단위로 레이블을 예측* 했지만, `Segmentation`은 더 *낮은 단위* 로 분류한다. 
* 동일한 의미(사람, 자동차, 도로, 인도, 자연물 등)마다 해당되는 픽셀이 모두 레이블링 되어있는 데이터셋을 **픽셀 단위**에서 레이블을 예측한다.
* 위 그림에서는 같은 의미를 가진 요소를 모두 같은 영역으로 분할했으나, 같은 의미를 가졌더라도 요소가 다르다면 각각의 요소를 구분하여 분할하는 방식도 있다.
* 의료 이미지, 자율 주행, 위성 및 항공 사진 등의 분야에서 많이 사용된다.

<br/>

### **1-1. Semantic Segmentation vs (Semantic) Instance Segmentation**
![Instance Segmentation2](https://user-images.githubusercontent.com/81222323/214996479-59b7f921-a5e5-43d3-8463-648a0082fa4b.png)
* 동일한 종류의 객체를 하나로 분류하느냐 vs 요소별로 분류하느냐의 차이
* (Semantic) Instance Segmentation 쪽이 Semantic Segmentation 보다 어렵다.
### **2-1. 이미지 분할(Segmentation)을 위한 대표적인 모델**
#### **a. FCN(Fully Convolutional Networks)**
  ![FCN](https://user-images.githubusercontent.com/81222323/214997112-6290527d-db1c-493e-b439-01ba65c2d49c.png)
  * 2015년에 등장했으며, 앞 부분이 `vgg`와 유사함을 알 수 있다.
  * `Segmentation`은 **픽셀 단위로 분류**가 이루어지기 때문에 **픽셀의 위치 정보를 끝까지 보존**해야하지만, 기존 `CNN`에서 사용했던 `완전 연결 신경망`은 **위치 정보를 무시** 한다는 단점을 가지고 있다. 이에 이미지 분류를 위한 신경망에 사용되었던 CNN의 분류기 부분 즉, `완전 연결 신경망(Fully Connected Layer)` 부분을 `합성곱 층(Convolutional Layer)`으로 **모두** 대체해 문제를 해결했다.
  * 위의 이미지에서 pixelwise prediction에 21이라고 적혀있는 부분은 *class의 갯수* 이다.
  * 위의 이미지에서 pixelwise prediction 부분에서 이미지가 커지는 모습을 확인할 수 있다. 이를 `Upsampling` 이라고 한다.
  * 반대로 `CNN`에서 사용되는 것처럼 Convolution과 Pooling을 사용해 이미지의 특징을 추출하는 과정을 `Downsampling` 이라고 한다.
  * `DownSampling`을 수행하는 부분을 **인코더**, `Upsampling`를 수행하는 부분을 **디코더**라고도 부른다.
  * `U-net`에서는 `DownSampling`을 수행할 때 `Max Pooling`을 사용한다.
  * `Upsampling`에는 기존 `Convolution`과 다른 `Transpose Convolution`이 적용되며, Transpose Convolution에서는 각 픽셀에 커널을 곱한 값에 Stride를 주고 나타내면서 이미지 크기를 키워나간다. 아래는 2x2 이미지가 입력됐을때, 3x3 필터에 의해 `Transpose Convolution` 되는 과정이 담긴 이미지이다.
  * ![upsampling](https://user-images.githubusercontent.com/81222323/214999134-e47bedf0-2861-41c2-9c08-8f0d8da1763d.gif)
  * 위의 이미지에서 셀이 겹치는 부분은 "더해준다" 라고 생각하면 된다.
  * `Upsampling`시 한 번에 너무 크게 키워버리면 경계선이 무너지면서 정확도가 낮아진다. 
  * 뒤에 붙인 숫자가 낮아질 수록 정확도가 높아진다. (FCN-32s, FCN-16s ...)
#### **b. U-net**
  ![u-net](https://user-images.githubusercontent.com/81222323/214999199-13869dc3-c909-41f2-b3c5-7d6236a83134.png){: width="500" height="500"}
  * convolution을 할 때마다, 이미지 사이즈가 2씩 감소하고 있다. 왜 그럴까? *따로 Padding처리를 하지 않았기 때문*이다.
  * U-net은 `Mirroring` 이라는 조금 다른 padding 기법을 사용한다.  **대칭 기법**이라고 볼 수 있다. U-net은 바이오 메디컬 분야에서 사용되기 때문에 이와 같은 방법을 사용한다. (특수성) 아래의 이미지에서 확인할 수 있다.
  * <img width="512" alt="스크린샷 2023-01-27 오전 11 47 56" src="https://user-images.githubusercontent.com/81222323/214999684-404aefc7-356f-4fad-b495-ac4baaf344c8.png">{: width="300" height="300"}
  * `copy and crop`은 정보손실 방지용이다. (`skip connection`과 비슷하다고 볼 수 있으며 `long skip connection`이라고도 한다.)

<br/>

## **2. 객체 탐지/인식(Object Detection/Recognition)**
* 객체의 경계에 `Bounding Box` 라고 하는 사각형 박스를 만들고 박스 내의 객체가 속하는 클래스가 무엇인지를 분류한다.
* 객체 탐지 결과를 평가하기 위해 IoU(Intersection over Union)라는 지표를 사용한다.
### **2-1. IoU(Intersection over Union)**
  ![iou](https://user-images.githubusercontent.com/81222323/215001078-975b76a9-02b3-4e10-8dc3-699b3a027f67.png){: width="300" height="300"}
  * 객체 탐지를 평가하는 지표이다. 공식은 위와 같으며 1에 가까울수록 정확도가 높은 것이다.
  * 정답에 해당하는 Bounding Box는 `Ground-truth` 라고 한다.
  * `Ground-truth`와 예측 영역의 교집합을 `Ground-truth`와 예측 영역의 합집합으로 나눠서 구한다.
  * `IoU`를 사용하면 객체를 포함하고 있지만 그 범위를 너무 크게 잡을 때의 문제를 해결할 수 있다.
  * `IoU`가 구해지는 예시는 아래 이미지와 같다. 
  ![iou2](https://user-images.githubusercontent.com/81222323/215014938-b3558af9-80ed-43c1-b923-e84c04dac3a5.png)
### **2-2. 대표적인 객체 탐지 Model**
![fuC2OJA](https://user-images.githubusercontent.com/81222323/215015080-9c4a5dae-342e-4754-9598-f62df0096eff.png)
위의 그림처럼 발전해 왔다. 어떤 단계를 거쳐 분류가 진행되는지에 따라서 `2-stage`방식과 `1-stage`방식으로 나눌 수 있다.

  1) **Two Stage Detector** <br/>
    * 물체가 있을만한 영역을 추천을 받는다. <br/>
    * 속도적인 측면에서는 One Stage Detector 보다 떨어지나 인식 성능은 One Stage Detector 보다 정확하다. <br/>
    * 대표적인 모델로 R-CNN, Fast R-CNN 이 있다. <br/>

  2) **One Stage Detector** <br/>
    * 추천을 받지 않고 물체를 Grid로 나눈뒤 분류를 진행한다. <br/>
    * 속도적인 측면에서는 Two Stage Detector 보다 빠르나 인식 성능은 Two Stage Detector 보다 떨어진다. <br/>
    * 대표적인 모델로 YOLO(You) <br/>

 <br/> 

## **Fast R-CNN 설명 및 정리**
`Fast R-CNN`은 이전 `R-CNN`의 한계점을 극복하고자 나왔다. 참고로 R-CNN의 한계점은 다음과 같다. <br/>
1) RoI (Region of Interest) 마다 CNN연산을 함으로써 속도저하 <br/>
2) multi-stage pipelines으로써 모델을 한번에 학습시키지 못함 <br/><br/>

`Fast R-CNN`에서는 다음 두 가지를 통해 위 한계점들을 극복했다. <br/>
1) RoI pooling <br/>
2) CNN 특징 추출부터 classification, bounding box regression까지 하나의 모델에서 학습 <br/>

[참조](https://ganghee-lee.tistory.com/36)

<br/><br/>

# **Q&A**
1. 이미지 증강을 하는 이유? : 일반화 모델을 만들기 위함. Test set에도 하면 과적합의 위함이 있기에 Train set에만 한다.
2. 파이썬은 대소문자를 구분하나요? : 하찮은 질문이 아니에용. 대소문자를 구분합니다. 이로 인해 자주 오류를 범하게 되니 항상 신경써주시면 좋을 것 같아용!

<br/><br/>

# **Code**
> "추가적으로 Upsampling 에서 Downsampling 출력으로 나왔던 Feature map 을 적당한 크기로 잘라서 붙여준 뒤 추가 데이터로 사용합니다." <br/>
> 위의 순서는 Part(B)에 수행된다.


```python
def unet_model(output_channels):

  # Part (A) --------------------------
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Part (B) --------------------------
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # Part (C) --------------------------
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')

  # Part (D) --------------------------
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
```

<br/><br/>
