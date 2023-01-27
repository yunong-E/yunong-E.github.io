---
title: Beyond Classification
author: yun
date: 2023-01-27
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, Segmentation, U-net]
---


# **키워드**
- [ ]
- [ ]
- [ ]
- [ ]
- [ ] MobileNetV2
- [ ] Pix2Pix

<br/><br/>

# **개념**
## **분할(Segmentation)**
![분할](https://user-images.githubusercontent.com/81222323/214996008-cb4afe67-5284-4b64-9151-a93ba5129729.png){: width="500" height="400"}
* `분할(Segmentation)`은 위의 이미지와 같이 하나의 이미지에서 같은 의미를 가지고 있는 부분을 구분해내는 `Task` 이다.
* 이미지 분류에서는 *이미지를 하나의 단위로 레이블을 예측* 했지만, `Segmentation`은 더 *낮은 단위* 로 분류한다. 
* 동일한 의미(사람, 자동차, 도로, 인도, 자연물 등)마다 해당되는 픽셀이 모두 레이블링 되어있는 데이터셋을 **픽셀 단위**에서 레이블을 예측한다.
* 위 그림에서는 같은 의미를 가진 요소를 모두 같은 영역으로 분할했으나, 같은 의미를 가졌더라도 요소가 다르다면 각각의 요소를 구분하여 분할하는 방식도 있다.
* 의료 이미지, 자율 주행, 위성 및 항공 사진 등의 분야에서 많이 사용된다.

<br/>

### **Semantic Segmentation vs (Semantic) Instance Segmentation**
![Instance Segmentation2](https://user-images.githubusercontent.com/81222323/214996479-59b7f921-a5e5-43d3-8463-648a0082fa4b.png)
* 동일한 종류의 객체를 하나로 분류하느냐 vs 요소별로 분류하느냐의 차이
* (Semantic) Instance Segmentation 쪽이 Semantic Segmentation 보다 어렵다.

<br/>

### **이미지 분할(Segmentation)을 위한 대표적인 모델**
1. **FCN(Fully Convolutional Networks)**
![FCN](https://user-images.githubusercontent.com/81222323/214997112-6290527d-db1c-493e-b439-01ba65c2d49c.png)
  * 2015년에 등장했으며, 앞 부분이 `vgg`와 유사함을 알 수 있다.
  * `Segmentation`은 **픽셀 단위로 분류**가 이루어지기 때문에 **픽셀의 위치 정보를 끝까지 보존**해야하지만, 기존 `CNN`에서 사용했던 `완전 연결 신경망`은 **위치 정보를 무시** 한다는 단점을 가지고 있다. 이에 이미지 분류를 위한 신경망에 사용되었던 CNN의 분류기 부분 즉, `완전 연결 신경망(Fully Connected Layer)` 부분을 `합성곱 층(Convolutional Layer)`으로 **모두** 대체해 문제를 해결했다.
  * 위의 이미지에서 pixelwise prediction 부분에서 이미지가 커지는 모습을 확인할 수 있다. 이를 `upsampling` 이라고 한다.
  * pixelwise prediction에 21이라고 적혀있는 부분은 *class의 갯수* 이다.
  * ![upsampling](https://user-images.githubusercontent.com/81222323/214999134-e47bedf0-2861-41c2-9c08-8f0d8da1763d.gif)
  * `Transpose Convolution` 위의 이미지에서 셀이 겹치는 부분은 "더해준다" 라고 생각하면 된다.
  * `upsampling`시 한 번에 너무 크게 키워버리면 경계선이 무너지면서 정확도가 낮아진다. 
  * 뒤에 붙인 숫자가 낮아질 수록 정확도가 높아진다. (FCN-32s, FCN-16s ...)

<br/><br/>

  2. **U-net**
  ![u-net](https://user-images.githubusercontent.com/81222323/214999199-13869dc3-c909-41f2-b3c5-7d6236a83134.png)
  * convolution을 할 때마다, 이미지 사이즈가 2씩 감소하고 있다. 왜 그럴까? __따로 Padding처리를 하지 않았기 때문이다.__
  * U-net은 `Mirroring` 이라는 조금 다른 padding 기법을 사용한다.  **대칭 기법**이라고 볼 수 있다. U-net은 바이오 메디컬 분야에서 사용되기 때문에 이와 같은 방법을 사용한다. (특수성)
  * <img width="512" alt="스크린샷 2023-01-27 오전 11 47 56" src="https://user-images.githubusercontent.com/81222323/214999684-404aefc7-356f-4fad-b495-ac4baaf344c8.png">{: width="100" height="100"}
  * `copy and crop`은 정보손실 방지용이다. (`skip connection`과 비슷하다고 볼 수 있으며 `long skip connection`이라고도 한다.)

<br/>

## **객체 탐지/인식(Object Detection/Recognition)**
  ### **IoU(Intersection over Union)**
  ![iou](https://user-images.githubusercontent.com/81222323/215001078-975b76a9-02b3-4e10-8dc3-699b3a027f67.png)
  * 객체 탐지를 평가하는 지표이다. 공식은 위와 같으며 1에 가까울수록 정확도가 높은 것이다.
  * 정답에 해당하는 Bounding Box는 `Ground-truth` 라고 한다.
  * `IoU`를 사용하면 객체를 포함하고 있지만 그 범위를 너무 크게 잡을 때의 문제를 해결할 수 있다.
  * `IoU`가 구해지는 예시는 아래 이미지와 같다.
  ![iou2](https://user-images.githubusercontent.com/81222323/215014938-b3558af9-80ed-43c1-b923-e84c04dac3a5.png)

 <br/> 
  
  ### 대표적인 객체 탐지 Model
  ![fuC2OJA](https://user-images.githubusercontent.com/81222323/215015080-9c4a5dae-342e-4754-9598-f62df0096eff.png)
  위의 그림처럼 발전해 왔다. 어떤 단계를 거쳐 분류가 진행되는지에 따라서 `2-stage`방식과 `1-stage`방식으로 나눌 수 있다.

  
    1. **Two Stage Detector**
      * 물체가 있을만한 영역을 추천을 받는다.
      * 속도적인 측면에서는 One Stage Detector 보다 떨어지나 인식 성능은 One Stage Detector 보다 정확하다. 


    2. **One Stage Detector**
      * 추천을 받지 않고 물체를 Grid로 나눈뒤 분류를 진행한다.
      * 속도적인 측면에서는 Two Stage Detector 보다 빠르나 인식 성능은 Two Stage Detector 보다 떨어진다.
      * 대표적인 모델로 YOLO(You)

 <br/> 

# **Q&A**
1. 이미지 증강을 하는 이유? : 일반화 모델을 만들기 위함. Test set에도 하면 과적합의 위함이 있기에 Train set에만 한다.

<br/><br/>

# **Code**

<br/><br/>

# **눈치게임**
## **왜**
### **깨졌을까**
#### **알려주세요**
##### **오류 장난 아님.**
