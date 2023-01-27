---
title: Beyond Classification
author: yun
date: 2023-01-27
categories: [Blogging, Study, Ai, Summary]
tags: [study, python, deep learning, Segmentation, U-net]
---

니나노 

# 키워드
# 개념
## 분할(Segmentation)
![분할](https://user-images.githubusercontent.com/81222323/214996008-cb4afe67-5284-4b64-9151-a93ba5129729.png)
* `분할(Segmentation)`은 위의 이미지와 같이 하나의 이미지에서 같은 의미를 가지고 있는 부분을 구분해내는 Task 이다.
* 이미지 분류에서는 *이미지를 하나의 단위로 레이블을 예측*했지만, Segmentation 은 더 낮은 단위로 분류한다. 동일한 의미(사람, 자동차, 도로, 인도, 자연물 등)마다 해당되는 픽셀이 모두 레이블링 되어있는 데이터셋을 **픽셀 단위**에서 레이블을 예측한다.
* 위 그림에서는 같은 의미를 가진 객체를 모두 같은 영역으로 분할했으나, 같은 의미를 가졌더라도 개체가 다르다면 각각의 개체를 구분하여 분할하는 방식도 있다.
* 의료 이미지, 자율 주행, 위성 및 항공 사진 등의 분야에서 많이 사용된다.

  ### Semantic Segmentation vs (Semantic) Instance Segmentation
  ![Instance Segmentation2](https://user-images.githubusercontent.com/81222323/214996479-59b7f921-a5e5-43d3-8463-648a0082fa4b.png)
  * 동일한 종류의 객체를 하나로 분류하느냐 vs 객체를 개체별로 분류하느냐의 차이
  

  ### 이미지 분할(Segmentation)을 위한 대표적인 모델 
  1. **FCN(Fully Convolutional Networks)**
  ![FCN](https://user-images.githubusercontent.com/81222323/214997112-6290527d-db1c-493e-b439-01ba65c2d49c.png)
    * 2015년에 등장.
    * Segmentation 은 픽셀 단위로 분류가 이루어지기 때문에 픽셀의 위치 정보를 끝까지 보존해야하지만, 기존 `CNN`에서 사용했던 `완전 연결 신경망`은 *위치 정보를 무시*한다는 단점을 가지고 있다. 이에 이미지 분류를 위한 신경망에 사용되었던 CNN의 분류기 부분 즉, `완전 연결 신경망(Fully Connected Layer)` 부분을 `합성곱 층(Convolutional Layer)`으로 **모두** 대체해 문제를 해결했다.
    * 위의 이미지에서 pixelwise prediction 부분에서 이미지가 커지는 모습을 확인할 수 있다. 이를 `upsampling` 이라고 한다.



  2. **U-net**
