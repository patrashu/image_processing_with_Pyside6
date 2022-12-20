# Pyside6를 활용한 이미지 처리기

<div align="center">
    <h3>Stack</h3>
    <img src="https://img.shields.io/badge/PyQt-41CD52?style=flat-square&logo=qt&logoColor=white"/>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/><br>
    <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/>
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/>
    <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/>
</div>

<hr>

## 구현된 기능

<div align="center">

|**<h3>기능</h3>**  |**<h3>설명</h3>**             |
|:-----|:---------------|
|**![home](https://user-images.githubusercontent.com/78347296/208606709-d41b34ca-1317-4b2e-97e0-c87dfc68e31d.png) Home Button**|
|**파일열기**|경로 내 이미지 불러오기|
|**새로 고침**|불러온 이미지 초기화|
|**좌우반전**|이미지 좌우 반전|
|**이미지 정규화**|정규 분포를 따를 수 있도록 조정된 이미지 |
|**이미지 선명화**||
|**경계선 검출**|threshold 변화에 따른 경계선 검출 결과 출력|
|**이미지 회전**|radian 변화에 따른 이미지 회전 결과 출력|
|**이진화**|threshold 변화에 따른 이진화 결과 출력|
|**모자이크**|이미지 내 존재하는 사람 얼굴을 모자이크한 결과 출력|
|**얼굴 윤곽 검출**|이미지 내 존재하는 사람 얼굴 윤곽 검출 결과 출력|
|**신체 관절 검출**|이미지 내 존재하는 사람의 관절 어노테이션 및 신체 키포인트 검출 결과 출력|
|**이미지 확대**|마우스 클릭으로 지정된 두점 사이의 픽셀 값을 확대한 결과 출력|
|**이미지 워핑**|마우스 클릭 이벤트로 지정된 관심 영역 사이 이미지 워핑 결과 출력|
|**![plug](https://user-images.githubusercontent.com/78347296/208606706-962ad825-b200-472b-8f76-a9c484e58443.png) Plug Button**|
|**이미지 분류기**|ImageNet으로 사전 학습된 가중치를 기반으로 이미지 분류 결과 출력|
|**![batch](https://user-images.githubusercontent.com/78347296/208606711-af4d7cee-8211-4a75-9ee7-09b3f3c3ca0b.png) Batch Button**|
|**객체 탐지**|COCO Dataset으로 학습된 가중치를 기반으로 이미지 내 존재하는 객체 탐지 결과 출력|

</div>
<hr>

## 사용 방법

### 1. 개발 환경 구축
- Miniconda 개발 환경 사용

```
git clone https://github.com/patrashu/image_processing_with_Pyside6.git
cd image_processing_with_Pyside6
conda create -n qt python=3.8
conda activate qt
pip install -r requirements.txt
```

### 2. 객체 탐지용 가중치
- [yolov7.pt](https://drive.google.com/file/d/1GeUhf_MHBBcEPyZf9Cg95kR1rMpcqkIA/view?usp=share_link)
- [traced_model.pt](https://drive.google.com/file/d/1lEgiWru9c53Rtin3NbeBRJ0Nw9o3NY4U/view?usp=share_link)

### 3. 실행 및 결과 확인

```python
python main.py
```

### 추가 주의사항

- 이미지 확대<br>
    | 마우스 클릭 이벤트로 확대할 영역의 좌상단, 우하단 총 2개의 점 선택.<br>
    | 그 후, 이미지 확대 버튼을 누르면 오른쪽에 관심 영역으로 지정한 부분이 확대된 이미지 결과 확인
- 이미지 워핑<br>
    | 마우스 클릭 이벤트로 총 8개의 점 선택<br>
    | 그 후, 이미지 워핑 버튼을 누르면 오른쪽에 관심 영역으로 지정한 부분이 확대된 이미지 결과 확인
