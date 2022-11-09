# Image Processing With Pyside6

### 1. 개발 환경 구축

```
git clone https://github.com/patrashu/image_processing_with_Pyside6.git
cd image_processing_with_Pyside6
pip install -r requirements.txt
```

### 2. 객체 탐지용 가중치

- [yolov7.pt]("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
- [traced_model.pt](https://drive.google.com/file/d/1lEgiWru9c53Rtin3NbeBRJ0Nw9o3NY4U/view?usp=share_link)

### 3. 실행

```python
python main.py
```

### 4. 결과창 출력

![show](https://user-images.githubusercontent.com/78347296/200880020-8f2179d6-d572-4573-876a-45193398d9e8.jpg)

<hr>

### 사용법

#### 이미지 처리

- 경계선 출력, 이진화, 이미지 회전 기능은 우측 다이얼을 통해 실시간으로 변화를 확인할 수 있습니다.
- 얼굴 윤곽 확인, 근골격계 탐지 등의 다양한 기능이 구현되어 있습니다.

#### 이미지 분류

- 이미지 선택 후 버튼을 클릭하면, 이미지 분류기의 결과를 확인할 수 있습니다.

#### 이미지 내 객체 탐지

- 이미지 선택 후 버튼을 클릭하면, 객체 탐지기의 결과를 확인할 수 있습니다.

#### 이미지 처리단계에서 주의할 점

- 이미지 확대는 2개의 점을 선택해야합니다. (좌 상단, 우 하단)
- 이미지 워핑은 8개의 점을 선택해야합니다. (워핑에 사용할 값 지정)
- 그 후, 결과 출력 버튼을 클릭하면 결과창이 우측에 출력됩니다.
