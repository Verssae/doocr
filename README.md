# DOOCR: Delivery Order OCR
표 형태의 물류 주문서(Delivery Order) PDF 파일을 인식하여, 주문서의 내용을 추출하는 프로그램입니다. 텍스트 감지 모델과 텍스트 인식 모델으로 구성된 [DocTR](https://github.com/mindee/doctr)을 기반으로 개발되었습니다. 주된 목표는 텍스트 인식 모델로 ViTSTR(Vision Transformer for Scene Text Recognition by [ICDAR 2021](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_21))을 사용하고, Attention Rollout을 통해 모델의 예측 결과에 대한 설명가능성을 제공하는 것입니다.

이를 위해 [vit-explain](https://github.com/jacobgil/vit-explain)의 코드를 일부 수정하여 DocTR의 ViTSTR 및 RecognitionPredictor 관련 부분을 수정하여 사용하였습니다.

GUI 기반의 사용성을 제공하기 위해 [streamlit](https://streamlit.io) 로컬 서버 기반의 웹 애플리케이션으로 개발되었습니다.

## 주요 기능
- 텍스트 검출(Text Detection): DocTR에서 제공하는 db_resnet, mobilenet_v3, linknet 기반의 사전 훈련된 모델을 사용할 수 있습니다. Binarization Threshold를 조정하여 텍스트 검출의 성능을 조절할 수 있습니다.
- 텍스트 인식(Text Recognition): DocTR에서 제공하는 ViTSTR 모델인 사전 훈련된 vitstr_small 및 vitstr_base를 사용할 수 있습니다. Attention Rollout을 통해 모델의 예측 결과에 대한 설명가능성을 제공합니다.
- 텍스트 인식 결과 시각화: 텍스트 인식 결과를 시각화하여 제공합니다. 입력 PDF 페이지, Segmentation heatmap, OCR 결과 텍스트 박스 시각화, Attention Rollout 결과를 시각화하여 제공합니다.
- 텍스트 인식 결과 추출: 텍스트 인식 결과를 텍스트 파일로 추출하여 제공합니다. 이 때, 단어 별로 인식되어 라인 별로 정렬하기 위한 알고리즘을 적용하여 제공합니다. 추가적으로 인접한 단어들을 문장으로 간주하여 문장 단위로 추출할 수 있습니다.
- 설명가능성 결과 추출: Attention Rollout 결과를 각 텍스트 박스 별로 추출하여 제공합니다. 이 때, 어텐션 헤드를 통합하는 방식(head_fusion: min, max, mean)과 제거할 낮은 어텐션 가중치 비율(discard_ratio)을 조절할 수 있습니다.
- Apple Silicon (MPS) 지원: Apple Silicon에서 실행 가능하도록 pytorch 모델 실행 시 MPS(Metal Performance Shaders)를 사용하여 최적화하였습니다. 

## 설치 

```bash
git clone https://github.com/mindee/doctr.git doctr_github
pip install -e doctr_github/.
pip install -e doctr_github/.[torch]
```
DocTR을 위와 같이 develop mode로 설치합니다. 

## 실행
```bash
streamlit run app.py
```

## 환경
- Python 3.11.6
- MacOS 14.1.2 (Apple M2)