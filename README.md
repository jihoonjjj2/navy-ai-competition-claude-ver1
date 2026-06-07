# Navy AI Competition — 자동 멀티모달 파이프라인

**데이터만 넣으면 종류(오디오/이미지/표/시계열)를 자동 판별해 알맞게 처리하고 제출까지** 만듭니다.

## 한 번에 실행
```
python run_all.py
```
끝입니다. 종류 판별 → EDA/시각화 → 학습 → (이진이면 임계값 튜닝) → 제출.

## 지원하는 데이터 종류 (자동 판별)
| 종류 | 형태 | 처리 |
|------|------|------|
| 오디오 분류 | 클래스별 wav 폴더 | 스펙트로그램(LOFAR/Mel/DEMON) + CNN |
| 이미지 분류 | 클래스별 이미지 폴더 | CNN(timm) |
| 표 데이터 | CSV(라벨 컬럼) | LightGBM + 특성공학 |
| 시계열/항적 | id+시계열 CSV(AIS 등) | id별 특성집계 + LightGBM |

## 모델/기법 바꾸기 (쉬움)
- 딥: `model.deep_name` = resnet34, efficientnet_b0~b3, convnext_tiny 등 (timm 전부)
- 표: `model.tabular_name` = lightgbm/xgboost/catboost/histgb/randomforest
- 오디오 특징: `audio.feature` = lofar/mel/demon
- 명령줄: `python run_all.py --deep_model resnet34 --feature mel`
- 새 표 모델 추가: `src/models/registry.py`의 `TABULAR_REGISTRY`에 한 줄 등록
- 시퀀스 특성 추가: `src/data/tabular.py`의 `engineer_sequence` 수정

## 자동으로 처리되는 것 (완전 자동화)
- 데이터 경로 자동 탐색 / 종류 자동 판별
- 라벨·id 컬럼 자동 추정, 비대상 폴더(happy 등) 자동 제외
- 이진/다중 자동 판별 → 이진이면 **임계값 튜닝**(F1 최대화)
- 클래스 불균형 → class_weight 자동
- 누수 방지 CV(그룹/계층) + fold 수 자동 조정
- 제출 전 assert 검증 + 클래스 분포 점검
- 결과 시각화 자동 저장(분포/스펙트로그램/혼동행렬/중요도/학습곡선)

## 시각화 결과
`outputs/figures/`에 EDA·혼동행렬·feature importance·학습곡선·스펙트로그램 샘플이 저장됩니다.

## 폴더 구조
```
src/
├── detect.py            데이터 종류 자동 판별
├── pipeline.py          라우팅(완전 자동)
├── data/{paths,audio,image,tabular,folds}.py
├── models/registry.py   모델 레지스트리(딥+표)
├── train/{deep,tabular}.py
├── infer/{threshold,submission,ensemble}.py
└── utils/{log,seed,metrics,viz}.py
```
