# ⛑️ Implementation and Performance Analysis of Lightweight Object Detection System for Real-time Construction Safety Monitoring on Edge Device

This project uses compatible Pi and Hailo AI accelerators to verify the use of safety equipment in imports.

## 📝 Abstract (소개)
To address the limitations of manual supervision and cloud-based architectures in high-risk environments, this paper proposes a real-time Personal Protective Equipment (PPE) detection system implemented on a low-power edge device using a Raspberry Pi 5 and Hailo-8 NPU. By deploying a YOLOv8s model optimized via Post-Training Quantization (PTQ), we achieved a 54.7% reduction in model size while maintaining a high mAP@0.5 of 0.8817. Experimental results demonstrate that the proposed system reaches an inference speed of 32.99 FPS—an approximate 30-fold increase over CPU-only execution—thereby proving that a decentralized edge solution can effectively ensure bandwidth efficiency and privacy while delivering server-level performance for real-world safety monitoring.

## 📂 Dataset 
The following datasets were used for this project.
* Construction Site Safety Image Dataset Roboflow : https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow - For detecting people, helmets and vests

본 프로젝트는 Kaggle의 'Construction Site Safety' 데이터셋을 재가공하여 사용했습니다.

| 클래스 (Class) | 학습 (Train) | 검증 (Validation) | 테스트 (Test) | 합계 (Total Instances) | 비율 (Ratio) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Person (작업자)** | 10,026 | 163 | 181 | 10,370 | 63.0% |
| **Hardhat (안전모)** | 2,889 | 70 | 103 | 3,062 | 18.6% |
| **Safety Vest (조끼)** | 2,933 | 37 | 61 | 3,031 | 18.4% |
| **합계 (Images)** | **2,605장** | **114장** | **82장** | **2,801장** | **100%** |

> **Note:** 클래스 불균형(Imbalance)이 존재하지만, Person, Hardhat, Safety Vest 3가지 핵심 클래스를 중점적으로 학습했습니다.

## 🛠️ Environment (개발 환경)
* **Hardware**: Raspberry Pi 5 (Broadcom BCM2712), Hailo-8 NPU
* **Language**: Python 3.12.12
* **Libraries**: PyTorch 2.9.0+cu126, Ultralytics YOLOv8.3.233, Hailo Dataflow Compiler v3.33.0

## 📊 Performance & Benchmark (실험 결과)
`cpu_benchmark.py`와 `hailo_benchmark.py`를 통해 측정한 결과입니다.

| 디바이스 | 해상도 | 추론 속도 (FPS) | 전력 소모 (W) |
| :---: | :---: | :---: | :---: |
| Raspberry Pi (CPU) | 640x640 | 2.5 | 4.2 |
| **RPi + Hailo** | **640x640** | **30.1** | **5.5** |

> Hailo 가속기를 사용했을 때 CPU 대비 약 12배 빠른 속도를 보였습니다.

## 🚀 How to Run (실행 방법)
1. 의존성 라이브러리 설치
   ```bash
   pip install -r requirements.txt
