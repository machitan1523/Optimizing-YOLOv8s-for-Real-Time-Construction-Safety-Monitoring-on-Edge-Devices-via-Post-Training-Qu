# ⛑️ Implementation and Performance Analysis of Lightweight Object Detection System for Real-time Construction Safety Monitoring on Edge Device

This project uses compatible Pi and Hailo AI accelerators to verify the use of safety equipment in imports.

## 📝 Abstract (소개)
To address the limitations of manual supervision and cloud-based architectures in high-risk environments, this paper proposes a real-time Personal Protective Equipment (PPE) detection system implemented on a low-power edge device using a Raspberry Pi 5 and Hailo-8 NPU. By deploying a YOLOv8s model optimized via Post-Training Quantization (PTQ), we achieved a 54.7% reduction in model size while maintaining a high mAP@0.5 of 0.8817. Experimental results demonstrate that the proposed system reaches an inference speed of 32.99 FPS—an approximate 30-fold increase over CPU-only execution—thereby proving that a decentralized edge solution can effectively ensure bandwidth efficiency and privacy while delivering server-level performance for real-world safety monitoring.

## 📂 Dataset 
The following datasets were used for this project.
* Construction Site Safety Image Dataset Roboflow : https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow - For detecting people, helmets and vests

## 🛠️ Environment (개발 환경)
* **Hardware**: Raspberry Pi 4, Hailo-8L
* **Language**: Python 3.8
* **Libraries**: PyTorch, Hailo RT

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
