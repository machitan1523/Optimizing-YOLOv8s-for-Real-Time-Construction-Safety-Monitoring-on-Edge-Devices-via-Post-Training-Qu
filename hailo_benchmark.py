import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import cv2
import time
import sys
import psutil
import csv
import threading
import matplotlib.pyplot as plt
import os
import signal
from threading import Thread, Lock

# ==========================================
# 1. 설정 (파일명 확인!)
# ==========================================
HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
SAVE_VIDEO_PATH = "hailo_visual_350_result.mp4"
CSV_FILENAME = "hailo_visual_350_log.csv"
GRAPH_FILENAME = "hailo_visual_350_graph.png"

# 클래스 정의
CLASSES = { 0: 'Person', 1: 'Hardhat', 2: 'Safety Vest' }
CONF_THRESHOLD = 0.45  # 45% 이상일 때만 박스 그리기

# 공유 변수
shared_status = {
    'fps': 0.0,
    'running': True
}

# ==========================================
# [시스템 모니터링 클래스] (그래프 저장용)
# ==========================================
class SystemMonitor:
    def __init__(self, filename=CSV_FILENAME, interval=1.0):
        self.filename = filename
        self.interval = interval
        self.thread = None
        self.records = []
        plt.switch_backend('Agg')

        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "FPS", "CPU", "Memory", "Temperature"])

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except: return 0.0

    def _monitor_loop(self):
        start_time = time.time()
        while shared_status['running']:
            try:
                curr_t = time.time() - start_time
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().used / (1024 * 1024)
                temp = self.get_cpu_temp()
                fps = shared_status['fps']

                if curr_t > 2.0: # 초기 2초 제외
                    self.records.append((curr_t, fps, cpu, mem, temp))

                with open(self.filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([round(curr_t, 2), round(fps, 1), cpu, round(mem, 1), round(temp, 1)])
            except: pass
            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        shared_status['running'] = False
        if self.thread: self.thread.join()
        self.generate_report()

    def generate_report(self):
        if not self.records: return
        times = [r[0] for r in self.records]
        fpss = [r[1] for r in self.records]
        cpus = [r[2] for r in self.records]
        mems = [r[3] for r in self.records]
        temps = [r[4] for r in self.records]

        # 통계 출력
        print("\n" + "="*40)
        print("   📊 [최종 측정 결과 (시각화 포함)]")
        print("="*40)
        print(f" ✅ 평균 FPS      : {sum(fpss)/len(fpss):.2f} FPS")
        print(f" ✅ 평균 CPU 사용 : {sum(cpus)/len(cpus):.2f} %")
        print(f" ✅ 최고 온도     : {max(temps):.1f} ℃")
        print("="*40)

        # 그래프 그리기
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
       
        ax1.plot(times, fpss, color='purple'); ax1.set_ylabel('FPS'); ax1.grid(True)
        ax1.set_title('Real-time Inference with Visualization')
       
        ax2.plot(times, cpus, color='blue'); ax2.set_ylabel('CPU (%)'); ax2.set_ylim(0, 100); ax2.grid(True)
       
        ax3.plot(times, mems, color='green'); ax3.set_ylabel('Mem (MB)'); ax3.grid(True)
       
        ax4.plot(times, temps, color='red'); ax4.set_ylabel('Temp (C)'); ax4.grid(True)
        ax4.axhline(y=80, color='gray', linestyle='--')
        ax4.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(GRAPH_FILENAME)
        plt.close()
        print(f">>> 그래프 저장 완료: {GRAPH_FILENAME}")

# ==========================================
# [카메라 클래스]
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.status, self.frame = self.capture.read()
        self.lock = Lock()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            status, frame = self.capture.read()
            if status:
                with self.lock:
                    self.status, self.frame = status, frame
            else: self.stopped = True

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.capture.release()

# ==========================================
# [메인 실행 함수]
# ==========================================
def run_visual_benchmark():
    # Ctrl+C 처리
    def signal_handler(sig, frame): shared_status['running'] = False
    signal.signal(signal.SIGINT, signal_handler)

    monitor = SystemMonitor()
    monitor.start()

    webcam = ThreadedCamera(0)
    if not webcam.status: return
    webcam.start()
    time.sleep(1.0) # 카메라 안정화

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, 30.0, (640, 480))

    print(f">>> [Hailo] 모델 로딩 중: {HEF_FILE}")

    try:
        hef = HEF(HEF_FILE)
        params = VDevice.create_params()

        with VDevice(params) as target:
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()
           
            input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            input_info = hef.get_input_vstream_infos()[0]
            mw, mh = input_info.shape[1], input_info.shape[0]

            with network_group.activate(network_group_params):
                with InferVStreams(network_group, input_params, output_params) as pipeline:
                   
                    pipeline.infer(np.zeros((1, mh, mw, 3), dtype=np.float32)) # 워밍업
                    print(">>> [시작] 실시간 화면에 박스가 그려집니다. (종료: 'q')")
                   
                    prev_time = time.time()

                    while shared_status['running']:
                        frame = webcam.read()
                        if frame is None: continue
                        h, w, _ = frame.shape

                        # 1. 전처리
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inp = cv2.resize(img_rgb, (mw, mh)).astype(np.float32) / 255.0
                        inp = np.expand_dims(inp, axis=0)

                        # 2. 추론 (NPU)
                        outputs = pipeline.infer(inp)

                        # 3. 후처리 (박스 그리기)
                        try:
                            # 출력 텐서 파싱
                            raw_data_list = list(outputs.values())[0]
                            class_arrays = raw_data_list[0]
                           
                            detections = []
                            for class_idx, class_dets in enumerate(class_arrays):
                                if len(class_dets) == 0: continue
                                for det in class_dets:
                                    bbox, score = det[:4], det[4]
                                    if score >= CONF_THRESHOLD:
                                        detections.append({'box': bbox, 'score': score, 'class_id': class_idx})

                            # 화면에 그리기
                            for det in detections:
                                py1, px1, py2, px2 = det['box']
                                score = det['score']
                                class_id = det['class_id']
                                label = CLASSES.get(class_id, "Unknown")

                                # 좌표 변환 (0~1 -> 픽셀)
                                x1, y1 = int(px1 * w), int(py1 * h)
                                x2, y2 = int(px2 * w), int(py2 * h)

                                # 색상 설정 (사람:빨강, 헬멧/조끼:초록)
                                if label == 'Person': color = (0, 0, 255)
                                else: color = (0, 255, 0)

                                # 박스 및 텍스트
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                text = f"{label} {score:.2f}"
                                cv2.putText(frame, text, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            pass # 파싱 에러 무시

                        # 4. FPS 계산
                        curr_time = time.time()
                        fps = 1 / (curr_time - prev_time + 1e-6)
                        prev_time = curr_time
                        shared_status['fps'] = fps

                        # 화면 표시
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Hailo Detection', frame)
                        out_video.write(frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt: print("\n>>> 강제 종료")
    except Exception as e: print(f"\n>>> 오류: {e}")
    finally:
        shared_status['running'] = False
        if webcam: webcam.stop()
        if 'out_video' in locals(): out_video.release()
        cv2.destroyAllWindows()
        monitor.stop()

if __name__ == "__main__":
    run_visual_benchmark()
