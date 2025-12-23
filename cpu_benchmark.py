import cv2
import time
import sys
import psutil
import csv
import threading
import matplotlib.pyplot as plt
import os
import signal
from ultralytics import YOLO


MODEL_FILE = "best.pt"       
SAVE_VIDEO_PATH = "cpu_benchmark_result.mp4"
CSV_FILENAME = "cpu_resource_log.csv"


class SystemMonitor:
    def __init__(self, filename=CSV_FILENAME, interval=1.0):
        self.filename = filename
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time_ref = None
        plt.switch_backend('Agg') 
       
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "CPU", "Memory", "Temperature"])

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except:
            return 0.0

    def _monitor_loop(self):
        self.start_time_ref = time.time()
        while self.running:
            try:
                curr_t = time.time() - self.start_time_ref
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().used / (1024 * 1024) 
                temp = self.get_cpu_temp()
               
                with open(self.filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([round(curr_t, 2), cpu, round(mem, 1), round(temp, 1)])
            except: pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f">>> [시스템] CPU 모드 자원 측정 시작...")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        print(">>> [시스템] 측정 종료. 그래프 저장 중...")
        self.save_graph()

    def save_graph(self):
        times, cpus, mems, temps = [], [], [], []
        try:
            if not os.path.exists(self.filename): return
            with open(self.filename, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    try:
                        times.append(float(row[0]))
                        cpus.append(float(row[1]))
                        mems.append(float(row[2]))
                        temps.append(float(row[3]))
                    except: continue
           
            if not times: return

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
           
            
            ax1.plot(times, cpus, color='tab:blue', label='CPU Usage')
            ax1.set_ylabel('CPU (%)'); ax1.set_ylim(0, 100); ax1.grid(True); ax1.legend()
            ax1.set_title('System Resource Monitoring (CPU Only Mode)')

            
            ax2.plot(times, mems, color='tab:green', label='Memory Used')
            ax2.set_ylabel('Memory (MB)'); ax2.grid(True); ax2.legend()

            
            ax3.plot(times, temps, color='tab:red', label='Temperature')
            ax3.axhline(y=80, color='gray', linestyle='--'); ax3.set_ylabel('Temp (C)'); ax3.grid(True); ax3.legend()
            ax3.set_xlabel('Time (s)')

            img_name = self.filename.replace('.csv', '.png')
            plt.tight_layout()
            plt.savefig(img_name)
            plt.close()
            print(f">>> [완료] 그래프 저장됨: {img_name}")
        except Exception as e:
            print(f"[에러] 그래프 생성 실패: {e}")


def run_cpu_benchmark():
    
    monitor = SystemMonitor(filename=CSV_FILENAME)
    monitor.start()

    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, 5.0, (640, 480)) 

    print(f">>> [모델 로딩] {MODEL_FILE} (CPU 모드)... 잠시만 기다리세요.")
    try:
        # YOLO 모델 로드 (.pt 파일)
        model = YOLO(MODEL_FILE)
       
        print(">>> [추론 시작] 속도가 매우 느릴 수 있습니다. (종료: 'q' 또는 Ctrl+C)")
       
        prev_time = time.time()
       
        while True:
            ret, frame = cap.read()
            if not ret: break

            
            results = model(frame, verbose=False)
           
            
            annotated_frame = results[0].plot()

            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            
            cv2.putText(annotated_frame, f"CPU FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("CPU Only Benchmark", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n>>> 강제 종료 요청됨")
    except Exception as e:
        print(f"\n>>> 오류 발생: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        monitor.stop() 
        print(">>> 프로그램 종료")

if __name__ == "__main__":
    run_cpu_benchmark()
