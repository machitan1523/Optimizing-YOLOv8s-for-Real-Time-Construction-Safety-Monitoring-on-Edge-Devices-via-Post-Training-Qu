import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import cv2
import time
import sys
import csv
from threading import Thread, Lock

# ==========================================
# 설정
# ==========================================
HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
WEBCAM_ID = 0
CLASSES = { 0: 'Person', 1: 'Hardhat', 2: 'Safety Vest' }

# 트래커 설정
CONF_THRESHOLD = 0.55
SMOOTH_FACTOR = 0.2
MISS_TOLERANCE = 5

# 저장 파일명
VIDEO_FILENAME = "evidence_video.mp4"
CSV_FILENAME = "performance_log.csv"

# ==========================================
# 카메라 & 트래커 클래스 (기존과 동일)
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
            else:
                self.stopped = True

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        self.stopped = True
        self.capture.release()

def compute_iou(boxA, boxB):
    ay1, ax1, ay2, ax2 = boxA
    by1, bx1, by2, bx2 = boxB
    yA = max(ay1, by1); xA = max(ax1, bx1)
    yB = min(ay2, by2); xB = min(ax2, bx2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (ay2 - ay1) * (ax2 - ax1)
    boxBArea = (by2 - by1) * (bx2 - bx1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

class TrackedObject:
    def __init__(self, box, score, class_id):
        self.box = box
        self.score = score
        self.class_id = class_id
        self.missed_frames = 0
    def update(self, new_box, new_score):
        alpha = SMOOTH_FACTOR
        self.box = [self.box[0]*(1-alpha)+new_box[0]*alpha, self.box[1]*(1-alpha)+new_box[1]*alpha, 
                    self.box[2]*(1-alpha)+new_box[2]*alpha, self.box[3]*(1-alpha)+new_box[3]*alpha]
        self.score = new_score
        self.missed_frames = 0

trackers = []
def update_trackers(detections):
    global trackers
    matched_indices = []
    for det in detections:
        best_iou = 0; best_tracker_idx = -1
        for i, trk in enumerate(trackers):
            if trk.class_id != det['class_id']: continue
            iou = compute_iou(det['box'], trk.box)
            if iou > best_iou: best_iou = iou; best_tracker_idx = i
        if best_iou > 0.3 and best_tracker_idx != -1:
            trackers[best_tracker_idx].update(det['box'], det['score'])
            matched_indices.append(best_tracker_idx)
        else:
            trackers.append(TrackedObject(det['box'], det['score'], det['class_id']))
    active_trackers = []
    for i, trk in enumerate(trackers):
        if i in matched_indices: active_trackers.append(trk)
        else:
            trk.missed_frames += 1
            if trk.missed_frames < MISS_TOLERANCE: active_trackers.append(trk)
    trackers = active_trackers
    return [{'box': t.box, 'score': t.score, 'class_id': t.class_id} for t in trackers]

def is_inside(person_box, gear_box):
    py1, px1, py2, px2 = person_box
    gy1, gx1, gy2, gx2 = gear_box
    g_cx, g_cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    return (px1 < g_cx < px2) and (py1 < g_cy < py2)

def run_recorder():
    webcam = ThreadedCamera(WEBCAM_ID).start()
    time.sleep(1.0)
    
    # 영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 30.0, (640, 480))
    
    # CSV 로그 파일 열기
    csv_file = open(CSV_FILENAME, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "FPS", "Person_Count", "Unsafe_Count"])

    print(f"-> 증거 수집 시작 (종료: 'q' 또는 Ctrl+C)")
    print(f"-> 영상 저장: {VIDEO_FILENAME}")
    print(f"-> 로그 저장: {CSV_FILENAME}")

    hef = HEF(HEF_FILE)
    params = VDevice.create_params()

    try: # [추가] 여기서부터 에러 감지 시작
        with VDevice(params) as target:
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()
            input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            input_info = hef.get_input_vstream_infos()[0]
            model_w, model_h = input_info.shape[1], input_info.shape[0]

            with network_group.activate(network_group_params):
                with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                    infer_pipeline.infer(np.zeros((1, model_h, model_w, 3), dtype=np.float32))
                    
                    prev_time = time.time()
                    start_record_time = time.time()

                    while True:
                        frame = webcam.read()
                        if frame is None: continue

                        # 추론
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        input_data = np.expand_dims(cv2.resize(img_rgb, (model_w, model_h)).astype(np.float32)/255.0, axis=0)
                        output_data = infer_pipeline.infer(input_data)
                        
                        # 파싱 & 트래킹
                        raw_data = list(output_data.values())[0]
                        curr_dets = []
                        try:
                            for c_idx, c_dets in enumerate(raw_data[0]):
                                if len(c_dets) == 0: continue
                                for d in c_dets:
                                    if d[4] >= CONF_THRESHOLD: curr_dets.append({'box': d[:4], 'score': d[4], 'class_id': c_idx})
                        except: pass
                        
                        final_dets = update_trackers(curr_dets)

                        # 그리기 & 통계
                        persons = []; gears = []
                        h, w, _ = frame.shape
                        for d in final_dets:
                            cls = d['class_id']; box = d['box']
                            if CLASSES[cls] == 'Person': persons.append(d)
                            elif CLASSES[cls] in ['Hardhat', 'Safety Vest']: gears.append(d)

                        unsafe_count = 0
                        for p in persons:
                            py1, px1, py2, px2 = p['box']
                            x1, y1 = int(px1*w), int(py1*h); x2, y2 = int(px2*w), int(py2*h)
                            helmet=False; vest=False
                            for g in gears:
                                if is_inside(p['box'], g['box']):
                                    gy1, gx1, gy2, gx2 = g['box']
                                    gx1, gy1 = int(gx1*w), int(gy1*h); gx2, gy2 = int(gx2*w), int(gy2*h)
                                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0,255,0), 2)
                                    if CLASSES[g['class_id']] == 'Hardhat': helmet=True
                                    if CLASSES[g['class_id']] == 'Safety Vest': vest=True
                            
                            status = "Safe" if (helmet and vest) else "Unsafe"
                            color = (255,0,0) if status == "Safe" else (0,0,255)
                            if status == "Unsafe": unsafe_count += 1
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"Person ({status})", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # FPS 계산
                        curr_time = time.time()
                        fps = 1 / (curr_time - prev_time)
                        prev_time = curr_time
                        
                        # 로그 저장
                        writer.writerow([f"{curr_time - start_record_time:.2f}", f"{fps:.2f}", len(persons), unsafe_count])

                        # 화면 표시 & 저장
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Evidence Recorder', frame)
                        out_video.write(frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt:
        print("\n-> [경고] 강제 종료(Ctrl+C) 감지! 데이터를 저장합니다...")
    
    except Exception as e:
        print(f"\n-> [오류] 예기치 못한 오류 발생: {e}")

    finally: # [중요] 무슨 일이 있어도 여긴 꼭 실행됨
        webcam.stop()
        out_video.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print("-> [완료] 증거 영상과 로그 파일이 안전하게 저장되었습니다.")

if __name__ == "__main__":
    run_recorder()
