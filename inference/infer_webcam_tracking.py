import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import cv2
import time
import sys
from threading import Thread, Lock


HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
WEBCAM_ID = 0 
CLASSES = { 0: 'Person', 1: 'Hardhat', 2: 'Safety Vest' }

# [튜닝 포인트]
CONF_THRESHOLD = 0.55
SMOOTH_FACTOR = 0.2 
MISS_TOLERANCE = 5 


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
    # box: [y1, x1, y2, x2]
    ay1, ax1, ay2, ax2 = boxA
    by1, bx1, by2, bx2 = boxB

    yA = max(ay1, by1)
    xA = max(ax1, bx1)
    yB = min(ay2, by2)
    xB = min(ax2, bx2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (ay2 - ay1) * (ax2 - ax1)
    boxBArea = (by2 - by1) * (bx2 - bx1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class TrackedObject:
    def __init__(self, box, score, class_id):
        self.box = box
        self.score = score
        self.class_id = class_id
        self.missed_frames = 0 

    def update(self, new_box, new_score):
        
        alpha = SMOOTH_FACTOR
        self.box = [
            self.box[0] * (1 - alpha) + new_box[0] * alpha,
            self.box[1] * (1 - alpha) + new_box[1] * alpha,
            self.box[2] * (1 - alpha) + new_box[2] * alpha,
            self.box[3] * (1 - alpha) + new_box[3] * alpha
        ]
        self.score = new_score
        self.missed_frames = 0 


trackers = []

def update_trackers(detections):
    global trackers
    
    
    matched_indices = []
    
    for det in detections:
        best_iou = 0
        best_tracker_idx = -1
        
       
        for i, trk in enumerate(trackers):
            if trk.class_id != det['class_id']: continue 
            
            iou = compute_iou(det['box'], trk.box)
            if iou > best_iou:
                best_iou = iou
                best_tracker_idx = i
        
        
        if best_iou > 0.3 and best_tracker_idx != -1:
            trackers[best_tracker_idx].update(det['box'], det['score'])
            matched_indices.append(best_tracker_idx)
        else:
            
            new_trk = TrackedObject(det['box'], det['score'], det['class_id'])
            trackers.append(new_trk)
            
   
    active_trackers = []
    for i, trk in enumerate(trackers):
        if i in matched_indices:
            active_trackers.append(trk)
        else:
            trk.missed_frames += 1
            if trk.missed_frames < MISS_TOLERANCE:
                active_trackers.append(trk)
    
    trackers = active_trackers
    
    
    final_results = []
    for trk in trackers:
        final_results.append({
            'box': trk.box,
            'score': trk.score,
            'class_id': trk.class_id
        })
    return final_results


def is_inside(person_box, gear_box):
    py1, px1, py2, px2 = person_box
    gy1, gx1, gy2, gx2 = gear_box
    g_cx, g_cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    return (px1 < g_cx < px2) and (py1 < g_cy < py2)

def run_tracker_inference():
    webcam = ThreadedCamera(WEBCAM_ID).start()
    time.sleep(1.0) 

    print(f"-> 강력한 떨림 방지(Tracker) 적용 시작 (종료: 'q')")

    hef = HEF(HEF_FILE)
    params = VDevice.create_params()

    with VDevice(params) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_vstream_info = hef.get_input_vstream_infos()[0]
        model_w, model_h = input_vstream_info.shape[1], input_vstream_info.shape[0]

        with network_group.activate(network_group_params):
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                
                infer_pipeline.infer(np.zeros((1, model_h, model_w, 3), dtype=np.float32))
                prev_time = 0

                while True:
                    frame = webcam.read()
                    if frame is None: continue

                    
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(image_rgb, (model_w, model_h))
                    input_data = resized_image.astype(np.float32) / 255.0
                    input_data = np.expand_dims(input_data, axis=0)

                    
                    output_data = infer_pipeline.infer(input_data)
                    
                    
                    raw_data_list = list(output_data.values())[0]
                    current_dets = []
                    try:
                        class_arrays = raw_data_list[0]
                        for class_idx, class_detections in enumerate(class_arrays):
                            if len(class_detections) == 0: continue
                            for det in class_detections:
                                bbox, score = det[:4], det[4]
                                if score >= CONF_THRESHOLD: 
                                    current_dets.append({'box': bbox, 'score': score, 'class_id': class_idx})
                    except:
                        pass
                    
                    
                    final_dets = update_trackers(current_dets)

                    
                    persons = []
                    gears = [] 
                    h, w, _ = frame.shape

                    for det in final_dets:
                        box, score, class_id = det['box'], det['score'], det['class_id']
                        name = CLASSES.get(class_id, "Unknown")
                        py1, px1, py2, px2 = box
                        if name == 'Person':
                            persons.append({'box': [py1, px1, py2, px2], 'score': score})
                        elif name in ['Hardhat', 'Safety Vest']:
                            gears.append({'name': name, 'box': [py1, px1, py2, px2], 'score': score})

                    for p in persons:
                        p_box = p['box']
                        py1, px1, py2, px2 = p_box
                        x1, y1 = int(px1 * w), int(py1 * h)
                        x2, y2 = int(px2 * w), int(py2 * h)
                        
                        wearing_helmet = False
                        wearing_vest = False
                        
                        for g in gears:
                            if is_inside(p_box, g['box']):
                                gy1, gx1, gy2, gx2 = g['box']
                                gx1, gy1 = int(gx1 * w), int(gy1 * h)
                                gx2, gy2 = int(gx2 * w), int(gy2 * h)
                                g_name = g['name']
                                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                                cv2.putText(frame, g_name, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                if g_name == 'Hardhat': wearing_helmet = True
                                if g_name == 'Safety Vest': wearing_vest = True
                        
                        if not wearing_helmet or not wearing_vest:
                            p_color = (0, 0, 255) # Unsafe
                            status = "Unsafe"
                        else:
                            p_color = (255, 0, 0) # Safe
                            status = "Safe"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, 2)
                        cv2.putText(frame, f"{status}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

                        if not wearing_helmet:
                            head_h = int((y2 - y1) / 6)
                            cv2.rectangle(frame, (x1, y1), (x2, y1 + head_h), (0, 0, 255), 2)
                            cv2.putText(frame, "NO-Hardhat", (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if not wearing_vest:
                            body_y1 = y1 + int((y2 - y1) / 5)
                            cv2.putText(frame, "NO-Vest", (x1, body_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    cv2.imshow('Tracker Safety Monitor', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    webcam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker_inference()
