import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import os
import cv2
import sys
import time

# ==========================================
# 1. 설정
# ==========================================
HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
DEFAULT_VIDEO = "test.mp4" # 기본 비디오 파일명

CLASSES = {
    0: 'Person',
    1: 'Hardhat',
    2: 'Safety Vest'
}

# -------------------------------------------------
# [함수] 박스 포함 여부 확인
# -------------------------------------------------
def is_inside(person_box, gear_box):
    py1, px1, py2, px2 = person_box
    gy1, gx1, gy2, gx2 = gear_box
    g_center_x = (gx1 + gx2) / 2
    g_center_y = (gy1 + gy2) / 2
    if (px1 < g_center_x < px2) and (py1 < g_center_y < py2):
        return True
    return False

def run_video_inference():
    # 비디오 파일 경로 확인
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = DEFAULT_VIDEO

    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다 -> {video_path}")
        return

    # 1. 비디오 캡처 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 비디오를 열 수 없습니다.")
        return

    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"-> 비디오 분석 시작: {video_path}")
    print(f"-> 해상도: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")

    # 결과 저장을 위한 VideoWriter 설정
    save_path = f"result_{os.path.basename(video_path)}"
    # mp4 저장을 위한 코덱 (라즈베리파이 등 환경에 따라 'XVID'나 'MJPG'로 변경 필요할 수 있음)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 2. Hailo 모델 로딩 (루프 밖에서 한 번만 실행)
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
                
                # 워밍업
                dummy = np.zeros((1, model_h, model_w, 3), dtype=np.float32)
                infer_pipeline.infer(dummy)

                frame_count = 0
                start_time = time.time()

                # 3. 프레임 반복 처리 (Loop)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break # 영상 끝

                    # 전처리
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(image_rgb, (model_w, model_h))
                    input_data = resized_image.astype(np.float32) / 255.0
                    input_data = np.expand_dims(input_data, axis=0)

                    # 추론
                    output_data = infer_pipeline.infer(input_data)
                    
                    # 파싱 & 그리기 로직 (기존 코드와 동일)
                    raw_data_list = list(output_data.values())[0]
                    final_dets = []
                    try:
                        class_arrays = raw_data_list[0]
                        for class_idx, class_detections in enumerate(class_arrays):
                            if len(class_detections) == 0: continue
                            for det in class_detections:
                                bbox, score = det[:4], det[4]
                                if score >= 0.25:
                                    final_dets.append({'box': bbox, 'score': score, 'class_id': class_idx})
                    except:
                        pass

                    # 사람/장비 분류
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

                    # 그리기
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

                    # 처리된 프레임을 영상 파일에 쓰기
                    out.write(frame)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f" -> 처리 중... {frame_count}/{total_frames} 프레임")

    # 종료 처리
    cap.release()
    out.release()
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*40)
    print(f" [완료] 결과 저장됨: {save_path}")
    print(f" 총 소요 시간: {duration:.2f}초")
    print(f" 평균 처리 속도: {frame_count/duration:.2f} FPS")
    print("="*40)

if __name__ == "__main__":
    run_video_inference()
