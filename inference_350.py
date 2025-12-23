import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import os
import cv2
import sys


HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
DEFAULT_IMAGE = "test_occlusion(2).jpg"


CLASSES = {
    0: 'Person',
    1: 'Hardhat',
    2: 'Safety Vest'
}


def is_inside(person_box, gear_box):
    # person_box, gear_box: [ymin, xmin, ymax, xmax]
    py1, px1, py2, px2 = person_box
    gy1, gx1, gy2, gx2 = gear_box
    
    
    g_center_x = (gx1 + gx2) / 2
    g_center_y = (gy1 + gy2) / 2
    
    
    if (px1 < g_center_x < px2) and (py1 < g_center_y < py2):
        return True
    return False

def run_inference():
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = DEFAULT_IMAGE

    image = cv2.imread(target_file)
    if image is None:
        print(f"오류: {target_file} 파일을 찾을 수 없습니다.")
        return

    print(f"-> 분석 시작: {target_file}")

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        
        resized_image = cv2.resize(image_rgb, (model_w, model_h))
        
        input_data = resized_image.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        with network_group.activate(network_group_params):
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                output_data = infer_pipeline.infer(input_data)
                
                
                
                raw_data_list = list(output_data.values())[0]

                final_dets = []

                try:
                    
                    class_arrays = raw_data_list[0]
                    
                    
                    
                    for class_idx, class_detections in enumerate(class_arrays):
                        
                        
                        if len(class_detections) == 0:
                            continue
                            
                        
                        for det in class_detections:
                            
                            bbox = det[:4] 
                            score = det[4]
                            
                            if score >= 0.25: # Threshold
                                final_dets.append({
                                    'box': bbox,
                                    'score': score,
                                    'class_id': class_idx 
                                })

                    print(f"[디버깅] 최종 유효 탐지 개수: {len(final_dets)}")

                except Exception as e:
                    print(f"\n[파싱 오류] {e}")
                    
                    print(f"데이터 구조: {raw_data_list}")
                    return

                
                persons = []
                gears = [] 
                
                h, w, _ = image.shape

                for det in final_dets:
                    box = det['box']
                    score = det['score']
                    class_id = det['class_id']
                    
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
                            
                            cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                            cv2.putText(image, g_name, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if g_name == 'Hardhat': wearing_helmet = True
                            if g_name == 'Safety Vest': wearing_vest = True
                    
                   
                    if not wearing_helmet or not wearing_vest:
                        p_color = (0, 0, 255) # 위험 (빨강)
                        status = "Unsafe"
                    else:
                        p_color = (255, 0, 0) # 안전 (파랑)
                        status = "Safe"
                        
                    cv2.rectangle(image, (x1, y1), (x2, y2), p_color, 2)
                    cv2.putText(image, f"Person ({status})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

                    
                    if not wearing_helmet:
                        head_h = int((y2 - y1) / 6)
                        head_y2 = y1 + head_h
                        label = "NO-Hardhat"
                        text_s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        cv2.rectangle(image, (x1, y1), (x2, head_y2), (0, 0, 255), 2)
                        cv2.rectangle(image, (x1, head_y2), (x1 + text_s[0], head_y2 + 20), (0, 0, 255), -1)
                        cv2.putText(image, label, (x1, head_y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if not wearing_vest:
                        body_y1 = y1 + int((y2 - y1) / 5)
                        body_y2 = y1 + int((y2 - y1) / 2)
                        label = "NO-Vest"
                        text_s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        cv2.rectangle(image, (x1, body_y1), (x2, body_y2), (0, 0, 255), 2)
                        cv2.rectangle(image, (x1, body_y1), (x1 + text_s[0], body_y1 + 20), (0, 0, 255), -1)
                        cv2.putText(image, label, (x1, body_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                print(f"-> 처리 완료: {len(persons)}명 검사됨")
                save_name = f"result_{target_file}"
                cv2.imwrite(save_name, image)
                print(f"-> 결과 저장됨: {save_name}")

if __name__ == "__main__":
    run_inference()
