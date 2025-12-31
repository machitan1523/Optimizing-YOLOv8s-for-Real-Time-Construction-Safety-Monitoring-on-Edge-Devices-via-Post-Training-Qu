import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import os
import cv2
import time
import sys


HEF_FILE = "best_epoch200_1201_nms_350.1.hef"
INPUT_FOLDER = "/home/hongik/Desktop/512_folder/archive/css-data/valid/images"
OUTPUT_FOLDER = "results_detail_breakdown"

CLASSES = { 0: 'Person', 1: 'Hardhat', 2: 'Safety Vest' }

def is_inside(person_box, gear_box):
    py1, px1, py2, px2 = person_box
    gy1, gx1, gy2, gx2 = gear_box
    g_cx = (gx1 + gx2) / 2
    g_cy = (gy1 + gy2) / 2
    if (px1 < g_cx < px2) and (py1 < g_cy < py2):
        return True
    return False

def run_detail_latency_analysis():
    
    if not os.path.exists(INPUT_FOLDER):
        print("입력 폴더 없음")
        return
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
    total_files = len(image_files)
   
    if total_files == 0: return
    print(f"-> 총 {total_files}장의 이미지로 [초정밀] 지연 시간 분석을 시작합니다...")

    
    total_resize = 0.0      
    total_norm = 0.0       
    total_infer = 0.0       
    total_decode = 0.0      
    total_nms = 0.0         
    total_vis = 0.0         

    # 1. 모델 준비
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
                processed_count = 0

                
                for filename in image_files:
                    file_path = os.path.join(INPUT_FOLDER, filename)
                    image = cv2.imread(file_path)
                    if image is None: continue
                    h, w, _ = image.shape
                   
                    
                    t0 = time.perf_counter()
                   
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(image_rgb, (model_w, model_h))
                   
                    t1 = time.perf_counter() 

                    
                    input_data = resized_image.astype(np.float32) / 255.0
                    input_data = np.expand_dims(input_data, axis=0)
                   
                    t2 = time.perf_counter() 
                    
                    output_data = infer_pipeline.infer(input_data)
                   
                    t3 = time.perf_counter() 
                    
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
                    except: pass
                   
                    t4 = time.perf_counter() 

                    
                    persons = []
                    gears = []
                    for det in final_dets:
                        
                        box, score, class_id = det['box'], det['score'], det['class_id']
                        name = CLASSES.get(class_id, "Unknown")
                        py1, px1, py2, px2 = box
                        if name == 'Person':
                            persons.append({'box': [py1, px1, py2, px2], 'score': score})
                        elif name in ['Hardhat', 'Safety Vest']:
                            gears.append({'name': name, 'box': [py1, px1, py2, px2], 'score': score})
                   
                    t5 = time.perf_counter() 

                    
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
                            p_color = (0, 0, 255); status = "Unsafe"
                        else:
                            p_color = (255, 0, 0); status = "Safe"
                       
                        cv2.rectangle(image, (x1, y1), (x2, y2), p_color, 2)
                        cv2.putText(image, f"{status}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

                    save_path = os.path.join(OUTPUT_FOLDER, f"res_{filename}")
                    cv2.imwrite(save_path, image)

                    t6 = time.perf_counter() 

                    
                    total_resize += (t1 - t0)
                    total_norm   += (t2 - t1)
                    total_infer  += (t3 - t2)
                    total_decode += (t4 - t3)
                    total_nms    += (t5 - t4)
                    total_vis    += (t6 - t5)

                    processed_count += 1
                    if processed_count % 50 == 0: print(f" -> {processed_count}장 완료...")

                
                def ms(val): return (val / processed_count) * 1000

                print("\n" + "="*60)
                print(f" 🔬 [초정밀 지연 시간 분석 (Detail Latency Breakdown)]")
                print("="*60)
                print(f" [A] 전처리 (Pre-processing)")
                print(f"  ├─ 1. 리사이징 (Resizing)      : {ms(total_resize):.3f} ms")
                print(f"  └─ 2. 정규화 (Normalization)   : {ms(total_norm):.3f} ms")
                print("-" * 60)
                print(f" [B] 추론 (Inference)")
                print(f"  └─ 1. NPU 연산 + 데이터 전송   : {ms(total_infer):.3f} ms")
                print(f"      (※ Python API 특성상 전송/연산 통합 측정됨)")
                print("-" * 60)
                print(f" [C] 후처리 (Post-processing)")
                print(f"  ├─ 1. 디코딩 (Decoding)        : {ms(total_decode):.3f} ms")
                print(f"  ├─ 2. 로직/NMS (Logic)         : {ms(total_nms):.3f} ms")
                print(f"  └─ 3. 시각화 (Vis & Save)      : {ms(total_vis):.3f} ms")
                print("="*60)
                total_avg = ms(total_resize + total_norm + total_infer + total_decode + total_nms + total_vis)
                print(f" ✅ 전체 평균 합계 : {total_avg:.2f} ms")
                print("="*60)

if __name__ == "__main__":
    run_detail_latency_analysis()
