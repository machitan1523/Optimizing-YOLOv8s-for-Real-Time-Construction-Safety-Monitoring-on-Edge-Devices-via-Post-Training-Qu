import os
import numpy as np
import glob

# ======================================================================================
# [설정 구역] 경로를 본인 환경에 맞게 수정하세요
# ======================================================================================
GT_PATH = "/home/hongik/Desktop/mAP_1218_TJ/archive/css-data/valid/labels"      # 정답지 폴더
PRED_PATH = "/home/hongik/Desktop/512_folder/prediction_350.1_TJ"       # 추론 결과 폴더 (v12 결과 권장)

# 평가할 클래스 ID 목록
TARGET_CLASSES = [0, 5, 7]
CLASS_NAMES = {0: 'Hardhat', 5: 'Person', 7: 'Safety Vest'}

# [중요] F1-Score 계산을 위한 점수 커트라인 (보통 0.5 사용)
# 이 점수보다 낮은 박스는 "없는 셈" 치고 계산합니다.
CONF_THRESHOLD = 0.5
# ======================================================================================

def parse_txt(file_path):
    """ txt 파일을 읽어서 [class, x, y, w, h, score] 리스트로 반환 """
    boxes = []
    if not os.path.exists(file_path):
        return np.array([])
   
    with open(file_path, 'r') as f:
        lines = f.readlines()
       
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])
        x, y, w, h = parts[1], parts[2], parts[3], parts[4]
        # 추론 결과에는 score가 있지만, GT에는 없으므로 1.0으로 처리
        score = parts[5] if len(parts) > 5 else 1.0
        boxes.append([cls, x, y, w, h, score])
       
    return np.array(boxes)

def compute_iou(box1, box2):
    """ IoU(Intersection over Union) 계산 """
    b1_x1, b1_x2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
    b1_y1, b1_y2 = box1[2] - box1[4]/2, box1[2] + box1[4]/2
   
    b2_x1, b2_x2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2
    b2_y1, b2_y2 = box2[2] - box2[4]/2, box2[2] + box2[4]/2
   
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
   
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
   
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
   
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(recalls, precisions):
    """ Precision-Recall 곡선 아래 면적(AP) 계산 """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
   
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
       
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def calculate_map():
    print(f"[평가 시작] GT: {GT_PATH}")
    print(f"[평가 대상] PRED: {PRED_PATH}")
    print(f"[설정] Confidence Threshold: {CONF_THRESHOLD}")
   
    pred_files = glob.glob(os.path.join(PRED_PATH, "*.txt"))
    if len(pred_files) == 0:
        print("경고: 추론 결과 파일(.txt)을 찾을 수 없습니다.")
        return

    aps = []
   
    # -------------------------------------------------------------
    # [출력 헤더] 논문에 넣기 좋게 표 형식으로 출력
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print(f"{'Class':<12} {'AP@0.5':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'GT Count':<10}")
    print("="*85)

    for c in TARGET_CLASSES:
        true_positives = []
        scores = []
        n_gt = 0
       
        # 모든 파일 순회
        for pred_file in pred_files:
            file_id = os.path.basename(pred_file)
            gt_file = os.path.join(GT_PATH, file_id)
           
            pred_boxes = parse_txt(pred_file)
            gt_boxes = parse_txt(gt_file)
           
            # 현재 클래스(c)만 필터링
            if len(pred_boxes) > 0:
                pred_c = pred_boxes[pred_boxes[:, 0] == c]
            else:
                pred_c = np.array([])
               
            if len(gt_boxes) > 0:
                gt_c = gt_boxes[gt_boxes[:, 0] == c]
            else:
                gt_c = np.array([])
           
            n_gt += len(gt_c)
           
            if len(pred_c) == 0 and len(gt_c) == 0:
                continue
           
            # 예측은 없는데 정답은 있는 경우 (FN)
            if len(pred_c) == 0:
                continue
           
            # 정답은 없는데 예측만 있는 경우 (FP)
            if len(gt_c) == 0:
                for _ in pred_c:
                    scores.append(_[5])
                    true_positives.append(0)
                continue
           
            # 매칭 시작 (Score 높은 순으로 정렬)
            pred_c = pred_c[(-pred_c[:, 5]).argsort()]
            gt_detected = [False] * len(gt_c)
           
            for p_box in pred_c:
                scores.append(p_box[5])
               
                best_iou = 0
                best_gt_idx = -1
               
                # 가장 IoU가 높은 정답 박스 찾기
                for i, g_box in enumerate(gt_c):
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
               
                # IoU가 0.5 이상이면 정답 후보
                if best_iou >= 0.5:
                    if not gt_detected[best_gt_idx]:
                        true_positives.append(1) # 정답 (TP)
                        gt_detected[best_gt_idx] = True
                    else:
                        true_positives.append(0) # 이미 찾은 거 또 찾음 (FP - 중복)
                else:
                    true_positives.append(0) # 위치 틀림 (FP)
                   
        # 데이터가 없으면 건너뜀
        if n_gt == 0:
            print(f"{CLASS_NAMES[c]:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {0:<10}")
            continue
           
        scores = np.array(scores)
        true_positives = np.array(true_positives)
       
        # 1. AP(mAP) 계산용 정렬
        indices = np.argsort(-scores)
        tp_sorted = true_positives[indices]
       
        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(1 - tp_sorted)
       
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

        # -------------------------------------------------------------------------
        # [핵심 수정] Precision, Recall, F1-Score 계산 로직
        # -------------------------------------------------------------------------
        # 설정한 Threshold(0.5)보다 낮은 점수의 박스는 다 버림
        valid_indices = scores >= CONF_THRESHOLD
       
        # 남은 것들 중에서...
        # TP(정답): 맞은 개수
        tp_count = np.sum(true_positives[valid_indices])
       
        # FP(오답): (Threshold 넘은 전체 개수) - (맞은 개수)
        fp_count = len(true_positives[valid_indices]) - tp_count
       
        # FN(놓침): (전체 정답 개수) - (맞은 개수)
        fn_count = n_gt - tp_count
       
        # 수식 적용 (0으로 나누기 방지용 epsilon 추가)
        epsilon = 1e-7
        final_precision = tp_count / (tp_count + fp_count + epsilon)
        final_recall = tp_count / (tp_count + fn_count + epsilon)
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + epsilon)
        # -------------------------------------------------------------------------
       
        # 결과 출력
        print(f"{CLASS_NAMES[c]:<12} {ap:.4f}     {final_precision:.4f}       {final_recall:.4f}     {final_f1:.4f}     {n_gt:<10}")

    print("="*85)
    if len(aps) > 0:
        print(f"\n✅ 최종 mAP@0.5: {np.mean(aps):.4f}")
    else:
        print("\n평가할 데이터가 없습니다.")

if __name__ == "__main__":
    calculate_map()
