import csv
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib  # 用于保存和加载模型

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

REFERENCE_HOLE_BOUNDS = (160.3027, 738.9948, 251.1068, 637.7195)

# 参考图中5个孔的位置
REFERENCE_BOXES_5 = [
    (220, 415, 50, 50),
    (270, 415, 50, 50),
    (405, 415, 50, 50),
    (570, 415, 50, 50),
    (630, 415, 50, 50),
]

MIN_MATCHED_COUNT = 3


def find_hole_bounds(gray):
    _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs, ys = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        M = cv2.moments(contour)
        if M["m00"]:
            xs.append(M["m10"] / M["m00"])
            ys.append(M["m01"] / M["m00"])
    if not xs:
        raise ValueError("没有检测到定位圆")
    return min(xs), max(xs), min(ys), max(ys)


def infer_boxes_from_relative_position(gray):
    left, right, top, bottom = find_hole_bounds(gray)
    ref_left, ref_right, ref_top, ref_bottom = REFERENCE_HOLE_BOUNDS
    scale_x = (right - left) / (ref_right - ref_left)
    scale_y = (bottom - top) / (ref_bottom - ref_top)
    boxes = []
    for x, y, w, h in REFERENCE_BOXES_5:
        boxes.append((
            round(left + (x - ref_left) * scale_x),
            round(top + (y - ref_top) * scale_y),
            round(w * scale_x),
            round(h * scale_y)
        ))
    return boxes


def extract_box_features(gray, box, box_index):
    """提取多维特征：[NCC得分, 均值, 标准差, 梯度幅值, 位置编号]"""
    x, y, width, height = box
    h_img, w_img = gray.shape
    if x < 0 or y < 0 or x + width > w_img or y + height > h_img:
        return np.zeros(5, dtype=np.float32)

    patch = gray[y: y + height, x: y + width]  # 修复高度越界切片
    patch = gray[y: y + height, x: x + width]

    # 1. 动态生成标准模板并计算 NCC 相似度
    t_w = max(int(width * 0.85), 5)
    t_h = max(int(height * 0.85), 5)
    template = np.ones((t_h, t_w), dtype=np.uint8) * 200
    sq_w, sq_h = int(t_w * 0.8), int(t_h * 0.8)
    ox, oy = (t_w - sq_w) // 2, (t_h - sq_h) // 2
    cv2.rectangle(template, (ox, oy), (ox + sq_w, oy + sq_h), 100, -1)
    radius = int(min(t_w, t_h) * 0.3)
    cv2.circle(template, (t_w // 2, t_h // 2), radius, 50, -1)
    template = cv2.GaussianBlur(template, (5, 5), 0)

    if patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]:
        res_score = 0.0
    else:
        res = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        res_score = float(max_val)

    # 2. 统计特征与边缘响应
    mean_val = float(np.mean(patch))
    std_val = float(np.std(patch))

    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    pos_id = float(box_index)

    return np.array([res_score, mean_val, std_val, grad_mag, pos_id], dtype=np.float32)


def train_svm(image_dir, labels_dict):
    """
    使用标注好的数据集训练 SVM
    labels_dict: 格式为 { '0002.png': [1, 0, 1, 1, 1], '0004.png': [1, 1, 1, 1, 1] }
                 1 表示该框正常(Pass)，0 表示异常(Fail)
    """
    X, y = [], []
    image_dir = Path(image_dir)
    for img_path in image_dir.iterdir():
        if img_path.name not in labels_dict:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            boxes = infer_boxes_from_relative_position(gray)
            slot_labels = labels_dict[img_path.name]
            for idx, box in enumerate(boxes, start=1):
                features = extract_box_features(gray, box, idx)
                X.append(features)
                y.append(slot_labels[idx - 1])
        except Exception as e:
            print(f"处理 {img_path.name} 训练特征时出错: {e}")

    X = np.array(X)
    y = np.array(y)

    # 建立包含标准化和 RBF 核 SVM 的 Pipeline
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10.0, probability=True))
    clf.fit(X, y)
    print("SVM 模型训练完成！")
    return clf


def evaluate_with_svm(gray, box, box_index, model):
    features = extract_box_features(gray, box, box_index).reshape(1, -1)
    prediction = model.predict(features)[0]
    matched = bool(prediction == 1)
    return float(prediction), matched


def process_image_svm(image_path, model, output_path=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes = infer_boxes_from_relative_position(gray)
    results = []
    for idx, box in enumerate(boxes, start=1):
        score, matched = evaluate_with_svm(gray, box, idx, model)
        results.append((score, 0.0, matched))

    matched_count = sum(r[2] for r in results)
    contrasts = [r[0] for r in results]
    median_contrast = float(np.median(contrasts)) if contrasts else 0.0
    passed = matched_count >= MIN_MATCHED_COUNT

    if output_path:
        output = image.copy()
        color = ((0, 200, 0) if passed else (0, 0, 255))
        for index, box in enumerate(boxes, start=1):
            x, y, w, h = box
            matched = results[index - 1][2]
            box_color = ((0, 255, 0) if matched else (0, 0, 255))
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(output, str(index), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        status = "Class II (Pass)" if passed else "Fail"
        cv2.putText(output, f"{status}: {matched_count}/5", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)

    return boxes, matched_count, median_contrast, passed