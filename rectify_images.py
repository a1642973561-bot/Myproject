import csv
from pathlib import Path
import cv2
import numpy as np
import joblib

from utils import get_base_path

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}

REFERENCE_HOLE_BOUNDS = (
    160.3027,
    738.9948,
    251.1068,
    637.7195,
)

# 参考图中5个孔的位置
REFERENCE_BOXES_5 = [
    (220, 415, 50, 50),
    (270, 415, 50, 50),
    (405, 415, 50, 50),
    (570, 415, 50, 50),
    (630, 415, 50, 50),
]

MIN_MATCHED_COUNT = 3

# 加载本地训练好的 SVM 模型（如果存在）
# 导入前面定义的路径函数
# from utils import get_base_path

SVM_MODEL_PATH = get_base_path() / "svm_model.pkl"
_loaded_svm_model = None


def get_svm_model():
    global _loaded_svm_model
    if _loaded_svm_model is None and SVM_MODEL_PATH.exists():
        try:
            _loaded_svm_model = joblib.load(SVM_MODEL_PATH)
            print("[SVM] 成功加载 SVM 模型进行智能判定。")
        except Exception as e:
            print(f"[SVM] 加载模型失败: {e}")
    return _loaded_svm_model


def find_hole_bounds(gray):
    _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs = []
    ys = []

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
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
            xs.append(x)
            ys.append(y)

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
    """提取多维特征：[NCC相似度得分, 均值, 标准差, 梯度幅值, 位置编号]"""
    x, y, width, height = box
    h_img, w_img = gray.shape

    if x < 0 or y < 0 or x + width > w_img or y + height > h_img:
        return np.zeros(5, dtype=np.float32)

    patch = gray[y: y + height, x: x + width]

    # 动态生成标准模板并计算 NCC
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

    mean_val = float(np.mean(patch))
    std_val = float(np.std(patch))

    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    pos_id = float(box_index)

    return np.array([res_score, mean_val, std_val, grad_mag, pos_id], dtype=np.float32)


def evaluate_slot_relative(gray, box, box_index):
    x, y, width, height = box
    h_img, w_img = gray.shape

    if x < 0 or y < 0 or x + width > w_img or y + height > h_img:
        return 0.0, 0.0, False

    patch = gray[y: y + height, x: x + width]

    # 计算 NCC 相似度
    t_w = max(int(width * 0.85), 5)
    t_h = max(int(height * 0.85), 5)
    template = np.ones((t_h, t_w), dtype=np.uint8) * 200
    sq_w, sq_h = int(t_w * 0.8), int(t_h * 0.8)
    ox, oy = (t_w - sq_w) // 2, (t_h - sq_h) // 2
    cv2.rectangle(template, (ox, oy), (ox + sq_w, oy + sq_h), 100, -1)
    radius = int(min(t_w, t_h) * 0.3)
    cv2.circle(template, (t_w // 2, t_h // 2), radius, 50, -1)
    template = cv2.GaussianBlur(template, (5, 5), 0)

    res = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    similarity_score = float(max_val)

    # 优先使用 SVM 模型判定
    model = get_svm_model()
    if model is not None:
        features = extract_box_features(gray, box, box_index).reshape(1, -1)
        prediction = model.predict(features)[0]
        matched = bool(prediction == 1)
        print(f"[SVM 判定] {box_index}号框 | 得分: {similarity_score:.4f} | 结果: {'通过' if matched else '失败'}")
    else:
        # 如果没有训练模型，则回退到原有的硬编码阈值
        matched = similarity_score >= 0.10
        print(f"[阈值判定] {box_index}号框 | 得分: {similarity_score:.4f} | 结果: {'通过' if matched else 'Fail'}")

    return similarity_score, 0.0, matched


def detect_square_circles(gray):
    boxes = infer_boxes_from_relative_position(gray)
    results = []

    for index, box in enumerate(boxes, start=1):
        results.append(evaluate_slot_relative(gray, box, index))

    matched_count = sum(r[2] for r in results)
    contrasts = [r[0] for r in results]
    median_contrast = float(np.median(contrasts)) if contrasts else 0.0
    passed = matched_count >= MIN_MATCHED_COUNT

    return boxes, results, matched_count, median_contrast, passed


def annotate_image(image, boxes, results, matched_count, median_contrast, passed):
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

    return output


def process_image(image_path, output_path=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, results, count, contrast, passed = detect_square_circles(gray)

    if output_path:
        output = annotate_image(image, boxes, results, count, contrast, passed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)

    return boxes, count, contrast, passed


def process_directory(input_dir, output_dir, save_annotations=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rows = []

    for image_path in sorted(input_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        output_path = (output_dir / f"{image_path.stem}_boxed.png" if save_annotations else None)

        try:
            boxes, count, contrast, passed = process_image(image_path, output_path)
            category = ("yes" if passed else "no")
        except Exception as e:
            print(f"{image_path.name} 错误: {e}")
            boxes, count, contrast, category = [], 0, 0.0, "no"

        rows.append({
            "filename": image_path.name,
            "matched_slot_count": count,
            "median_contrast": f"{contrast:.4f}",
            "has_square_circles": category,
            "box_coordinates": repr(boxes)
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "matched_slot_count", "median_contrast", "has_square_circles", "box_coordinates"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return rows