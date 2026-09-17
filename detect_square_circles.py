import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    from .train_svm import (
        evaluate_slot,
        infer_boxes_from_holes,
    )
except ImportError:
    from train_svm import (
        evaluate_slot,
        infer_boxes_from_holes,
    )

POSITIVE_LABELS = {
    "0032.png": [
        (261, 429),
        (325, 429),
        (487, 429),
        (678, 428),
        (748, 429),
    ],
    "0767.png": [(198, 456), (417, 463), (679, 473)],
}
NEGATIVE_IMAGES = [
    "0002.png",
    "0003.png",
    "0258.png",
    "0259.png",
    "0266.png",
    "0267.png",
]
EDGE_POSITIVE_LABELS = {
    "0032.png": (261, 429),
    "0516.png": (174, 447),
    "0517.png": (174, 447),
    "0518.png": (174, 446),
    "0519.png": (174, 446),
    "0767.png": (198, 456),
}
EDGE_NEGATIVE_LABELS = {"0767.png": [(159, 458)]}

PATCH_SIZE = 40
SCAN_STEP = 3
ROW_POSITION = 0.502
ROW_HALF_HEIGHT = 12
ROW_ALIGNMENT_TOLERANCE = 5
CENTER_RING_MEDIAN_CONTRAST_THRESHOLD = 5.0
CENTER_RING_MEDIAN_TEXTURE_THRESHOLD = 3.0
CENTER_RING_GEOMETRY_THRESHOLD = 0.68
CENTER_RING_TEMPLATE_THRESHOLD = 0.70
CENTER_RING_TEMPLATE_IMAGES = [
    "0270.png",
    "0322.png",
    "0368.png",
    "0378.png",
    "0382.png",
    "0502.png",
    "0504.png",
    "0506.png",
]
CENTER_RING_NEGATIVE_TEMPLATE_IMAGES = [
    "0416.png",
    "0486.png",
]
CENTER_RING_NEGATIVE_TEMPLATE_THRESHOLD = 0.70

HOG = cv2.HOGDescriptor(
    (32, 32),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def find_hole_centers(gray):
    _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if 350 <= area <= 1200 and circularity > 0.65:
            moments = cv2.moments(contour)
            if moments["m00"]:
                centers.append(
                    (
                        moments["m10"] / moments["m00"],
                        moments["m01"] / moments["m00"],
                    )
                )

    if len(centers) < 60:
        raise ValueError(f"黑色定位圆数量不足：{len(centers)}")
    return centers


def find_hole_bounds(gray):
    centers = find_hole_centers(gray)
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    return min(xs), max(xs), min(ys), max(ys)


def extract_feature(gray, center_x, center_y):
    half = PATCH_SIZE // 2
    patch = gray[
        center_y - half : center_y + half,
        center_x - half : center_x + half,
    ]
    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        return None

    patch = cv2.resize(patch, (32, 32))
    patch = CLAHE.apply(patch)
    return HOG.compute(patch).ravel()


def target_row_line(gray):
    left, right, top, bottom = find_hole_bounds(gray)
    center_y = top + ROW_POSITION * (bottom - top)
    center_x = (left + right) / 2

    # 六排定位孔在板面坐标中彼此平行，它们的中位斜率即目标行斜率。
    centers = sorted(find_hole_centers(gray), key=lambda point: point[1])
    slopes = []
    for row in np.array_split(np.asarray(centers), 6):
        if len(row) >= 5:
            slopes.append(float(np.polyfit(row[:, 0], row[:, 1], 1)[0]))
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = center_y - slope * center_x
    return round(left), round(right), slope, intercept


def target_row(gray):
    left, right, slope, intercept = target_row_line(gray)
    center_x = (left + right) / 2
    center_y = round(slope * center_x + intercept)
    return left, right, center_y


def train_detector(input_dir):
    features = []
    labels = []

    # 用户圈出的 8 个正样本，并通过轻微平移进行数据增强。
    for filename, points in POSITIVE_LABELS.items():
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        if gray is None:
            raise FileNotFoundError(f"无法读取训练图片：{filename}")

        for center_x, center_y in points:
            for offset_x in (-3, 0, 3):
                for offset_y in (-3, 0, 3):
                    feature = extract_feature(
                        gray,
                        center_x + offset_x,
                        center_y + offset_y,
                    )
                    features.append(feature)
                    labels.append(1)

    # 用户确认的无目标图片作为难负样本。
    for filename in NEGATIVE_IMAGES:
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        if gray is None:
            raise FileNotFoundError(f"无法读取训练图片：{filename}")

        left, right, center_y = target_row(gray)
        for y in (center_y - 8, center_y, center_y + 8):
            for x in range(left, right + 1, 8):
                feature = extract_feature(gray, x, y)
                if feature is not None:
                    features.append(feature)
                    labels.append(-1)

    # 正样本图片中远离标注框的位置也作为负样本。
    for filename, points in POSITIVE_LABELS.items():
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        left, right, center_y = target_row(gray)
        for x in range(left, right + 1, 8):
            if min(abs(x - point_x) for point_x, _ in points) > 30:
                feature = extract_feature(gray, x, center_y)
                if feature is not None:
                    features.append(feature)
                    labels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10)
    svm.setGamma(0.01)
    svm.train(
        np.asarray(features, dtype=np.float32),
        cv2.ml.ROW_SAMPLE,
        np.asarray(labels, dtype=np.int32),
    )
    return svm


def train_edge_detector(input_dir):
    features = []
    labels = []

    for filename, (center_x, center_y) in EDGE_POSITIVE_LABELS.items():
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        for offset_x in (-3, 0, 3):
            for offset_y in (-3, 0, 3):
                features.append(
                    extract_feature(
                        gray,
                        center_x + offset_x,
                        center_y + offset_y,
                    )
                )
                labels.append(1)

    for filename, points in EDGE_NEGATIVE_LABELS.items():
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        for center_x, center_y in points:
            for offset_x in (-3, 0, 3):
                for offset_y in (-3, 0, 3):
                    features.append(
                        extract_feature(
                            gray,
                            center_x + offset_x,
                            center_y + offset_y,
                        )
                    )
                    labels.append(-1)

    for filename in NEGATIVE_IMAGES:
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        left, _, center_y = target_row(gray)
        for y in range(center_y - 9, center_y + 10, 3):
            for x in range(left, left + 100, 4):
                feature = extract_feature(gray, x, y)
                if feature is not None:
                    features.append(feature)
                    labels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10)
    svm.setGamma(0.03)
    svm.train(
        np.asarray(features, dtype=np.float32),
        cv2.ml.ROW_SAMPLE,
        np.asarray(labels, dtype=np.int32),
    )
    return svm


def group_positive_windows(windows):
    if not windows:
        return []

    windows.sort(key=lambda point: point[0])
    groups = [[windows[0]]]
    for point in windows[1:]:
        if point[0] - groups[-1][-1][0] > 18:
            groups.append([point])
        else:
            groups[-1].append(point)

    detections = []
    for group in groups:
        # 单个偶然命中的滑窗不视为目标。
        if len(group) < 3:
            continue
        center_x = round(float(np.median([point[0] for point in group])))
        center_y = round(float(np.median([point[1] for point in group])))
        detections.append((center_x, center_y))
    return detections


def has_circular_center(gray, center_x, center_y):
    half = PATCH_SIZE // 2
    patch = gray[
        center_y - half : center_y + half,
        center_x - half : center_x + half,
    ].astype(np.float32)
    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        return False

    yy, xx = np.ogrid[-half:half, -half:half]
    radius = np.sqrt(xx * xx + yy * yy)
    center_mask = radius <= 8
    square_radius = np.maximum(np.abs(xx), np.abs(yy))
    corner_mask = (
        (square_radius >= 11)
        & (square_radius <= 16)
        & (radius >= 13)
    )

    # 六框的中心整体比四角亮；圆形目标的中心与四角亮度更接近。
    center_contrast = (
        float(patch[center_mask].mean())
        - float(patch[corner_mask].mean())
    )
    return center_contrast < 4.5


def scan_with_svm(gray, svm, left, right, center_y):
    positive_windows = []
    for y in range(
        center_y - ROW_HALF_HEIGHT,
        center_y + ROW_HALF_HEIGHT + 1,
        SCAN_STEP,
    ):
        for x in range(left, right + 1, SCAN_STEP):
            feature = extract_feature(gray, x, y)
            if feature is None:
                continue
            _, prediction = svm.predict(feature[None, :])
            if prediction[0, 0] > 0:
                positive_windows.append((x, y))
    return group_positive_windows(positive_windows)


def detect_patterns(gray, svm, edge_svm):
    left, right, slope, intercept = target_row_line(gray)
    center_y = round(slope * ((left + right) / 2) + intercept)
    detections = scan_with_svm(gray, svm, left, right, center_y)
    edge_detections = scan_with_svm(
        gray,
        edge_svm,
        left,
        min(right, left + 90),
        center_y,
    )

    filtered = [
        (center_x, round(slope * center_x + intercept))
        for center_x, detected_y in detections + edge_detections
        if (
            abs(detected_y - (slope * center_x + intercept))
            <= ROW_ALIGNMENT_TOLERANCE
            and has_circular_center(gray, center_x, detected_y)
        )
    ]
    filtered.sort()

    merged = []
    for detection in filtered:
        if not merged or detection[0] - merged[-1][0] > 45:
            merged.append(detection)
    return merged


def extract_center_ring_template_feature(gray):
    boxes = infer_boxes_from_holes(gray)
    features = []
    for x, y, width, height in boxes:
        patch = cv2.resize(
            gray[y : y + height, x : x + width],
            (24, 24),
        ).astype(np.float32)
        patch -= patch.mean()
        patch /= patch.std() + 1e-6
        features.append(patch.ravel())

    feature = np.concatenate(features)
    return feature / (np.linalg.norm(feature) + 1e-6)


def load_center_ring_templates(input_dir, filenames):
    templates = []
    for filename in filenames:
        gray = cv2.imread(
            str(input_dir / filename), cv2.IMREAD_GRAYSCALE
        )
        if gray is None:
            raise FileNotFoundError(f"无法读取环框模板图片：{filename}")
        templates.append(extract_center_ring_template_feature(gray))
    return templates


def center_ring_geometry_score(gray, boxes):
    half = 18
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    local_mask = np.maximum(np.abs(xx), np.abs(yy)) <= 16
    patches = []
    for x, y, width, height in boxes:
        center = (x + (width - 1) / 2, y + (height - 1) / 2)
        patches.append(
            cv2.getRectSubPix(gray, (37, 37), center).astype(np.float32)
        )

    best_score = -float("inf")
    # 在小范围内共同调整六个槽位，匹配等粗方框、灰环和中心亮圆。
    for offset_x in range(-4, 5, 2):
        for offset_y in range(-6, 7, 2):
            shifted_x = xx - offset_x
            shifted_y = yy - offset_y
            radius = np.hypot(shifted_x, shifted_y)
            square_radius = np.maximum(
                np.abs(shifted_x), np.abs(shifted_y)
            )
            for square_half_size in range(9, 15):
                frame_mask = (
                    np.abs(square_radius - square_half_size) <= 1.5
                )
                for center_radius in range(3, 7):
                    center_mask = radius <= center_radius
                    annulus_mask = (
                        (radius >= center_radius + 2)
                        & (radius <= center_radius + 5)
                    )
                    slot_scores = []
                    for patch in patches:
                        local_std = float(patch[local_mask].std()) + 1e-6
                        annulus_mean = float(patch[annulus_mask].mean())
                        center_contrast = (
                            float(patch[center_mask].mean())
                            - annulus_mean
                        )
                        frame_contrast = (
                            float(patch[frame_mask].mean())
                            - annulus_mean
                        )
                        slot_scores.append(
                            (center_contrast + frame_contrast) / local_std
                        )
                    best_score = max(
                        best_score, float(np.median(slot_scores))
                    )
    return best_score


def detect_center_ring_boxes(gray, templates, negative_templates):
    boxes = infer_boxes_from_holes(gray)
    slot_results = [evaluate_slot(gray, box) for box in boxes]
    median_contrast = float(
        np.median([result[0] for result in slot_results])
    )
    median_center_ring_contrast = float(
        np.median([result[2] for result in slot_results])
    )
    feature = extract_center_ring_template_feature(gray)
    template_score = max(
        float(np.dot(feature, template)) for template in templates
    )
    negative_template_score = max(
        float(np.dot(feature, template))
        for template in negative_templates
    )
    geometry_score = center_ring_geometry_score(gray, boxes)

    # 使用六个槽位的整体特征，避免局部曝光或线路纹理导致单槽漏检。
    # 极低对比度时，使用解析几何结构和归一化模板相似度补充。
    # 一旦确认属于该模板，六个固定位置均作为检测结果输出。
    detected = (
        (
            median_contrast >= CENTER_RING_MEDIAN_CONTRAST_THRESHOLD
            and median_center_ring_contrast
            >= CENTER_RING_MEDIAN_TEXTURE_THRESHOLD
        )
        or (
            geometry_score >= CENTER_RING_GEOMETRY_THRESHOLD
            and median_contrast >= 2.0
            and median_center_ring_contrast >= 0.5
        )
        or template_score >= CENTER_RING_TEMPLATE_THRESHOLD
    )
    if (
        detected
        and negative_template_score
        < CENTER_RING_NEGATIVE_TEMPLATE_THRESHOLD
    ):
        detections = boxes
    else:
        detections = []
    return (
        detections,
        slot_results,
        geometry_score,
        template_score,
        negative_template_score,
    )


def annotate_image(
    image,
    detections,
    center_ring_boxes,
    center_ring_checked,
):
    output = image.copy()
    half = PATCH_SIZE // 2

    for index, (center_x, center_y) in enumerate(detections, start=1):
        cv2.rectangle(
            output,
            (center_x - half, center_y - half),
            (center_x + half, center_y + half),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            output,
            str(index),
            (center_x - half, center_y - half - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    for index, (x, y, width, height) in enumerate(
        center_ring_boxes, start=1
    ):
        cv2.rectangle(
            output,
            (x - 3, y - 3),
            (x + width + 3, y + height + 3),
            (255, 255, 0),
            3,
        )
        cv2.putText(
            output,
            f"R{index}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    cv2.putText(
        output,
        f"square-circle count: {len(detections)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if detections else (0, 0, 255),
        2,
    )
    if center_ring_checked:
        cv2.putText(
            output,
            f"center-ring count: {len(center_ring_boxes)}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0) if center_ring_boxes else (0, 0, 255),
            2,
        )
    return output


def load_images_without_six_boxes(summary_path):
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            row["filename"]
            for row in csv.DictReader(file)
            if row["has_six_boxes"] == "no"
        ]


def process_directory(
    input_dir,
    six_box_summary,
    output_dir,
    save_annotations=True,
):
    svm = train_detector(input_dir)
    edge_svm = train_edge_detector(input_dir)
    center_ring_templates = load_center_ring_templates(
        input_dir, CENTER_RING_TEMPLATE_IMAGES
    )
    center_ring_negative_templates = load_center_ring_templates(
        input_dir, CENTER_RING_NEGATIVE_TEMPLATE_IMAGES
    )
    filenames = load_images_without_six_boxes(six_box_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for filename in filenames:
        image_path = input_dir / filename
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{filename}: 无法读取，跳过")
            continue

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (
                center_ring_boxes,
                center_ring_results,
                center_ring_geometry_score,
                center_ring_template_score,
                center_ring_negative_template_score,
            ) = detect_center_ring_boxes(
                gray,
                center_ring_templates,
                center_ring_negative_templates,
            )
            center_ring_checked = True
            if center_ring_boxes:
                # 两类结构互斥；确认六个环框后不再保留方圆形误检。
                detections = []
            else:
                detections = detect_patterns(
                    gray, svm, edge_svm
                )
        except ValueError as error:
            print(f"{filename}: {error}")
            continue

        if save_annotations:
            annotated = annotate_image(
                image,
                detections,
                center_ring_boxes,
                center_ring_checked,
            )
            output_path = (
                output_dir / f"{image_path.stem}_square_circle.png"
            )
            cv2.imwrite(str(output_path), annotated)
        rows.append(
            {
                "filename": filename,
                "square_circle_count": len(detections),
                "has_square_circle": "yes" if detections else "no",
                "detected_centers": repr(detections),
                "center_ring_checked": "yes",
                "center_ring_count": len(center_ring_boxes),
                "has_center_ring": (
                    "yes" if center_ring_boxes else "no"
                ),
                "center_ring_contrasts": repr(
                    [
                        round(result[2], 4)
                        for result in center_ring_results
                    ]
                ),
                "center_ring_geometry_score": (
                    f"{center_ring_geometry_score:.4f}"
                ),
                "center_ring_template_score": (
                    f"{center_ring_template_score:.4f}"
                ),
                "center_ring_negative_template_score": (
                    f"{center_ring_negative_template_score:.4f}"
                ),
                "center_ring_boxes": repr(center_ring_boxes),
            }
        )
        message = f"{filename}: 检测到 {len(detections)} 个方圆形"
        message += f"，{len(center_ring_boxes)} 个中心白圆灰环白框"
        print(message)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"标注结果：{output_dir}")
    print(f"汇总文件：{summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("zhangli/output_images"))
    parser.add_argument(
        "--six-box-summary",
        type=Path,
        default=Path("zhangli/boxed_results/summary.csv"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("zhangli/square_circle_results"),
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.six_box_summary, args.output)


if __name__ == "__main__":
    main()
