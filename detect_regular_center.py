import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import joblib

from utils import get_base_path
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SVM_MODEL_PATH = get_base_path() / "svm_regular_model.pkl"


def load_remaining_images(square_summary_path):
    """
    读取方圆孔检测结果的 summary.csv。
    剔除已经被识别为方圆孔 (has_square_circles == 'yes') 的图片。
    返回需要继续检测的图片文件名集合。
    """
    remaining = set()
    if not square_summary_path.exists():
        return remaining
    with square_summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("has_square_circles", "no") == "no":
                remaining.add(row["filename"])
    return remaining


def extract_center_features(gray_img, ref_gray=None):
    """
    提取中心区域的特征向量：
    [NCC相似度, 均值, 标准差, 拉普拉斯方差]
    """
    h, w = gray_img.shape
    crop = gray_img[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]

    mean_val = np.mean(crop)
    std_val = np.std(crop)
    lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()

    features = [mean_val, std_val, lap_var]

    if ref_gray is not None:
        ref_h, ref_w = ref_gray.shape
        if ref_gray.shape != (h, w):
            ref_crop = ref_gray[int(ref_h * 0.3):int(ref_h * 0.7), int(ref_w * 0.3):int(ref_w * 0.7)]
        else:
            ref_crop = ref_gray[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]

        if crop.shape != ref_crop.shape:
            target_resized = cv2.resize(crop, (ref_crop.shape[1], ref_crop.shape[0]))
        else:
            target_resized = crop

        res = cv2.matchTemplate(target_resized, ref_crop, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        features.insert(0, float(max_val))
    else:
        features.insert(0, 1.0)

    return np.array(features, dtype=np.float32)


def process_directory(
        input_dir,
        square_summary,
        reference_image,
        output_dir,
        model_path=SVM_MODEL_PATH,
        save_annotations=True,
):
    """
    处理整个目录，使用 SVM 模型执行常规中心特征检测。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 SVM 模型
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"未找到 SVM 模型文件：{model_path}，请先运行训练脚本生成模型！")
    clf = joblib.load(model_path)

    # 2. 筛选出待检测图片集合
    remaining_images = load_remaining_images(Path(square_summary))

    # 3. 读取参考图
    ref_path = Path(reference_image)
    ref_img_gray = None
    if ref_path.exists():
        ref_img_gray = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

    # 4. 遍历原图目录
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    rows = []

    for img_path in image_paths:
        filename = img_path.name

        # 如果该图已经被方圆孔流程截胡了，直接标记为跳过/不匹配
        if filename not in remaining_images:
            rows.append({
                "filename": filename,
                "similarity": "N/A",
                "is_regular_center": "no",
            })
            continue

        # 读取灰度图
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"警告: 无法读取图片 {filename}")
            continue

        # 提取特征并使用 SVM 预测
        features = extract_center_features(gray, ref_img_gray)
        pred_label = clf.predict([features])[0]  # 1 表示通过（第一类），0 表示不通过

        # 获取置信度分数
        try:
            probs = clf.predict_proba([features])[0]
            score = float(probs[1])
        except Exception:
            score = float(pred_label)

        passed = (pred_label == 1)
        category = "yes" if passed else "no"

        rows.append({
            "filename": filename,
            "similarity": f"{score:.4f}",
            "is_regular_center": category,
        })

        print(f"{filename}: SVM预测={'符合常规中心(Class 1)' if passed else '不符合'}, 置信度={score:.4f}")

        # 保存标注图
        if passed and save_annotations:
            output_img_path = output_dir / f"{img_path.stem}_regular.png"
            color_img = cv2.imread(str(img_path))
            if color_img is not None:
                cv2.putText(
                    color_img,
                    f"Class I (Pass): Score={score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(output_img_path), color_img)

    # 5. 生成 summary.csv
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["filename", "similarity", "is_regular_center"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(description="常规中心特征检测（基于 SVM 分类器）")
    parser.add_argument("--input_dir", type=Path, required=True, help="原始图片目录")
    parser.add_argument("--square_summary", type=Path, required=True, help="方圆孔检测结果 csv 文件路径")
    parser.add_argument("--reference_image", type=Path, required=True, help="标准参考图路径")
    parser.add_argument("--output_dir", type=Path, required=True, help="结果输出目录")
    parser.add_argument("--model_path", type=Path, default=SVM_MODEL_PATH, help="SVM 模型路径")
    parser.add_argument("--no-annotations", action="store_true", help="关闭标注图输出")
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        square_summary=args.square_summary,
        reference_image=args.reference_image,
        output_dir=args.output_dir,
        model_path=args.model_path,
        save_annotations=not args.no_annotations,
    )
    print(f"常规中心 SVM 检测完毕，结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()