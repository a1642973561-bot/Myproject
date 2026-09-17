import csv
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def extract_center_features(gray_img, ref_gray=None):
    """
    提取中心区域的特征向量：
    [NCC相似度(若有参考图), 均值, 标准差, 拉普拉斯方差(清晰度)]
    """
    h, w = gray_img.shape
    # 裁剪中央特征区域 (中间 40% 区域)
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


def train_regular_svm(image_dir, sample_labels, reference_image_path, model_output_path="svm_regular_model.pkl"):
    """
    训练常规中心检测的 SVM 模型
    sample_labels: 字典，格式如 {"0002.png": 1, "0006.png": 1, "0084.png": 0, ...}
    标签说明: 1 代表符合常规中心（第一类），0 代表不符合
    """
    image_dir = Path(image_dir)
    ref_img = cv2.imread(str(reference_image_path), cv2.IMREAD_GRAYSCALE)

    X = []
    y = []

    for filename, label in sample_labels.items():
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"警告: 训练样本图片不存在: {img_path}")
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"警告: 无法读取图片: {img_path}")
            continue

        feat = extract_center_features(gray, ref_img)
        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # 建立带有标准化预处理的 SVM 管道
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True))
    clf.fit(X, y)

    joblib.dump(clf, model_output_path)
    print(f"常规中心 SVM 模型已成功训练并保存至: {model_output_path}")
    return clf


if __name__ == "__main__":
    # 示例：与之前一致的训练图库及标签配置
    # 挑选若干张典型的通过图片(1)与异常/未通过图片(0)
    train_labels = {
        "0002.png": 1,
        "0004.png": 0,
        "0036.png": 1,
        "0018.png": 0,
        "0046.png": 1,
        "0092.png": 1,
    }

    # 假设图片存放在 PDF 解压后的 output_images 目录中
    img_dir = Path("output_images")
    ref_img_path = img_dir / "0002.png"

    if img_dir.exists():
        train_regular_svm(img_dir, train_labels, ref_img_path)
    else:
        print(f"请先运行流水线提取图片目录: {img_dir}")