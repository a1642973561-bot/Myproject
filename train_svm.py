from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from rectify_images import infer_boxes_from_relative_position, extract_box_features


def train_and_save_model(image_dir, labels_dict, model_path="svm_model.pkl"):
    """
    image_dir: 存放训练图片的文件夹路径
    labels_dict: 样本标签字典。格式为:
                 {
                     "0004.png": [1, 1, 1, 1, 1],  # 1表示该框通过，0表示失败
                     "0002.png": [1, 0, 1, 1, 1]   # 比如0002的2号框异常
                 }
    """
    X, y = [], []
    image_dir = Path(image_dir)

    for img_path in image_dir.iterdir():
        if img_path.name not in labels_dict:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            boxes = infer_boxes_from_relative_position(gray)
            slot_labels = labels_dict[img_path.name]

            for idx, box in enumerate(boxes, start=1):
                # 提取 5 维特征: [NCC得分, 均值, 标准差, 梯度幅值, 位置编号]
                features = extract_box_features(gray, box, idx)
                X.append(features)
                y.append(slot_labels[idx - 1])
        except Exception as e:
            print(f"处理图片 {img_path.name} 失败: {e}")

    if not X:
        raise ValueError("没有收集到任何有效的训练特征，请检查图片路径和标签字典！")

    X = np.array(X)
    y = np.array(y)

    # 建立 Pipeline：自动特征标准化 + RBF 核 SVM 分类器
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10.0, probability=True))
    clf.fit(X, y)

    joblib.dump(clf, model_path)
    print(f"SVM 模型已成功训练并保存至: {model_path}")
    return clf


if __name__ == "__main__":
    # 在这里填写您的训练图片文件夹以及对应的通过/异常标签 (1代表通过，0代表异常)
    sample_labels = {
        "0004.png": [0, 0, 0, 0, 0],
        "0002.png": [0, 0, 0, 0, 0],
        "0072.png": [0, 0, 0, 0, 0],
        "0668.png": [1, 1, 1, 1, 1],
        "0634.png": [1, 1, 1, 1, 1],
        "0534.png": [1, 1, 1, 1, 1],
        "0018.png": [0, 0, 0, 0, 0],
    }
    # 假设您的训练图存放在当前目录的 output_images 或指定文件夹
    train_and_save_model("output_images", sample_labels)