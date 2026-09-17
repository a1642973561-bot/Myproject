import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# 阵列规格
COLUMNS = 14
ROWS = 24

CELL_SIZE = 72
GAP = 4
MARGIN = 8


def parse_class_id(row):
    if "class_id" in row and row["class_id"]:
        return str(row["class_id"])

    cat = row.get("Category", row.get("category", "")).strip()

    if "Class I" in cat or "常规" in cat:
        return "1"
    elif "Class II" in cat or "方圆" in cat:
        return "2"
    elif "Class III" in cat:
        return "3"
    else:
        return "Fail"


def load_classifications(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV为空：{csv_path}")

    items = []

    for row in rows:
        filename = row.get("filename") or row.get("Filename") or ""
        stem = Path(filename).stem

        try:
            img_num = int(stem)
        except ValueError:
            img_num = 0

        class_id = parse_class_id(row)

        items.append(
            (img_num, stem, class_id)
        )

    # 按编号排序
    items.sort(key=lambda x: x[0])

    return items


def cell_position(index):
    """
    蛇形排列：

    底部:
    0002 -> 左
    0028 -> 右

    倒数第二行:
    0056 -> 右
    0030 -> 左

    """

    row_from_bottom = index // COLUMNS
    col_index = index % COLUMNS


    # 偶数层：左到右
    if row_from_bottom % 2 == 0:
        column = col_index

    # 奇数层：右到左
    else:
        column = COLUMNS - 1 - col_index


    # 图片坐标顶部为0，所以反转
    row = ROWS - 1 - row_from_bottom

    return row, column


def centered_text(image, text, center, scale, thickness):

    size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    x = round(center[0] - size[0] / 2)
    y = round(center[1] + size[1] / 2)


    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (30, 30, 30),
        thickness,
        cv2.LINE_AA
    )


def generate_grid(items):

    if len(items) != COLUMNS * ROWS:
        raise ValueError(
            f"数量错误，需要 {COLUMNS*ROWS} 张，当前 {len(items)} 张"
        )


    width = (
        MARGIN * 2
        + COLUMNS * CELL_SIZE
        + (COLUMNS - 1) * GAP
    )

    height = (
        MARGIN * 2
        + ROWS * CELL_SIZE
        + (ROWS - 1) * GAP
    )


    image = np.full(
        (height, width, 3),
        245,
        dtype=np.uint8
    )


    for index, (_, stem, class_id) in enumerate(items):

        row, column = cell_position(index)


        x1 = MARGIN + column * (CELL_SIZE + GAP)
        y1 = MARGIN + row * (CELL_SIZE + GAP)

        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE


        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (185,185,185),
            -1
        )

        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (100,100,100),
            1
        )


        # 图片编号
        centered_text(
            image,
            stem,
            (
                x1 + CELL_SIZE//2,
                y1 + 15
            ),
            0.34,
            1
        )


        # 分类
        centered_text(
            image,
            str(class_id),
            (
                x1 + CELL_SIZE//2,
                y1 + 43
            ),
            1.05,
            2
        )


    return image



def write_grid(classification_csv, output):

    items = load_classifications(classification_csv)

    image = generate_grid(items)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(str(output), image):
        raise OSError(
            f"无法保存图片:{output}"
        )

    return output



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("image_classification.csv")
    )


    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("image_classification_grid.png")
    )


    args = parser.parse_args()


    write_grid(
        args.classification_csv,
        args.output
    )


    print(
        f"分类阵列图已生成:{args.output}"
    )


if __name__ == "__main__":
    main()