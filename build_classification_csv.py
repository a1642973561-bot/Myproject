import argparse
import csv
from pathlib import Path

from utils import get_base_path

CLASS_NAMES = {
    1: "无孔",
    2: "五个孔",
    3: "其他",
}


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def classify_images(regular_center_summary_path, square_circles_summary_path):
    regular_rows = {row["filename"]: row for row in read_rows(regular_center_summary_path)}
    circles_rows = {row["filename"]: row for row in read_rows(square_circles_summary_path)}

    all_filenames = sorted(list(set(regular_rows.keys()).union(set(circles_rows.keys()))))

    if not all_filenames:
        raise ValueError(
            f"未在以下路径找到有效的检测结果摘要文件:\n- {regular_center_summary_path}\n- {square_circles_summary_path}")

    results = []
    for filename in all_filenames:
        regular_data = regular_rows.get(filename, {})
        circle_data = circles_rows.get(filename, {})

        # 获取方圆孔匹配数量
        try:
            square_circles_count = int(
                circle_data.get("matched_slot_count", circle_data.get("square_circle_count", -1)))
        except ValueError:
            square_circles_count = -1

        # 获取常规中心判定结果 (适配不同版本的字段名 is_regular_center 或 has_regular_center)
        is_regular = (
                regular_data.get("is_regular_center") == "yes" or
                regular_data.get("has_regular_center") == "yes"
        )

        # 核心分类逻辑：优先第二类(方圆孔/五个孔) -> 其次第一类(常规) -> 剩下的全归为第三类(其他，绝不出现 Fail)
        if square_circles_count >= 3:
            class_id = 2
        elif is_regular:
            class_id = 1
        else:
            class_id = 3

        results.append({
            "filename": filename,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
        })

    return results


def remove_duplicate_pairs(results):
    if len(results) == 0:
        raise ValueError("分类结果为空，无法进行去重配对校验。")

    if len(results) % 2:
        raise ValueError(f"图片数量为 {len(results)}（不是偶数），无法按相邻图片配对")

    unique_results = []
    for index in range(0, len(results), 2):
        first = results[index]
        second = results[index + 1]
        try:
            first_number = int(Path(first["filename"]).stem)
            second_number = int(Path(second["filename"]).stem)
        except ValueError:
            unique_results.append(first)
            continue

        if second_number != first_number + 1:
            raise ValueError(f"图片无法配对：{first['filename']}、{second['filename']}")

        # 确保统计输出仅评估有意义的偶数图片，单数由于是复制图片则作为配对参照
        even_result = second if second_number % 2 == 0 else first
        odd_result = first if second_number % 2 == 0 else second

        # 如果相邻两张图分类不一致，降级归为第三类（其他）
        if even_result["class_id"] != odd_result["class_id"]:
            print(
                f"警告: 相邻图片分类不一致 {first['filename']}(类{first['class_id']}) vs {second['filename']}(类{second['class_id']})。降级为 3(其他)。")
            even_result["class_id"] = 3
            even_result["class_name"] = CLASS_NAMES[3]

        unique_results.append(even_result)
    return unique_results



def write_classification_csv(regular_summary_path, circles_summary_path, output_csv_path):
    # 1. 先跑基础分类（应用优先级：二类 -> 一类 -> 三类兜底）
    raw_results = classify_images(regular_summary_path, circles_summary_path)

    # 2. 执行相邻去重配对校验
    final_results = remove_duplicate_pairs(raw_results)

    # 3. 写入最终 CSV 供阵列图读取（包含兼容字段：Filename, Category/class_id）
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "class_id", "class_name"])
        writer.writeheader()
        writer.writerows(final_results)

    print(f"分类汇总 CSV 已成功生成并完成配对校验：{output_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    # 使用 get_base_path() 确保路径绝对化
    parser.add_argument("--regular-summary", type=Path, default=get_base_path() / "regular_center_results/summary.csv")
    parser.add_argument("--circles-summary", type=Path, default=get_base_path() / "square_circles_results/summary.csv")
    parser.add_argument("-o", "--output", type=Path, default=get_base_path() / "image_classification.csv")
    args = parser.parse_args()

    write_classification_csv(args.regular_summary, args.circles_summary, args.output)


if __name__ == "__main__":
    main()