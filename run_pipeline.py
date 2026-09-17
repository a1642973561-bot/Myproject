import argparse
from pathlib import Path

from crop_images import crop_pdf
from rectify_images import process_directory as detect_square_circles
from detect_regular_center import process_directory as detect_regular
from build_classification_csv import write_classification_csv
from generate_classification_grid import write_grid


def run_pipeline(pdf_path: Path, output_dir: Path = None, save_annotations: bool = True):
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到输入的 PDF 文件：{pdf_path}")

    # 如果没有指定输出目录，默认在 PDF 文件所在的同级目录下创建一个以 PDF 名命名的文件夹
    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 所有子结果和阵列图都统一基于这个 output_dir 进行拼接
    rectified_images_dir = output_dir / "output_images"
    circles_results_dir = output_dir / "square_circles_results"
    regular_results_dir = output_dir / "regular_center_results"
    classification_csv = output_dir / "image_classification.csv"

    # 保证蛇形图和它们在同一个目录下
    grid_path = output_dir / "image_classification_grid.png"


    reference_image_path = rectified_images_dir / "0002.png"

    print(f"开始处理 PDF 文件：{pdf_path.name}")
    print(f"输出目标路径：{output_dir}\n")

    print("=== 步骤 1/4: 从 PDF 提取原始图片 ===")
    crop_pdf(
        pdf_path,
        rectified_images_dir,
        skip_pages=2,
        start_index=2,
        zoom=2.0,
    )

    print("\n=== 步骤 2/4: 执行特征检测算法 ===")

    # 1. 方圆孔检测 (原第二类)
    print("1. 运行方圆孔检测算法 (rectify_images.py)...")
    detect_square_circles(
        input_dir=rectified_images_dir,
        output_dir=circles_results_dir,
        save_annotations=save_annotations,
    )
    circles_summary_csv = circles_results_dir / "summary.csv"

    # 2. 模板对比法 (原第一类)
    print("2. 运行模版对比法 (判断常规中心)...")
    if not reference_image_path.exists():
        imgs = sorted(list(rectified_images_dir.glob("*.png")))
        if imgs:
            reference_image_path = imgs[0]

    detect_regular(
        input_dir=rectified_images_dir,
        square_summary=circles_summary_csv,
        reference_image=reference_image_path,
        output_dir=regular_results_dir,
        model_path=Path("svm_regular_model.pkl"),  # 传入训练好的 SVM 模型路径
        save_annotations=save_annotations,
    )
    regular_summary_csv = regular_results_dir / "summary.csv"

    print("\n=== 步骤 3/4: 裁决并生成分类 CSV ===")
    write_classification_csv(
        regular_summary_path=regular_summary_csv,
        circles_summary_path=circles_summary_csv,
        output_csv_path=classification_csv,
    )
    print(f"分类汇总文件已写入：{classification_csv}")

    print("\n=== 步骤 4/4: 生成蛇形分类阵列图 ===")
    write_grid(classification_csv, grid_path)
    print(f"最终阵列图已保存：{grid_path}")

    return grid_path


def main():
    parser = argparse.ArgumentParser(description="运行电路板 X 光图片分类自动化流水线")
    parser.add_argument("pdf", type=Path, help="输入的 PDF 报告路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="指定输出目录")
    parser.add_argument("--no-annotations", action="store_true", help="关闭过程标注图的保存")
    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        save_annotations=not args.no_annotations,
    )


if __name__ == "__main__":
    main()