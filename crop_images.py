import argparse
from pathlib import Path

import fitz


def crop_pdf(
    pdf_path,
    output_dir,
    skip_pages=2,
    start_index=2,
    zoom=2.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    with fitz.open(pdf_path) as doc:
        if doc.page_count <= skip_pages:
            raise ValueError(
                f"PDF 只有 {doc.page_count} 页，无法跳过 {skip_pages} 页"
            )

        pages = range(skip_pages, doc.page_count)
        for output_index, page_number in enumerate(
            pages, start=start_index
        ):
            page = doc[page_number]
            image_rects = []

            for image in page.get_images(full=True):
                xref = image[0]
                for rect in page.get_image_rects(xref):
                    rect = rect & page.rect
                    if not rect.is_empty and not rect.is_infinite:
                        image_rects.append(rect)

            if not image_rects:
                print(
                    f"第 {page_number + 1} 页未检测到图片，使用完整页面"
                )
                crop_rect = page.rect
            else:
                crop_rect = max(
                    image_rects,
                    key=lambda rect: rect.width * rect.height,
                )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                clip=crop_rect,
                alpha=False,
            )
            output_path = output_dir / f"{output_index:04d}.png"
            pix.save(output_path)
            output_paths.append(output_path)
            print(f"已保存：{output_path}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="从 PDF 每页提取面积最大的主图"
    )
    parser.add_argument("pdf", type=Path, help="输入 PDF")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="PNG 输出目录",
    )
    parser.add_argument(
        "--skip-pages",
        type=int,
        default=2,
        help="跳过开头页数，当前报告默认跳过前两页",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=2,
        help="首张 PNG 编号，默认 2",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="PDF 渲染倍率，默认 2.0",
    )
    args = parser.parse_args()

    paths = crop_pdf(
        args.pdf,
        args.output_dir,
        args.skip_pages,
        args.start_index,
        args.zoom,
    )
    print(f"共提取 {len(paths)} 张图片")


if __name__ == "__main__":
    main()