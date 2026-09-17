import os
import gc
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import sys
from pathlib import Path
from utils import get_base_path
# 直接在同一个脚本中导入后端所有核心函数，避免任何打包后找不到模块的问题
from crop_images import crop_pdf
from rectify_images import process_directory as detect_square_circles
from detect_regular_center import process_directory as detect_regular
from build_classification_csv import write_classification_csv
from generate_classification_grid import write_grid


def run_pipeline(pdf_path: Path, output_dir: Path = None, save_annotations: bool = True, skip_pages: int = 2):
    # ... 前面代码保持不变 ...
    """将原本 run_pipeline.py 的核心逻辑直接内联到前端，确保万无一失"""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到输入的 PDF 文件：{pdf_path}")

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    rectified_images_dir = output_dir / "output_images"
    circles_results_dir = output_dir / "square_circles_results"
    regular_results_dir = output_dir / "regular_center_results"
    classification_csv = output_dir / "image_classification.csv"
    grid_path = output_dir / "image_classification_grid.png"

    reference_image_path = rectified_images_dir / "0002.png"

    print(f"开始处理 PDF 文件：{pdf_path.name}")
    print(f"输出目标路径：{output_dir}\n")



    print("=== 步骤 1/4: 从 PDF 提取原始图片 ===")
    crop_pdf(
        pdf_path,
        rectified_images_dir,
        skip_pages=skip_pages,  # <- 接收传进来的变量
        start_index=skip_pages,  # 起始序号通常和跳过页数保持同步
        zoom=2.0,
    )

    print("\n=== 步骤 2/4: 执行特征检测算法 ===")

    print("1. 运行方圆孔检测算法...")
    detect_square_circles(
        input_dir=rectified_images_dir,
        output_dir=circles_results_dir,
        save_annotations=save_annotations,
    )
    circles_summary_csv = circles_results_dir / "summary.csv"

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
        model_path=get_base_path() / "svm_regular_model.pkl",  # <- 修改这里
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


class RedirectText:
    """将后台标准输出实时重定向到界面的文本框中"""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str_text):
        self.text_widget.after(0, self._append, str_text)

    def _append(self, str_text):
        self.text_widget.insert(tk.END, str_text)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class BatchPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("code_z")
        self.root.geometry("850x650")

        # 1. 文件夹选择栏
        frame_dir = tk.Frame(root, padx=10, pady=10)
        frame_dir.pack(fill=tk.X)

        tk.Label(frame_dir, text="PDF 根目录:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        tk.Entry(frame_dir, textvariable=self.path_var, width=50, font=("Arial", 10)).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        tk.Button(frame_dir, text="选择文件夹...", command=self.browse_directory, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # 2. 控制面板
        frame_ctrl = tk.Frame(root, padx=10, pady=5)
        frame_ctrl.pack(fill=tk.X)

        tk.Label(frame_ctrl, text="跳过页数:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.skip_var = tk.StringVar(value="2")
        tk.Entry(frame_ctrl, textvariable=self.skip_var, width=8, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.btn_run = tk.Button(frame_ctrl, text="开始处理", command=self.start_thread,
                                 bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), padx=15, pady=4)
        self.btn_run.pack(side=tk.LEFT, padx=20)

        # 3. 实时日志滚动输出框
        frame_log = tk.Frame(root, padx=10, pady=10)
        frame_log.pack(expand=True, fill=tk.BOTH)

        tk.Label(frame_log, text="实时运行日志:", font=("Arial", 10)).pack(anchor=tk.W, padx=5)
        self.log_box = scrolledtext.ScrolledText(frame_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_box.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # 重定向输出
        sys.stdout = RedirectText(self.log_box)
        sys.stderr = RedirectText(self.log_box)

    def browse_directory(self):
        d = filedialog.askdirectory()
        if d:
            self.path_var.set(d)

    def start_thread(self):
        target_dir = self.path_var.get().strip()
        if not target_dir or not Path(target_dir).exists():
            messagebox.showerror("错误", "请先选择一个有效的 PDF 根目录路径！")
            return

        self.btn_run.config(state=tk.DISABLED, bg="grey")
        threading.Thread(target=self.batch_task_worker, args=(target_dir,), daemon=True).start()

    def batch_task_worker(self, target_dir):
        # 获取界面的跳过页数，如果输入的不是数字，兜底为 2
        try:
            skip_count = int(self.skip_var.get().strip())
        except ValueError:
            skip_count = 2
        try:
            root_path = Path(target_dir)
            out_root = root_path / "batch_output_results"
            out_root.mkdir(parents=True, exist_ok=True)

            print(f"🔍 正在递归扫描目录: {root_path}")
            pdf_files = list(root_path.rglob("*.pdf"))
            total = len(pdf_files)

            if total == 0:
                print("❌ 未在指定目录下发现任何 PDF 文件。")
                messagebox.showwarning("提示", "未找到任何 PDF 文件。")
                return

            print(f"📦 扫描完成！共发现 {total} 份 PDF 文件，开始逐份安全处理...\n" + "="*50 + "\n")

            success, fail = 0, 0

            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"[{i}/{total}] 正在处理: {pdf_file.name}")
                file_out_dir = out_root / pdf_file.stem

                try:
                    run_pipeline(
                        pdf_path=pdf_file,
                        output_dir=file_out_dir,
                        save_annotations=True,
                        skip_pages=skip_count  # <- 把参数喂进去
                    )
                    success += 1
                    print(f"✅ [{i}/{total}] 成功完成: {pdf_file.name}\n")
                except Exception as e:
                    fail += 1
                    print(f"❌ [{i}/{total}] 处理失败 {pdf_file.name}: {str(e)}\n")

                gc.collect()

            print("="*50)
            print(f"🎉 批量任务全部执行完毕！")
            print(f"📊 总数: {total} | 成功: {success} | 失败: {fail}")
            print(f"📁 结果已输出至: {out_root.resolve()}")
            messagebox.showinfo("完成", f"批量处理完成！\n成功: {success} 份\n失败: {fail} 份")

        except Exception as e:
            print(f"批量任务发生致命异常: {str(e)}")
            messagebox.showerror("错误", f"发生异常:\n{str(e)}")
        finally:
            self.btn_run.config(state=tk.NORMAL, bg="#4CAF50")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatchPipelineGUI(root)
    root.mainloop()