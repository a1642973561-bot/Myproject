## 项目简介

本项目把**固定版式的电路板 X 光检测报告 PDF** 转换成按图片编号排列的分类结果：

1. 逐页提取报告中的主图（默认跳过前 2 页封面/说明页）；
2. 用电路板上的黑色定位圆做几何归一化，在固定的 5 个孔位区域上判断是否存在方圆形结构；
3. 对未被上一步命中的图片，截取中心区域提取特征，判断是否属于"常规中心"结构；
4. 合并两类检测结果裁决类别，并按"相邻两页是同一张图"的规律去重；
5. 输出逐图分类 CSV 以及一张 14 列 × 24 行的蛇形分类总览图。

检测依赖当前报告固定的定位圆边界、5 个孔位坐标、参考图编号与模板图片编号，因此流程适用于
**与现有报告同版式**的 PDF，而不是任意电路板图像。


## 功能

以下功能均能在源码中找到对应实现：

- **PDF 图像提取**：跳过指定页数，取每页面积最大的图片，按 2 倍缩放渲染为 PNG，编号从 `0002` 开始（`crop_images.py`）。
- **定位圆几何归一化**：灰度阈值 + 轮廓圆度筛选定位圆，取最小/最大 x、y 作为板面边界，再按参考图坐标比例推算 5 个固定检测框，降低平移、缩放、倾斜、曝光差异的影响（`rectify_images.py`）。
- **方圆形（5 孔位）检测**：对 5 个框各提取 5 维特征（NCC 模板相似度、灰度均值、标准差、梯度幅值、位置编号），用 `svm_model.pkl` 逐框判定，命中数 ≥ 3 即整体判为"方圆形"；模型缺失时回退到固定阈值 0.10（`rectify_images.py`）。
- **常规中心检测**：只处理未被方圆形流程命中的图片，截取图像中心 40% 区域提取 4 维特征（与 `0002.png` 的 NCC 相似度、均值、标准差、拉普拉斯方差），用 `svm_regular_model.pkl` 判定（`detect_regular_center.py`）。
- **分类裁决与配对去重**：按"方圆形优先 → 常规中心 → 其他"裁决为 3 类；按相邻编号 (n, n+1) 配对，只保留偶数编号那张；两张分类不一致时降级为"其他"（`build_classification_csv.py`）。
- **结果总览图**：渲染 14 列 × 24 行、自左下角开始蛇形排列的阵列图，每格写入图片编号与类别编号（`generate_classification_grid.py`）。
- **批量 GUI**：Tkinter 界面，选择根目录后递归处理该目录下所有 PDF，逐个输出到 `batch_output_results/<PDF 名>/`，运行日志实时回显（`gui.py`）。
- **命令行入口**：`run_pipeline.py`（端到端）、`crop_images.py`、`detect_regular_center.py`、`build_classification_csv.py`、`generate_classification_grid.py`、`train_svm.py`、`train_regular_svm.py`。
- **打包配置**：`VisionPipeline.spec` 以 `gui.py` 为入口，把各模块与两个 `.pkl` 模型一起打包为 `VisionSystemApp`（`console=False`，不显示控制台窗口）。
- **标注图输出**：方圆形结果输出 `<编号>_boxed.png`（画出 5 个框与命中状态），常规中心结果对判定通过的图片输出 `<编号>_regular.png`。

## 技术栈

| 类别 | 实际使用情况 |
| --- | --- |
| 语言 | Python（仓库自带 `.venv` 为 CPython 3.12.10；`__pycache__` 中还有 3.11 / 3.13 的残留字节码） |
| 图像处理 | OpenCV `opencv_python-5.0.0.93`（`cv2`：阈值、轮廓、SVM、HOG、CLAHE、模板匹配、绘制） |
| 数值计算 | NumPy `2.5.1`、SciPy `1.18.0` |
| 机器学习 | scikit-learn `1.9.0`（`StandardScaler` + RBF 核 `SVC` 组成的 pipeline）、joblib `1.5.3` 保存/加载模型 |
| PDF 解析 | PyMuPDF `1.28.0`（代码中以 `import fitz` 使用） |
| 桌面 GUI | Tkinter（Python 标准库，实测 Tk 8.6） |
| 打包 | PyInstaller（配置文件 `VisionPipeline.spec`；PyInstaller 本身未安装在当前 `.venv` 中） |
| 开发环境 | PyCharm（`.idea/`，工程 SDK 名为 `Python 3.13 (code_z)`） |

## 项目结构

```text
├── run_pipeline.py                  # 命令行端到端入口（4 个步骤串联）
├── gui.py                           # Tkinter 批量处理界面（内含一份流水线副本）
├── crop_images.py                   # 步骤 1：PDF → PNG
├── rectify_images.py                # 步骤 2a：5 个孔位的方圆形检测（SVM + 阈值回退）
├── detect_regular_center.py         # 步骤 2b：常规中心检测（SVM）
├── build_classification_csv.py      # 步骤 3：分类裁决 + 相邻配对去重
├── generate_classification_grid.py  # 步骤 4：14 × 24 蛇形阵列图
├── utils.py                         # 运行路径 / 资源路径（兼容 PyInstaller frozen 模式）
├── svm_detector.py                  # 与 rectify_images 重复的独立检测实现（未接入流水线）
├── detect_square_circles.py         # 旧版多类别检测器（当前无法导入，见"已知问题"）
├── train_svm.py                     # 训练 svm_model.pkl（方圆形槽位分类器）
├── train_regular_svm.py             # 训练 svm_regular_model.pkl（常规中心分类器）
├── svm_model.pkl                    # 已训练模型，输入 5 维特征
├── svm_regular_model.pkl            # 已训练模型，输入 4 维特征
├── CLASSIFICATION_ALGORITHM.md      # 分类算法说明（与当前代码存在偏差，见"已知问题"）
├── README.md                        # 本文件
├── __init__.py                      # 包说明文档字符串："PDF 图像分类流水线。"
├── output_images 

## 核心模块

| 模块 | 关键函数 / 类 | 职责 |
| --- | --- | --- |
| `run_pipeline.py` | `run_pipeline()`、`main()` | 串联 4 个步骤，创建 `output_images/`、`square_circles_results/`、`regular_center_results/`，产出 `image_classification.csv` 与 `image_classification_grid.png` |
| `crop_images.py` | `crop_pdf()` | 用 PyMuPDF 打开 PDF，跳过前 `skip_pages` 页，取每页面积最大的图像矩形渲染并保存为 PNG |
| `rectify_images.py` | `find_hole_bounds()`、`infer_boxes_from_relative_position()`、`extract_box_features()`、`evaluate_slot_relative()`、`detect_square_circles()`、`process_directory()` | 定位圆→5 个框→SVM 判定→写 `summary.csv`，可选输出标注图 |
| `detect_regular_center.py` | `load_remaining_images()`、`extract_center_features()`、`process_directory()` | 读取方圆形结果、筛出未命中的图片、中心区域特征 + SVM 预测、写 `summary.csv` |
| `build_classification_csv.py` | `classify_images()`、`remove_duplicate_pairs()`、`write_classification_csv()` | 合并两份 `summary.csv` 裁决类别，再做相邻配对去重后写出最终 CSV |
| `generate_classification_grid.py` | `load_classifications()`、`cell_position()`、`generate_grid()`、`write_grid()` | 读取分类 CSV，按蛇形顺序渲染阵列图（只画编号与类别号，不嵌入原始图片） |
| `gui.py` | `BatchPipelineGUI`、`RedirectText`、`run_pipeline()` | Tkinter 前端：选择根目录、设置跳过页数、后台线程批量处理所有 PDF、把 stdout/stderr 重定向到日志框 |
| `utils.py` | `get_base_path()`、`get_resource_path()` | 兼容源码运行与 PyInstaller 打包两种情形的路径解析 |
| `train_svm.py` | `train_and_save_model()` | 用人工标注的槽位标签训练 `svm_model.pkl`（`StandardScaler` + RBF `SVC`，`C=10`） |
| `train_regular_svm.py` | `extract_center_features()`、`train_regular_svm()` | 训练 `svm_regular_model.pkl`（`StandardScaler` + RBF `SVC`，`C=1.0`） |
| `svm_detector.py` | `train_svm()`、`process_image_svm()` | 一份独立的 SVM 检测实现，与 `rectify_images.py` 逻辑重复，未接入流水线入口 |
| `detect_square_circles.py` | `train_detector()`、`detect_patterns()`、`detect_center_ring_boxes()`、`process_directory()` | 旧版多类别检测器（含 HOG + OpenCV SVM 滑窗、中心环框模板比对），依赖已不存在的接口，当前无法导入 |

### 数据流与模块关系

```text
PDF 报告
  │  crop_images.crop_pdf()                     跳过前 2 页，取每页最大图，2 倍渲染
  ▼
output_images/0002.png, 0003.png, …
  │  rectify_images.detect_square_circles()     定位圆 → 5 个固定孔位 → svm_model.pkl
  ▼
square_circles_results/summary.csv              每图：matched_slot_count / has_square_circles
  │  detect_regular_center.process_directory()  跳过已命中图，中心特征 → svm_regular_model.pkl
  ▼
regular_center_results/summary.csv              每图：similarity / is_regular_center
  │  build_classification_csv.write_classification_csv()   裁决 + 相邻配对去重
  ▼
image_classification.csv
  │  generate_classification_grid.write_grid()
  ▼
image_classification_grid.png
```

两个检测器是**前后依赖**关系：`detect_regular_center` 通过读取方圆形检测的 `summary.csv`，
把 `has_square_circles == "yes"` 的图片直接标记为 `is_regular_center = no`，不重复处理。
最终裁决时方圆形数量 ≥ 3 的图片优先归为第 2 类（`class_name = "五个孔"`），其余按常规中心
判定归为第 1 类（`"无孔"`），都不满足则归为第 3 类（`"其他"`）。

## 运行环境

- **操作系统**：Windows（代码使用 Windows 路径风格、Tkinter 桌面界面，PyInstaller 配置面向 Windows；未做跨平台验证）。
- **Python**：3.12.10（仓库自带 `.venv`；`.idea/misc.xml` 中记录的 SDK 名为 Python 3.13）。
- **依赖库**：NumPy、SciPy、OpenCV、scikit-learn、joblib、PyMuPDF；GUI 需要 Tkinter。
- **硬件**：仅需 CPU，无 GPU 依赖。
- **仓库内没有 `requirements.txt`**，依赖需要按下面的说明手动安装。
- 需要输入的 PDF 满足：前 2 页为封面/说明页；之后每页一张主图；相邻两页为同一张图；图像尺寸与定位圆布局和现有样例一致（样例图为 900 × 900）。

## 安装与运行

### 1. 准备环境

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy scipy opencv-python scikit-learn joblib PyMuPDF
```

打包为可执行文件还需要额外安装 PyInstaller：`python -m pip install pyinstaller`。

### 2. 命令行运行完整流水线

在仓库根目录执行（输出目录缺省时，会在 PDF 同级的同名文件夹下生成结果）：

```powershell
python run_pipeline.py <报告.pdf>
python run_pipeline.py <报告.pdf> -o <输出目录>
python run_pipeline.py <报告.pdf> --no-annotations
```

注意：`run_pipeline.py` 的默认行为是 **保存标注图**（`save_annotations=True`），
`--no-annotations` 用于关闭标注图输出以减少磁盘占用。

### 3. 图形界面批量运行

```powershell
python gui.py
```

### 4. 单独运行某一步

```powershell
python crop_images.py <报告.pdf> --output-dir <输出目录> [--skip-pages 2] [--start-index 2] [--zoom 2.0]

python detect_regular_center.py --input_dir output_images --square_summary square_circles_results/summary.csv --reference_image output_images/0002.png --output_dir regular_center_results [--no-annotations]

python build_classification_csv.py --regular-summary <...>/summary.csv --circles-summary <...>/summary.csv -o image_classification.csv
python generate_classification_grid.py --classification-csv image_classification.csv -o image_classification_grid.png
```

`rectify_images.py` 和 `utils.py` 没有命令行入口，只能作为模块被导入。


### 6. 重新训练模型（可选）

```powershell
python train_svm.py            # 使用 output_images/ 与脚本内硬编码的样本标签，生成 svm_model.pkl
python train_regular_svm.py    # 使用 output_images/ 与脚本内硬编码的样本标签，生成 svm_regular_model.pkl
```

两个训练脚本的样本标签直接写在 `__main__` 中，需要按实际数据修改后才能复用。

## 使用说明

### 批量处理（推荐）

1. 运行 `python gui.py`，界面标题为 `code_z`。
2. 点击"选择文件夹…"选择包含 PDF 的根目录（程序会递归扫描该目录下的所有 `*.pdf`）。
3. 按需修改"跳过页数"（默认 2，非数字输入会回退为 2）。
4. 点击"开始处理"，日志框会实时显示每一步进度，处理期间按钮被禁用。
5. 完成后弹窗给出成功/失败数量，结果位于 `<所选根目录>/batch_output_results/<PDF 文件名>/` 下。

### 单份处理

使用 `python run_pipeline.py <报告.pdf>`，结果目录结构如下：

```text
<输出目录>/
├── output_images/                    # 从 PDF 提取的原始主图（0002.png 起编号）
├── square_circles_results/
│   ├── summary.csv                   # 方圆形检测结果
│   └── <编号>_boxed.png              # 标注图（默认输出，可用 --no-annotations 关闭）
├── regular_center_results/
│   ├── summary.csv                   # 常规中心检测结果
│   └── <编号>_regular.png            # 仅对判定为常规中心的图片输出
├── image_classification.csv          # 最终分类结果
└── image_classification_grid.png     # 蛇形阵列图
```

### 输出字段说明

- `square_circles_results/summary.csv`：`filename`、`matched_slot_count`（5 个框中的命中数）、
  `median_contrast`、`has_square_circles`（`yes`/`no`）、`box_coordinates`（5 个框的坐标）。
- `regular_center_results/summary.csv`：`filename`、`similarity`（SVM 正类概率；已命中方形圆的图片记为 `N/A`）、
  `is_regular_center`（`yes`/`no`）。
- `image_classification.csv`：`filename`、`class_id`、`class_name`，其中
  `1 = 无孔`、`2 = 五个孔`、`3 = 其他`（定义见 `build_classification_csv.py` 的 `CLASS_NAMES`）。
- `image_classification_grid.png`：14 列 × 24 行、从底部左侧开始的蛇形排列，每格显示图片编号与类别号。

