# 基于 T1 MRI 的 ASD HC 分类与预处理软件

本项目使用 136 例训练影像建立 ASD 与健康对照分类模型，并对 10 例盲测影像生成预测。工程同时提供一个 PySide6 桌面软件，用于 NIfTI 三平面浏览、结果叠加、批量预处理、模型训练与预测导出。

## 环境

- Windows 10 或 11
- Python 3.13
- Visual Studio Code
- 推荐至少 16 GB 内存；完整预处理需要数小时

VS Code 已配置使用 `.venv/Scripts/python.exe`。在项目根目录打开 VS Code 后，可以从“运行和调试”启动桌面软件，或在集成终端运行以下命令。

```powershell
.\.venv\Scripts\python.exe -m asd_mri.cli validate
.\.venv\Scripts\python.exe -m asd_mri.cli preprocess --workers 4
.\.venv\Scripts\python.exe -m asd_mri.cli features
.\.venv\Scripts\python.exe -m asd_mri.cli train
.\.venv\Scripts\python.exe -m asd_mri.cli predict
.\.venv\Scripts\python.exe -m asd_mri.cli report
```

也可以一次执行完整流程：

```powershell
.\.venv\Scripts\python.exe -m asd_mri.cli run-all --workers 4
```

## 处理流程

每例影像依次经过 RAS 重定向、N4 偏置场校正、稳健强度归一化、可复现 SyN 配准、模板脑掩膜反变换、原始空间 Atropos 三组织分割、概率图映射至 MNI152 和 4 mm FWHM 平滑。软件保存全部中间影像、变换文件、质量控制图、输入文件哈希以及处理参数。

特征包括 100 区 Schaefer 图谱内的灰质概率均值、独立 SynthStrip 脑掩膜限定的原始空间 GM/WM/CSF 体积及组织比例、网络左右不对称性和局部体积变形，共 214 维。模板相关性仅用于质量控制，不进入模型；低于 0.45 的影像须复核后才能建模。模型选择采用重复分层外层交叉验证和内层网格搜索；缺失值填补、标准化和单变量特征选择全部包含在每个训练折的 `Pipeline` 内。

## 输出目录

若某例的模板相关性低于 0.45，可运行 `python scripts/repair_low_qc.py`。该脚本只对异常样本使用官方 SynthStrip 脑提取权重重新配准，随后仍需检查新的质控结果。特征提取时对全部样本使用该权重生成独立脑掩膜，以防脑外组织计入体积。权重来源和校验值见 `resources/SOURCES.md`。

```text
outputs/
├── preprocessed/                 每例的 NIfTI、变换、metadata.json 和 qc.png
├── qc_summary.csv                预处理状态与质量指标
├── features.csv                  建模特征
├── model/
│   ├── final_model.joblib        完整训练集拟合的最终模型
│   ├── model_summary.json        交叉验证指标与置信区间
│   ├── model_evaluation.png      ROC 与混淆矩阵
│   └── nested_cv_*.csv           各折预测与参数
├── submission.csv                依据已知 5/5 类别数排序的正式提交
├── submission_default_threshold.csv  分类器默认阈值的对照结果
├── test_probabilities.csv        测试集 ASD 概率
└── report.pdf                    方法、结果、限制及软件说明
```

项目根目录的 `submission.csv` 与 `outputs/submission.csv` 内容相同，可直接作为作业提交文件。正式标签按作业公开的测试集 5 例 ASD / 5 例 HC 组成，对模型分数排序得到；默认阈值预测另存为对照文件。

## 桌面软件

```powershell
.\.venv\Scripts\python.exe -m asd_mri.gui
```

软件支持 NIfTI 三平面同步浏览、窗宽窗位、同空间标签叠加、批量预处理、低质控脑提取修复、取消任务以及训练和预测。耗时任务通过独立进程运行，界面保持响应。该软件用于教学和科研验证，不作为临床诊断工具。

## 可复现性与限制

随机种子固定为 `20260924`，ANTs 使用单线程可复现配准模式。训练数据没有年龄、性别和原始扫描协变量，因此无法完整控制这些混杂因素。测试集仅 10 例，单个样本会使准确率改变 10 个百分点，结果应结合交叉验证区间解释。

