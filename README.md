# T1 MRI 预处理与 ASD/HC 分类桌面软件

这是一个使用 Python 3.13 和 PySide6 编写的科研教学项目。桌面软件支持 NIfTI 三平面浏览、窗宽窗位调整、同空间结果叠加、单例或批量预处理、质量控制复核、模型训练和预测导出。它**不是临床诊断软件**。

本仓库是**源码包**，不包含任何受试者影像、诊断标签、盲测结果、已训练分类器或第三方脑提取权重。若只想查看界面和浏览自有 NIfTI 文件，无须训练数据；若要完整复现分类实验，需自行准备有权使用的数据与外部资源。

## 环境与启动

- Windows 10/11、Python 3.13、Visual Studio Code；完整预处理建议至少 16 GB 内存。
- 在 VS Code 中打开本文件所在目录。`.vscode/` 已配置解释器路径、调试入口和测试任务。
- 在 VS Code 终端执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m asd_mri.gui
```

也可以从 VS Code 的“运行和调试”选择“启动医学图像预处理软件”。`requirements-lock.txt` 记录了本项目 Windows 环境中的主要包版本，但不包含单独安装的 PyTorch CPU wheel，不应当当作完整的离线依赖包。

## 数据与资源准备

完整分类流程使用以下目录结构。`subject_id` 必须与文件名中的 ID 一致；标签定义为 `1 = ASD`、`2 = HC`。

```text
项目根目录/
├── train/                    训练影像：<subject_id>_T1w.nii.gz
├── test/                     待预测影像：<subject_id>_T1w.nii.gz
├── train_labels.csv          subject_id,label
├── submission_example.csv    subject_id（仅列出待预测者）
├── resources/                模板和图谱运行时生成；脑提取权重需另行获取
└── outputs/                  运行后生成，默认不上传
```

模板与 Schaefer 100 区图谱可由 `resources` 命令准备。特征提取还需要 FreeSurfer 官方 SynthStrip v1 权重，放在 `resources/synthstrip.1.pt`；下载地址和 SHA-256 校验值见 [外部资源说明](resources/SOURCES.md)。原始影像和第三方资源应按各自来源和使用条件获取；不要将它们复制进公开仓库。

```powershell
.\.venv\Scripts\python.exe -m asd_mri.cli resources
.\.venv\Scripts\python.exe -m asd_mri.cli validate
.\.venv\Scripts\python.exe -m asd_mri.cli run-all --workers 4
```

完整流程包括方向统一、N4 偏置场校正、稳健强度归一化、SyN 配准、脑组织分割、质量控制、特征提取、嵌套交叉验证和预测。技术细节见 [项目说明](PROJECT_README.md)。界面中的“模型训练”和“导出预测”使用 `config/default.yaml` 中的默认数据位置；手动选择的其他预处理目录不会自动成为训练输入。

**注意：当前预测逻辑针对原任务设计。**它要求恰好 10 例盲测影像，并使用已知的 5 例 ASD、5 例 HC 类别组成，对模型分数排序生成正式预测。迁移到其他数据集时，应先修改该约束，并重新训练、验证模型。没有盲测真值时，不能计算盲测准确率。

## 测试

以下测试使用合成的微型数据，不需要公开原始影像：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 发布边界

`.gitignore` 会排除 `train/`、`test/`、`outputs/`、受试者标签和预测文件、第三方影像资源、权重、虚拟环境及压缩包。上传前仍应检查 Git 暂存列表；忽略规则不会自动移除已经被 Git 跟踪的文件。

本源码包未预设开源许可证。若要允许他人复制、修改或再发布代码，请先确定并添加适合的 `LICENSE` 文件；公开可见不等于自动授予开源使用权。
