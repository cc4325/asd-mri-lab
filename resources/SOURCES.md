# 外部资源来源

- `synthstrip.1.pt`：FreeSurfer 官方 SynthStrip v1 权重，下载地址：<https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt>。
- SHA-256：`37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33`。
- `mni152_t1_2mm.nii.gz` 和 `mni152_brain_mask_2mm.nii.gz`：通过 Nilearn `load_mni152_template` 与 `load_mni152_brain_mask` 获取。
- `schaefer100_2mm.nii.gz` 和 `schaefer100_labels.json`：通过 Nilearn `fetch_atlas_schaefer_2018` 获取并重采样至上述模板空间。

这些外部资源仅用于预处理及特征定义，不提供本任务的诊断标签。
