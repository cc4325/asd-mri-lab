"""统一命令行入口，可从 VS Code 终端或桌面软件调用。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectConfig
from .data import discover_images, validate_dataset


def main():
    parser = argparse.ArgumentParser(description='T1 MRI 预处理与 ASD/HC 分类')
    parser.add_argument('command', choices=['resources','validate','preprocess','features','train','predict','report','run-all'])
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--input', help='GUI 指定输入目录')
    parser.add_argument('--output', help='GUI 指定预处理输出目录')
    parser.add_argument('--file', help='单例 NIfTI 文件')
    parser.add_argument('--n4-shrink', type=int, choices=[2,4])
    parser.add_argument('--smooth', type=float)
    args = parser.parse_args()
    cfg = ProjectConfig.load(args.config)
    command = args.command
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if command in ('validate','run-all'):
        labels = validate_dataset(cfg.train_dir, cfg.test_dir, cfg.labels_csv)
        print(labels.groupby('label').size().to_dict(), flush=True)
    if command in ('resources','preprocess','run-all'):
        from .resources import prepare_resources
        resources = prepare_resources(cfg.resources_dir)
    else:
        resources = {'atlas': cfg.resources_dir/'schaefer100_2mm.nii.gz', 'labels': cfg.resources_dir/'schaefer100_labels.json'}
    if command in ('preprocess','run-all'):
        from .preprocessing import preprocess_many, build_qc_table
        records = (discover_images(Path(args.input), 'custom') if args.input else
                   discover_images(cfg.train_dir,'train') + discover_images(cfg.test_dir,'test'))
        if args.file:
            from .data import ImageRecord
            image_path=Path(args.file).resolve()
            subject=image_path.name.removesuffix('.gz').removesuffix('.nii').removesuffix('_T1w')
            records=[ImageRecord(subject,image_path,'custom')]
        if not records:
            raise ValueError('未找到 *_T1w.nii.gz 影像文件')
        print(json.dumps({'total':len(records)}),flush=True)
        if args.limit:
            records = records[:args.limit]
        parameters = {k:v for k,v in cfg.preprocessing.items() if k != 'threads'}
        if args.n4_shrink is not None:parameters['n4_shrink_factor']=args.n4_shrink
        if args.smooth is not None:
            if args.smooth < 0:raise ValueError('平滑 FWHM 必须非负')
            parameters['smoothing_fwhm_mm']=args.smooth
        destination = Path(args.output) if args.output else cfg.output_dir/'preprocessed'
        results = preprocess_many(records,destination,resources['template'],resources['mask'],
                                  workers=args.workers, callback=lambda r: print(json.dumps({k:r.get(k) for k in ['subject_id','status','elapsed_seconds','error']},ensure_ascii=False),flush=True), **parameters)
        build_qc_table(destination,cfg.output_dir/'qc_summary.csv')
        if any(r['status'] != 'complete' for r in results):
            raise RuntimeError('存在失败影像，请查看 metadata.json 后重试')
    if command in ('features','run-all'):
        from .features import build_feature_table
        table = build_feature_table(cfg.output_dir/'preprocessed',resources['atlas'],resources['labels'],cfg.output_dir/'features.csv')
        print(f'Features: {table.shape}',flush=True)
    if command in ('train','run-all'):
        from .modeling import train_nested_cv
        print(train_nested_cv(cfg.output_dir/'features.csv',cfg.labels_csv,cfg.output_dir/'model',**cfg.modeling),flush=True)
    if command in ('predict','run-all'):
        from .modeling import predict_test
        print(predict_test(cfg.output_dir/'features.csv',cfg.output_dir/'model/final_model.joblib',cfg.submission_template,cfg.output_dir),flush=True)
    if command in ('report','run-all'):
        from .pdf_report import build_pdf
        print(build_pdf(cfg.root),flush=True)


if __name__ == '__main__':
    main()
