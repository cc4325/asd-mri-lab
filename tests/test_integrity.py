import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import nibabel as nib
from asd_mri.data import subject_id_from_path, validate_dataset
from asd_mri.modeling import _metric_dict, _pipeline
from asd_mri.features import extract_subject_features


class IntegrityTests(unittest.TestCase):
    def test_subject_id_and_labels(self):
        self.assertEqual(subject_id_from_path(Path('sub-001_T1w.nii.gz')), 'sub-001')
        with self.assertRaises(ValueError):subject_id_from_path(Path('wrong.nii.gz'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train').mkdir();(root/'test').mkdir()
            (root/'train/sub-001_T1w.nii.gz').touch()
            pd.DataFrame({'subject_id':['sub-002'],'label':[1]}).to_csv(root/'labels.csv',index=False)
            with self.assertRaises(ValueError):validate_dataset(root/'train',root/'test',root/'labels.csv')

    def test_positive_class_is_asd(self):
        metrics=_metric_dict(np.array([0,0,1,1]),np.array([0,0,0,1]),np.array([.1,.2,.3,.8]))
        self.assertEqual(metrics['sensitivity_asd'],.5)
        self.assertEqual(metrics['specificity_hc'],1)

    def test_selection_inside_pipeline(self):
        pipeline,grid=_pipeline(42)
        self.assertIn('select',pipeline.named_steps)
        self.assertIn('scaler',pipeline.named_steps)

    def test_native_volume_and_background_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subject=root/'sub-001';subject.mkdir()
            aff=np.diag([100,100,100,1]);data=np.ones((3,3,3),dtype=np.float32)*.5
            for name in ['gm_probability_smooth','wm_probability','csf_probability']:
                nib.save(nib.Nifti1Image(data,aff),subject/f'{name}.nii.gz')
            mask=np.zeros((3,3,3),dtype=np.uint8);mask[1,1,1]=1
            nib.save(nib.Nifti1Image(mask,aff),subject/'synthstrip_mask_native.nii.gz')
            for name,value in [('gm',.5),('wm',.3),('csf',.2)]:
                nib.save(nib.Nifti1Image(np.full((3,3,3),value,dtype=np.float32),aff),subject/f'{name}_native.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((3,3,3),dtype=np.int16),aff),root/'atlas.nii.gz')
            (root/'labels.json').write_text(json.dumps(['Background','ROI']),encoding='utf-8')
            (subject/'metadata.json').write_text(json.dumps({'status':'complete','split':'train','ncc_to_template':.8}))
            row=extract_subject_features(subject,root/'atlas.nii.gz',root/'labels.json')
            self.assertAlmostEqual(row['estimated_tiv_mm3'],1_000_000)
            self.assertEqual(row['gm_mean__ROI'],.5)
            self.assertNotIn('gm_mean__Background',row)

if __name__=='__main__':unittest.main()
