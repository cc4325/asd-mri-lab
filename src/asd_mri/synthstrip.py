"""在 Python 3.13/Windows 上运行官方 SynthStrip v1 权重。

模型结构与输入变换依据 FreeSurfer mri_synthstrip；仅使用公开脑提取权重，
不包含 ASD/HC 诊断信息。权重来源和 SHA-256 记录在 resources/ 下。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel import orientations
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import binary_fill_holes, label
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2) if activation else None

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x) if self.activation is not None else x


class StripModel(nn.Module):
    def __init__(self):
        super().__init__()
        features = [16, 32, 64, 64, 64, 64, 64]
        encode = np.repeat(features[:-1], 2)
        decode = np.repeat(features[::-1], 2)
        self.nb_levels = 7
        self.pooling = [nn.MaxPool3d(2) for _ in range(self.nb_levels)]
        self.upsampling = [nn.Upsample(scale_factor=2, mode='nearest') for _ in range(self.nb_levels)]
        self.encoder = nn.ModuleList()
        previous = 1
        encoded_channels = [1]
        for level in range(6):
            blocks = nn.ModuleList()
            for value in encode[level * 2:level * 2 + 2]:
                blocks.append(ConvBlock(previous, int(value)))
                previous = int(value)
            self.encoder.append(blocks)
            encoded_channels.append(previous)
        self.decoder = nn.ModuleList()
        encoded_channels = encoded_channels[::-1]
        for level in range(6):
            blocks = nn.ModuleList()
            for value in decode[level * 2:level * 2 + 2]:
                blocks.append(ConvBlock(previous, int(value)))
                previous = int(value)
            self.decoder.append(blocks)
            previous += encoded_channels[level]
        self.remaining = nn.ModuleList()
        for value in decode[12:]:
            self.remaining.append(ConvBlock(previous, int(value)))
            previous = int(value)
        self.remaining.append(ConvBlock(previous, 1, activation=False))

    def forward(self, x):
        history = [x]
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                x = block(x)
            history.append(x)
            x = self.pooling[level](x)
        for level, blocks in enumerate(self.decoder):
            for block in blocks:
                x = block(x)
            x = self.upsampling[level](x)
            x = torch.cat([x, history.pop()], dim=1)
        for block in self.remaining:
            x = block(x)
        return x


def synthstrip_mask(image_path: Path, model_path: Path, output_path: Path, border_mm: float = 1.0) -> Path:
    native = nib.load(image_path)
    if len(native.shape) != 3:
        raise ValueError('SynthStrip 只接受三维影像')
    iso = resample_to_output(native, voxel_sizes=(1, 1, 1), order=0)
    change = orientations.ornt_transform(orientations.io_orientation(iso.affine), orientations.axcodes2ornt(('L','I','A')))
    lia = iso.as_reoriented(change)
    data = lia.get_fdata(dtype=np.float32)
    nonzero = np.argwhere(data > 0)
    if not nonzero.size:
        raise ValueError('输入影像为空')
    lower = nonzero.min(axis=0)
    upper = nonzero.max(axis=0) + 1
    shape = np.clip(np.ceil((upper - lower) / 64).astype(int) * 64, 192, 320)
    center = ((lower + upper) / 2).astype(int)
    start = center - shape // 2
    start = np.clip(start, 0, np.maximum(0, np.asarray(data.shape) - shape))
    stop = start + shape
    cropped = np.zeros(tuple(shape), dtype=np.float32)
    native_start = np.maximum(start, 0)
    native_stop = np.minimum(stop, data.shape)
    target_start = native_start - start
    target_stop = target_start + native_stop - native_start
    cropped[tuple(slice(a,b) for a,b in zip(target_start,target_stop))] = data[tuple(slice(a,b) for a,b in zip(native_start,native_stop))]
    cropped -= cropped.min()
    percentile = np.percentile(cropped, 99)
    if percentile <= 0:
        raise ValueError('输入影像缺少有效脑组织信号')
    cropped = np.clip(cropped / percentile, 0, 1)

    torch.set_num_threads(1)
    model = StripModel().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    with torch.inference_mode():
        sdt = model(torch.from_numpy(cropped[None,None])).squeeze().numpy()
    del model
    patch_affine = lia.affine.copy()
    patch_affine[:3,3] = lia.affine[:3,3] + lia.affine[:3,:3] @ start
    native_sdt = resample_from_to(nib.Nifti1Image(sdt, patch_affine), native, order=1, cval=100).get_fdata(dtype=np.float32)
    mask = native_sdt < border_mm
    components, count = label(mask)
    if count < 1:
        raise RuntimeError('SynthStrip 没有识别到脑组织')
    sizes = np.bincount(components.ravel());sizes[0] = 0
    mask = binary_fill_holes(components == sizes.argmax())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), native.affine, native.header), output_path)
    return output_path
