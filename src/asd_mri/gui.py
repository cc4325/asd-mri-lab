"""科研教学桌面工具：影像浏览、后台处理与结果检查。"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QPushButton,QFileDialog,QLabel,QSlider,QPlainTextEdit,QTabWidget,QSplitter,QMessageBox,
    QDoubleSpinBox,QComboBox,QProgressBar)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

ROOT = Path(__file__).resolve().parents[2]


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        font_path=Path('C:/Windows/Fonts/simhei.ttf')
        if font_path.exists():QFontDatabase.addApplicationFont(str(font_path))
        self.setFont(QFont('SimHei',10))
        self.setWindowTitle('脑影像实验室 · T1 MRI 预处理与分类')
        self.resize(1280,850)
        self.data = None
        self.overlay = None
        self.image = None
        self.process = None
        self.image_path = None
        self.finished_count=0
        self.output_buffer=''
        self.setStyleSheet('QMainWindow,QWidget {background:#f3f5fa;color:#182c45;} QPushButton {padding:9px;background:#dce8f6;border:1px solid #b8cce3;border-radius:5px;} QPlainTextEdit {background:white;}')
        container=QWidget(); self.setCentralWidget(container); layout=QVBoxLayout(container)
        title=QLabel('脑影像实验室   |   T1 MRI'); title.setStyleSheet('font-size:24px;font-weight:600;padding:10px;'); layout.addWidget(title)
        layout.addWidget(QLabel('三平面浏览 · N4 校正 · MNI 配准 · 组织分割 · ASD / HC 分类（科研教学）'))
        bar=QHBoxLayout();layout.addLayout(bar)
        for text,slot in [('打开影像',self.open_image),('叠加结果',self.open_overlay),('处理当前影像',self.single),('批量预处理',self.batch),('修复低质控',self.repair_qc),('取消任务',self.cancel),('模型训练',self.train),('导出预测',self.predict)]:
            button=QPushButton(text);button.clicked.connect(slot);bar.addWidget(button)
        self.info=QLabel('请打开 .nii 或 .nii.gz 文件');layout.addWidget(self.info)
        self.figure=Figure(figsize=(12,5),facecolor='#f3f5fa');self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas,1)
        layout.addWidget(NavigationToolbar2QT(self.canvas,self))
        self.axes=self.figure.subplots(1,3)
        controls=QHBoxLayout();layout.addLayout(controls)
        self.sliders=[]
        for name in ['矢状面','冠状面','横断面']:
            column=QVBoxLayout();column.addWidget(QLabel(name)); slider=QSlider(Qt.Horizontal);slider.setRange(0,1);slider.valueChanged.connect(self.redraw);column.addWidget(slider);controls.addLayout(column);self.sliders.append(slider)
        controls2=QHBoxLayout();layout.addLayout(controls2)
        self.width=QDoubleSpinBox();self.width.setRange(0.001,100000);self.width.setValue(1)
        self.level=QDoubleSpinBox();self.level.setRange(-100000,100000);self.level.setValue(0.5)
        self.alpha=QDoubleSpinBox();self.alpha.setRange(0,1);self.alpha.setSingleStep(.1);self.alpha.setValue(.35)
        for label,spin in [('窗宽',self.width),('窗位',self.level),('叠加透明度',self.alpha)]:
            controls2.addWidget(QLabel(label));controls2.addWidget(spin);spin.valueChanged.connect(self.redraw)
        self.smooth=QDoubleSpinBox();self.smooth.setRange(0,12);self.smooth.setValue(4);self.smooth.setSuffix(' mm')
        self.n4=QComboBox();self.n4.addItems(['4','2'])
        controls2.addWidget(QLabel('平滑 FWHM'));controls2.addWidget(self.smooth)
        controls2.addWidget(QLabel('N4 缩小倍数'));controls2.addWidget(self.n4)
        self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);layout.addWidget(self.progress)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(160);layout.addWidget(self.log)
        self.redraw()

    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,'打开 NIfTI',str(ROOT),'NIfTI (*.nii *.nii.gz)')
        if path: self.load_image(path)

    def load_image(self,path):
        try:
            self.image=nib.as_closest_canonical(nib.load(path));self.data=self.image.get_fdata(dtype=np.float32)
            self.image_path=path
            if self.data.ndim != 3: raise ValueError('只支持三维 T1 影像')
            self.overlay=None
            for i,slider in enumerate(self.sliders):
                slider.blockSignals(True);slider.setRange(0,self.data.shape[i]-1);slider.setValue(self.data.shape[i]//2);slider.blockSignals(False)
            lo,hi=np.percentile(self.data[np.isfinite(self.data)],[1,99]);self.width.setValue(max(hi-lo,.001));self.level.setValue((hi+lo)/2)
            spacing=' × '.join(f'{float(v):.2f}' for v in self.image.header.get_zooms())
            self.info.setText(f'{Path(path).name}   |   {self.data.shape}   |   体素 {spacing} mm   |   RAS+ 方向')
            self.redraw()
        except Exception as exc: QMessageBox.warning(self,'读取失败',str(exc))

    def open_overlay(self):
        if self.image is None: return
        path,_=QFileDialog.getOpenFileName(self,'打开同空间叠加影像',str(ROOT/'outputs'),'NIfTI (*.nii *.nii.gz)')
        if path:
            try:
                from nibabel.processing import resample_from_to
                self.overlay=resample_from_to(nib.load(path),self.image,order=0).get_fdata(dtype=np.float32);self.redraw()
            except Exception as exc: QMessageBox.warning(self,'叠加失败',str(exc))

    def redraw(self,*args):
        for i,ax in enumerate(self.axes):
            ax.clear();ax.set_facecolor('#182c45');ax.axis('off')
            if self.data is not None:
                cut=np.rot90(np.take(self.data,self.sliders[i].value(),axis=i))
                width=self.width.value();level=self.level.value()
                other=[j for j in range(3) if j!=i];zooms=self.image.header.get_zooms()
                aspect=zooms[other[1]]/zooms[other[0]]
                ax.imshow(cut,cmap='gray',vmin=level-width/2,vmax=level+width/2,aspect=aspect)
                if self.overlay is not None:
                    overlay=np.rot90(np.take(self.overlay,self.sliders[i].value(),axis=i))
                    ax.imshow(np.ma.masked_where(overlay==0,overlay),cmap='turbo',alpha=self.alpha.value(),aspect=aspect)
                ax.set_title(['Sagittal','Coronal','Axial'][i])
        self.figure.tight_layout();self.canvas.draw_idle()

    def start(self,args,script=False):
        if self.process and self.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'处理中','请等待当前任务结束');return
        self.process=QProcess(self);self.process.setWorkingDirectory(str(ROOT));self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.output_buffer='';self.finished_count=0;self.progress.setRange(0,0)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(lambda code:self.log.appendPlainText(f'任务结束，退出码 {code}'))
        self.process.errorOccurred.connect(lambda error:self.log.appendPlainText('进程启动失败：'+self.process.errorString()))
        self.log.appendPlainText('开始任务：'+' '.join(args))
        command=['-X','utf8','-u',*args] if script else ['-X','utf8','-u','-m','asd_mri.cli',*args]
        self.process.start(sys.executable,command)

    def read_output(self):
        self.output_buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.output_buffer:
            line,self.output_buffer=self.output_buffer.split('\n',1);self.log.appendPlainText(line)
            try:
                data=json.loads(line)
                if 'total' in data:self.progress.setRange(0,data['total'])
                if 'subject_id' in data:self.finished_count+=1;self.progress.setValue(self.finished_count)
            except (ValueError,TypeError):pass

    def parameters(self):
        return ['--n4-shrink',self.n4.currentText(),'--smooth',str(self.smooth.value())]

    def single(self):
        if not self.image_path:return
        destination=QFileDialog.getExistingDirectory(self,'选择结果目录',str(ROOT/'outputs'))
        if destination:self.start(['preprocess','--file',str(self.image_path),'--output',destination,'--workers','1',*self.parameters()])

    def batch(self):
        source=QFileDialog.getExistingDirectory(self,'选择影像目录',str(ROOT/'train'))
        if not source:return
        destination=QFileDialog.getExistingDirectory(self,'选择结果目录',str(ROOT/'outputs'))
        if destination:self.start(['preprocess','--input',source,'--output',destination,'--workers','1',*self.parameters()])

    def cancel(self):
        if self.process and self.process.state()!=QProcess.NotRunning:self.process.kill();self.log.appendPlainText('已取消当前任务；重新运行将复用成功结果。')

    def train(self):self.start(['train'])
    def predict(self):self.start(['predict'])
    def repair_qc(self):self.start([str(ROOT/'scripts/repair_low_qc.py')],script=True)
    def closeEvent(self,event):
        self.cancel();event.accept()


def main():
    app=QApplication(sys.argv);window=Window();window.show();sys.exit(app.exec())

if __name__=='__main__':main()
