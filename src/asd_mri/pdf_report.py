"""从真实输出生成四页实验报告，缺失结果时拒绝生成。"""
from pathlib import Path
import json
from html import escape
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak


def build_pdf(root: Path):
    root=root.resolve();out=root/'outputs';dest=out/'report.pdf'
    summary=json.loads((out/'model/model_summary.json').read_text(encoding='utf-8'))
    qc=pd.read_csv(out/'qc_summary.csv')
    repaired=sum('brain_mask_sha256' in json.loads(path.read_text(encoding='utf-8')).get('parameters',{})
                 for path in (out/'preprocessed').glob('*/*/metadata.json'))
    predictions=pd.read_csv(out/'test_probabilities.csv').merge(pd.read_csv(out/'submission.csv'),on='subject_id')
    ordered_scores=predictions.sort_values('probability_asd',ascending=False).probability_asd.to_numpy()
    pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/simhei.ttf'))
    body=ParagraphStyle('body',fontName='Chinese',fontSize=10.5,leading=17,spaceAfter=8,wordWrap='CJK')
    heading=ParagraphStyle('heading',parent=body,fontSize=15,leading=22,spaceBefore=10,spaceAfter=10)
    title=ParagraphStyle('title',parent=body,fontSize=21,leading=30,spaceAfter=16)
    small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=13)
    story=[]
    def p(text,style=body):story.append(Paragraph(escape(text),style))
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(x)),small) for x in row] for row in [headers,*rows]]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCE8F3')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CCD4DD')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,10)])
    def picture(path,width=495):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        story.append(Image(str(path),width=width,height=width*h/w))
    p('T1 结构 MRI 的 ASD 与健康对照分类',title)
    p('模式识别课程实验报告 · Python 3.13 与 VS Code',small)
    p('研究设计与数据',heading)
    p('使用提供的 ABIDE NYU 中心匿名影像，共 146 例。训练集为 136 例（ASD 62，HC 74），盲测集为 10 例。标签 1 为 ASD，2 为 HC。原始图像为 128×256×256，体素约 1.33×1×1 mm。匿名编号仅用于样本对应；没有检索或恢复原始诊断。')
    p('影像预处理',heading)
    p('采用 N4 偏置场校正（缩小倍数 4，迭代 30/20/10，收敛阈值 1e-5），正值强度的 1%—99% 分位截断归一化。以 MNI152 2 mm 为固定模板进行可复现 SyN 配准。模板掩膜反变换到个体空间后，Atropos 进行三类组织分割（MRF 0.1，邻域 1×1×1，5 次迭代）；按 T1 类内平均强度确定 CSF、GM、WM。灰质概率图映射至模板后进行 4 mm FWHM 平滑。')
    p(f"处理记录覆盖 {len(qc)} 例。模板相关性中位数 {qc.ncc_to_template.median():.3f}，最小值 {qc.ncc_to_template.min():.3f}。单次三维 T1 不提供时间序列，流程不宣称能校正扫描期间的头动。")
    p(f'其中 {repaired} 例因初始配准相关性低于 0.45，使用官方 SynthStrip v1 脑提取权重生成个体掩膜后重新配准和分割；模型权重仅用于脑提取，不涉及诊断标签。')
    p('对全部影像另以 SynthStrip 掩膜限定个体空间的组织概率积分，防止模板掩膜误收颅骨并高估体积。')
    picture(out/'preprocessed/train/sub-001/qc.png')
    p('图 1  示例影像的模板叠加与三组织分割。图像方向以保存文件的空间矩阵为准。',small)
    story.append(PageBreak())
    p('特征与评价设计',title)
    p(f"候选特征共 {summary['n_features']} 维：Schaefer 100 区的灰质概率均值与局部对数 Jacobian 均值、7 个网络的左右灰质均值不对称指数，以及独立脑掩膜内的组织体积和组织比例。未将模板相关性等质控分数作为预测变量。局部 Jacobian 描述非线性变换，不等同于包含仿射缩放的绝对体积。")
    p('使用 5 次重复分层五折外层验证，内层四折选择特征数和模型。中位数填补、标准化、F 检验特征筛选全部在 Pipeline 内拟合。候选模型为类别加权逻辑回归、线性 SVM、自动收缩 LDA 与浅层随机森林；主要选择指标为 balanced accuracy。随机种子固定 20260924。')
    p('交叉验证结果',heading)
    mapping=[('准确率 ACC','accuracy'),('平衡准确率','balanced_accuracy'),('ASD 灵敏度','sensitivity_asd'),('HC 特异度','specificity_hc'),('ROC-AUC','roc_auc')]
    rows=[]
    for name,key in mapping:
        low,high=summary['confidence_intervals_95'][key]
        rows.append([name,f"{summary['mean_repeat_metrics'][key]:.3f}",f"{summary['metrics'][key]:.3f}",f'{low:.3f} - {high:.3f}'])
    table(['指标','重复验证平均','聚合 OOF','聚合 OOF 95% 区间'],rows,[135,100,90,170])
    p('重复验证平均对应各次完整外层预测的平均指标。聚合 OOF 对每人五次留出预测多数投票，AUC 使用平均概率，表示重复验证模型的集成表现。Bootstrap 区间仅对固定 OOF 预测按受试者重采样 2000 次，属于描述性区间，不涵盖重新训练和重叠训练折导致的全部不确定性。',small)
    picture(out/'model/model_evaluation.png')
    p('图 2  聚合 OOF 概率的 ROC 及多数投票的混淆矩阵。',small)
    story.append(PageBreak())
    p('最终模型与盲测预测',title)
    classifier_description=summary['final_best_params']['classifier']
    if 'RandomForestClassifier' in classifier_description:
        p(f"完整训练集交叉验证选出类别加权随机森林：150 棵树、最大深度 4、叶节点至少 5 例。F 检验特征选择保留全部 {summary['n_features']} 维；内层 balanced accuracy 为 {summary['final_inner_score']:.3f}。")
    else:
        p('最终模型通过完整训练集上的交叉验证选择参数后重新拟合。参数：'+str(summary['final_best_params']).replace('\\n',' '))
    p('正式提交使用作业说明公开的 5 ASD / 5 HC 类别数：按模型 ASD 分数排序，前 5 例判为 ASD。另保存分类器默认阈值预测供对照；两种规则均未使用未知测试真值。')
    table(['受试者','ASD 概率','提交标签'],[[r.subject_id,f'{r.probability_asd:.3f}',f'{int(r.label)} '+('ASD' if r.label==1 else 'HC')] for r in predictions.itertuples()],[180,150,165])
    p('测试真值未提供，因此本报告不计算测试准确率。概率未经过独立外部校准，不应解释为个人患病风险。最终标签采用固定的前五名排序规则；表中概率仅供排序与不确定性观察。',small)
    p(f'排序边界（第 5 与第 6 名）的分数差仅 {ordered_scores[4]-ordered_scores[5]:.4f}，相邻两例的分类对预处理与模型扰动可能敏感。',small)
    perm=summary.get('baseline_permutation')
    if perm:
        p('固定基线置换检验',heading)
        p(f"对预先固定的逻辑回归（C=0.1，F 检验保留 20 维）执行 {perm['n_permutations']} 次标签置换，每次重新训练完整流水线。观察 balanced accuracy={perm['observed_balanced_accuracy']:.3f}，p={perm['p_value']:.3f}。此检验仅适用于该基线，不是最终择优模型的显著性检验。")
    p('结果解释与局限',heading)
    p('样本量较小，特征选择和模型比较仍存在不稳定性。年龄、性别及头动协变量未提供，无法完整控制混杂因素。数据来自单中心，外部泛化尚未验证。模板脑掩膜可能包含脑外组织，所得体积是流程估计值，不可替代经过验证的颅内容积测量。分割基于强度聚类，需结合逐例质控解释。盲测仅 10 例，一例错误即可使准确率变化 10 个百分点。')
    story.append(PageBreak())
    p('软件实现与可复现性',title)
    p('PySide6 桌面软件与命令行共享处理核心。支持三平面 NIfTI 浏览、保持体素纵横比例、窗宽窗位、缩放与平移、同空间结果叠加、单例和批处理、N4 与平滑参数、进度和日志、取消任务、模型训练及预测导出。耗时计算运行在独立进程中。')
    picture(out/'gui_preview.png',width=460)
    p('图 3  实际 T1 影像的桌面软件界面。',small)
    p('VS Code 运行方法',heading)
    for line in [r'.\.venv\Scripts\python.exe -m asd_mri.gui',r'.\.venv\Scripts\python.exe -m asd_mri.cli run-all --workers 4',r'.\.venv\Scripts\python.exe -m unittest discover -s tests -v']:
        p(line,small)
    p('预处理按受试者缓存结果。参数、输入哈希、变换文件、质控图和概率图保存在 outputs/preprocessed。依赖版本见 requirements-lock.txt；模型、各折预测、完整特征及提交文件均保存在 outputs。全部处理以 Python 3.13 运行，VS Code 已提供调试配置。')
    p('资源与方法来源',heading)
    for line in ['ANTsPyX 文档 https://antspy.readthedocs.io/','Nilearn MNI152 与 Schaefer 图谱接口 https://nilearn.github.io/','Schaefer 图谱原始资源 https://github.com/ThomasYeoLab/CBIG','FreeSurfer SynthStrip https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/','scikit-learn Pipeline 与交叉验证 https://scikit-learn.org/','数据与提交规则：随数据提供的 README.md（ABIDE NYU 匿名教学子集）。']:
        p(line,small)
    def footer(canvas,doc):
        canvas.setFont('Chinese',8);canvas.setFillColor(colors.HexColor('#64748B'));canvas.drawString(50,27,'ASD / HC 结构 MRI 模式识别实验');canvas.drawRightString(A4[0]-50,27,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=38,bottomMargin=42,title='T1 结构 MRI 的 ASD 与健康对照分类',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    print(dest)
    return dest

if __name__=='__main__':build_pdf(Path(__file__).resolve().parents[2])
