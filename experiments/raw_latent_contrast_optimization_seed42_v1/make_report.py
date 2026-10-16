"""Generate executed-only Markdown from anonymous public artifacts."""
import argparse
import json
from pathlib import Path
import pandas as pd

LABELS={'B0_ORIGINAL_POOL_MACRO_LOSS':'B0 原池化/原损失',
        'B1_ORIGINAL_POOL_EZ_LOSS':'B1 原池化/EZ损失',
        'S1_SELECTED_POOL_EZ_LOSS':'S1 内部选择池化',
        'A1_SELECTED_POOL_ATTENTION_EZ_LOSS':'A1 Raw潜变量注意力',
        'SELECTED_POLICY':'Development选择策略'}
COLS={'auroc':'AUROC','auprc':'AP','macro_f1':'Macro-F1','f1':'EZ-F1','nez_f1':'NEZ-F1',
      'accuracy':'ACC','mrr':'MRR','hit_at_1':'Top-1','ez_fraction_bias':'EZ-Frac. Bias'}


def table(frame):
    shown=frame.copy()
    for col in shown:
        if pd.api.types.is_float_dtype(shown[col]):shown[col]=shown[col].map(lambda v:f'{v:.6f}')
    return shown.to_markdown(index=False,disable_numparse=True)


def main(root):
    load=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    status=load('OUTER_STATUS.json');verification=load('INDEPENDENT_RESULTS_VERIFICATION.json')
    assert status['status']=='ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED' and verification['status']=='PASS'
    target=load('TARGET_STATUS.json');train=load('TRAIN_STATUS.json');screen=load('STAGE1_STATUS.json')
    values=pd.read_csv(root/'MAIN_RESULTS.csv').set_index('model')
    primary=values.loc[list(LABELS)].reset_index();primary['model']=primary.model.map(LABELS)
    primary=primary.rename(columns=COLS|{'model':'模型'})
    selected=values.loc['SELECTED_POLICY'];ci=target['patient_bootstrap_95percent']
    target_text='点估计达到' if target['point_target_met'] else '未达到'
    selection=pd.read_csv(root/'OWN_FIT_POOL_SELECTION.csv',dtype={'recipe':str});selection['recipe']=selection.recipe.str.zfill(4)
    recipes=selection[['fold','recipe']].rename(columns={'fold':'折','recipe':'参考/分阶段/发作SD/患者Z'})
    effects=pd.read_csv(root/'FACTOR_CONDITIONAL_EFFECTS.csv')
    effect=effects[effects.metric.isin(['f1','auroc'])].groupby(['factor','metric']).paired_delta.agg(['mean','min','max']).reset_index()
    effect['factor']=effect.factor.map({0:'发作前潜变量参考',1:'0–8/8–20秒分阶段',2:'跨发作SD',3:'Patient-wise Z'})
    effect['metric']=effect.metric.map(COLS)
    effect=effect.rename(columns={'factor':'组件','metric':'指标','mean':'平均条件差','min':'最小条件差','max':'最大条件差'})
    paired=pd.read_csv(root/'PAIRED_INTERVALS.csv');paired=paired[paired.metric.isin(['auroc','auprc','f1','macro_f1','mrr','hit_at_1'])].copy()
    paired['left']=paired.left.map(LABELS);paired['right']=paired.right.map(LABELS)
    paired['metric']=paired.metric.map(COLS)
    paired=paired[['left','right','metric','delta','low','high']].rename(columns={'left':'模型','right':'对照','metric':'指标','delta':'配对差','low':'95%下界','high':'95%上界'})
    alternate=values.loc[[m+'__ALT_THRESHOLD' for m in list(LABELS)[:4]],['f1','macro_f1','nez_f1','accuracy','auroc']].reset_index()
    alternate['model']=alternate.model.map({m+'__ALT_THRESHOLD':LABELS[m]+('，EZ优先阈值' if m.startswith('B0') else '，Macro优先阈值') for m in list(LABELS)[:4]})
    alternate=alternate.rename(columns=COLS|{'model':'同权重阈值敏感性'})
    loss_delta=float(values.loc[list(LABELS)[1],'f1']-values.loc[list(LABELS)[0],'f1'])
    pool_delta=float(values.loc[list(LABELS)[2],'f1']-values.loc[list(LABELS)[1],'f1'])
    attention_delta=float(values.loc[list(LABELS)[3],'f1']-values.loc[list(LABELS)[2],'f1'])
    report=f'''# 纯Raw潜变量组件与轻量优化：完整五折研究

## 1. 结果

预先定义的目标是**患者平均EZ阳性F1＞0.65**。Development选择策略的Outer EZ-F1为
**{selected.f1:.6f}**，患者配对抽样的绝对95%区间为**[{ci[0]:.6f}, {ci[1]:.6f}]**，
本轮**{target_text}目标**。Macro-F1为{selected.macro_f1:.6f}，AUROC为{selected.auroc:.6f}，AP为{selected.auprc:.6f}。
这些指标不能互相替代，也没有按Outer挑选主模型。

本研究不使用人工生理特征或独立D1分支。模型输入仅为现有Raw250波形、原有效窗口掩码、
缓存相对时间和通道身份。历史Outer已多次被研究使用，因此本轮是**探索性重复测试**，不是新的独立确认。

## 2. 数据与模型

全部模型使用相同80位患者、7,635个通道、256次发作、1,471,965个原有效Raw窗口，
保留原五折patient-held-out划分。FIT患者数为51/51/50/52/51；Outer为16/16/17/15/16。
患者是统计单位，发作和窗口不被当作独立样本。跨来源结果是在来源混合训练下的患者留出，
不是leave-center-out试验。缓存相对发作时间沿用原数据，未新增临床时间标注验证或低频恢复。

共享CNN为两层1D卷积，通道宽度12/24，核9/7，步长4/4，GELU和全局平均池化。
每个有效2秒波形独立中心化、除以其population SD并截断到[-8,8]。
Raw24D按阶段、发作均值和SD形成96槽位，分类头为96→64→1无界NEZ logit。
预测EZ概率为sigmoid(-NEZ logit)。基础模型8,433参数，注意力模型8,851参数。
原160输入头的64个生理槽位为零；本轮纯96输入映射的历史表征误差0，logit最大误差2.3842e-7。
这个映射只验证数值接口，不是一次新的独立性能比较。

新模型均从头训练，配对共享Raw CNN与头初始化，不加载历史D1或训练后的Raw权重作初始化。
四个臂均采用AdamW(lr0.0003,wd0.001)、clip1、最多30轮、最少6轮、patience6，
每个完整FIT患者一次更新；FP32、TF32关闭。完整患者池化及Z的梯度传回CNN。

## 3. 四个组件的低成本筛选

完整2⁴矩阵检验同通道发作前latent扣除、0–8/8–20秒分阶段、可用发作population SD、
全患者latent Z。关闭分阶段时重复0–20秒汇总，关闭SD时置零SD槽位，保持同一96输入头接口。
槽位一致不代表有效信息维数一致。

复用20个真实inner-cross-fitted Raw编码器，原编码器仅见自己内层train/cal。
560次固定C1逻辑回归拟合和4,080个逻辑患者结果行完成。每个Outer FIT内独立使用
train/cal/choice/held角色，阈值只用cal，16组合选择只用该折FIT内结果；
原Development和该折Outer不参与此阶段。
255次患者-折出现来自79个唯一身份，重叠折不是独立重复。
所有16组合的assessment最高值参与选择，故筛选表不是无偏验证估计。
编码器原本按完整组件训练，因此该筛选也不能代替逐组件从头训练的因果消融。

{table(recipes)}

五折选中次数：发作前参考{selection.recipe.str[0].eq('1').sum()}/5、分阶段{selection.recipe.str[1].eq('1').sum()}/5、
发作SD{selection.recipe.str[2].eq('1').sum()}/5、患者Z{selection.recipe.str[3].eq('1').sum()}/5。
这里的频率是算法在重叠FIT数据上的描述，不是统计显著性或通用必要性证明。

下表为打开组件减关闭组件的条件差，平均/范围覆盖其余三个组件的8种背景。
范围不是置信区间，两个F1不同的阈值体系不混用。

{table(effect)}

![16组合及选择稳定性](figures/factorial_screen.png)

## 4. 端到端优化对照

B0保留原完整池化与0.75加权BCE+0.25 soft Macro-F1损失；B1仅将优化及选择目标改为EZ优先，
采用0.5同一加权BCE+0.25 soft EZ-F1+0.25患者内全EZ/NEZ配对排序损失。
S1在相同EZ损失下采用各折FIT选择的池化；A1再加入24→8→1 Raw latent阶段内与跨发作注意力。
注意力初始为均匀分布，不使用生理量。注意力模块本身不被宣称为新颖算法。

所有**20个实际完整训练单元、{train['epochs']}轮**完成。B0按Development Macro-F1优先选轮次和阈值；
其他臂按Development EZ-F1优先，再按Macro-F1/更早轮次打平。
最终部署策略也只在Development中逐折选择，五折选择为
{', '.join(str(f)+': '+LABELS[train['deployment'][str(f)]] for f in range(1,6))}。
阈值搜索均为EZ概率0到1、步长0.005，打平时接近0.5及更小阈值优先。
因此B1减B0包含训练目标、轮次和决策规则变化，不能仅归因于损失公式。

## 5. Outer主结果：80位患者等权

{table(primary)}

对照差的EZ-F1点估计：B1−B0={loss_delta:+.6f}，S1−B1={pool_delta:+.6f}，A1−S1={attention_delta:+.6f}。
是否支持增量价值须同时看下列配对区间与AUROC/AP/NEZ-F1，不能只看最佳单项点估计。

![完整模型与配对差](figures/end_to_end_results.png)

### 同权重阈值敏感性

以下只改变已冻结权重的Development阈值目标，不重训练、不据Outer选阈值。
AUROC、AP与排序分数不变；F1变化属于决策规则影响。

{table(alternate)}

### 预先规定的患者配对区间

全部区间来自同一10,000次患者抽样，保留模型间配对。报告的是固定训练结果下的患者不确定性，
不是多seed训练不确定性；区间未进行多重比较校正。跨零不能证明有改进，也不证明等效。

{table(paired)}

## 6. 解释与边界

本轮将“组件是否给冻结表征增加条件信息”“内部选中的汇总能否转移到端到端训练”
以及“Raw-only注意力是否增加最终定位能力”分开检验。最终集合由合法内部数据产生，
不要求保留原完整四组件。没有把低成本逻辑回归分数写成CNN结果，也没有把Patient Z
频率解释为生理机制。

单头CNN使用每窗口标准化，绝对波形幅度被去除。该表示的局限不等于原始EEG不存在可用信息。
本轮只检验预先固定的小模型、池化及损失范围，不证明Raw-only模型的性能上限。
本轮没有完整逐组件从头训练矩阵，冻结编码器的16组合效应不能直接量化每个组件的端到端训练贡献。
跨来源样本量不均，少于5位患者的来源汇总予以隐藏，不按来源表现排除患者。
单seed、历史重复使用队列及Development多候选选择限制了结论的独立性。

## 7. 结果文件与执行审计

`MAIN_RESULTS.csv`为患者等权主表；`POOLED_RESULTS.csv`是单独的通道池化表；
`FOLD_RESULTS.csv`、`FOLD_STABILITY.csv`、`SOURCE_RESULTS.csv`、`METRIC_COVERAGE.csv`覆盖稳定性与覆盖率。
`ABSOLUTE_INTERVALS.csv`和`PAIRED_INTERVALS.csv`保留全部九项指标。
每个模型先冻结checkpoint、配方、轮次与阈值，再封存320个患者级Raw分数；之后读取完整标签。
独立重算{verification['patient_rows']}行患者指标，以及所有绝对/配对区间的10,000次抽样。
训练、特征缓存、权重、个体身份/标签/预测与运行日志只留可信服务器。
GitHub仅发布源代码与匿名汇总。环境与启动器问题单独记录在`ENGINEERING_LEDGER.md`，不计作性能证据。
'''
    (root.parent/'REPORT.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');main(p.parse_args().results)
