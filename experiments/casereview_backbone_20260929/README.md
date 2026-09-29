# SkinGPT-X fixed-evidence CaseReview backbone comparison

## 实验范围

仅替换 **CaseReview** 的模型：历史 `Qwen3-VL-8B-Instruct` 对比
`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`。图像、冻结Vision描述、PanDerm Top-5候选及相似度、
历史病例、8B生成的记忆卡和CaseReview提示词保持不变。**不重新抽样、不训练PanDerm、不重建指南、不重新运行Vision。**

这不是全系统backbone替换。8B稠密BF16与30B-A3B MoE FP8同时改变规模、架构与精度，
应报告为“CaseReview backbone configuration comparison”，不能称为纯参数量因果消融。
历史8B采用Transformers，本包为适配指定FP8模型采用vLLM。引擎差异需披露；
建议使用本包提供的8B入口另做同引擎对照，但本次交付不运行任何推理。

## 数据集与类数（锁定）

| 新名称 | 原manifest来源 | 训练支持类别 | 训练图像 | 严格测试图像 |
|---|---|---:|---:|---:|
| DermNet-Curated-A | DermNet225_OntologyV1 | 197 | 8,512 | 2,343 |
| DermNet-Curated-B | DermNet353_OntologyV1 | 333 | 13,438 | 3,812 |
| DermNet-Curated-C | DermNet498_OntologyV1 | 385 | 13,089 | 3,718 |

这三个名字不表示嵌套关系。只包含refine后的 `primary_candidate`，去掉没有训练支持的测试标签。
A排除2例test-only，B排除3例，C排除0例。完整refine候选池的198/335/385类不能与这里的197/333/385类混用。
总计9,873个CaseReview任务。三套数据存在来源图像重叠，不得合并成独立大样本宣称统计显著性。
工程refine不等于医生已完成本体审定，也不等于患者级独立性得到证明。

Hugging Face: https://huggingface.co/datasets/czty/SkinGPTX-Images

各数据集位于同名文件夹，包含 `classes.json`、`class_support.json`、`manifest.jsonl`、
`case_jobs.jsonl`、`excluded_test_only.json`、`audit.json` 和 `images.tar`。
图像保留原字节；提示词逐字保持原8B输入，仅元数据路径和展示数据集名改为可迁移形式。
`job_id`保留旧版标识用于配对，`legacy_dataset`记录旧名称。标签不再做额外改名。

## 一致性与评测规则

- 所有图像SHA-256、类别表和冻结任务文件均校验；训练和测试无重复图像字节。
- 模型只收到RGB查询图像及原提示词，绝不收到JSONL中的ground_truth或历史预测字段。
- 与原CaseReview一致：图像processor的 `min_pixels=200704`、`max_pixels=501760`；最大输出320 tokens；贪心生成。
- 不添加system prompt，不添加思维链指令，不启用JSON约束解码，不改变Top-5或证据顺序。
- 过长输入和运行异常直接中止，不能静默截断、跳过或作为模型错误填充。
- 模型已生成但JSON/标签解析不合格时，主指标沿用历史PanDerm Top-1回退。
  同时报告invalid数量与invalid计错的敏感性准确率，不隐瞒回退贡献。
- `score.py`仅接受三个数据集全部完成、唯一且输入哈希一致的9,873条结果。
- 输出ACC、Balanced ACC、Macro/Weighted F1、MCC、Kappa、配对bootstrap CI、McNemar与三项比较Holm校正。
- Resume绑定模型配置、软件版本、提示词与图像哈希。配置变化时必须使用新输出文件。

## 1. 新服务器环境准备

获取本次专用分支（跳过旧仓库无关的LFS资源），进入独立实验目录：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch rebuttal/casereview-backbone-20260929 --single-branch \
  https://github.com/healme-225040511/Skingpt_X.git
cd Skingpt_X/experiments/casereview_backbone_20260929
python -m unittest test_protocol.py
```

请勿使用 `python -O`，协议检查依赖assert。脚本和数据导出均不修改原项目与历史实验结果。

使用Linux、Python 3.11和支持该FP8 checkpoint的NVIDIA GPU环境；本脚本保守要求计算能力≥8.9，
不支持时停止，不自动转BF16。显存是否足够取决于GPU、上下文和tensor parallel；不能仅按A3B激活参数估算权重显存。

官方部署资料：
- https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
- https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html

官方模型卡推荐vLLM/SGLang；vLLM配方要求版本≥0.11.0。下列为环境安装入口，不是已经在目标GPU验证的环境锁文件：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm>=0.11.0' -r requirements-cpu.txt
python -m pip freeze > environment.lock.txt
nvidia-smi
```

8B和30B比较应使用同一份环境锁。运行会将实际torch/transformers/vLLM版本、GPU型号及模型配置哈希写入meta。
不在安装脚本内下载模型。请在新服务器**自行准备完整本地模型快照**，记录其HF revision；
`run.py`只接收本地路径，设置离线模式，缺少权重则失败。

## 2. 下载数据并校验（不加载模型）

采用 `release.json` 中记录的HF数据commit，而不是浮动main：

```bash
DATA_REVISION=$(python -c 'import json; print(json.load(open("release.json"))["hf_dataset_revision"])')
python prepare_data.py --data-root ./data --revision "$DATA_REVISION"
```

只下载本实验三个目录，不下载仓库其他历史数据。归档解包时拒绝目录穿越与链接项。
已经有完整本地包可用 `--local-only` 代替下载。
如目标网络上的Xet传输反复超时，可在下载命令前加 `HF_HUB_DISABLE_XET=1` 使用常规传输；
数据revision和SHA-256校验不变。不要通过跳过校验来处理下载错误。

```bash
python run.py --data-root ./data --dataset DermNet-Curated-A \
  --output ./runs/30b/A.jsonl --model-path /path/to/local/model \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --model-revision YOUR_MODEL_COMMIT --preflight-only
```

preflight不加载模型，不检查权重存在性，仅检验数据与任务输入。

## 3. 启动30B-A3B-FP8（仅在准备好的新服务器手动执行）

替换下面的本地模型路径和真实revision。单卡时TP=1，多卡按可见GPU数量设置；
若OOM可调整显存利用率或TP，但不得改变图片像素限制、证据或输出token预算。

```bash
MODEL_PATH=/path/to/Qwen3-VL-30B-A3B-Instruct-FP8
MODEL_REVISION=YOUR_MODEL_COMMIT
for ds in DermNet-Curated-A DermNet-Curated-B DermNet-Curated-C; do
  python run.py --data-root ./data --dataset "$ds" \
    --model-path "$MODEL_PATH" --model-revision "$MODEL_REVISION" \
    --model-id Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --output "./runs/30b/${ds}.jsonl" || break
done
```

中断后同命令重启即可。若单条输出在写入时损坏，脚本会报错，需保留原文件后人工恢复完整行，不会静默忽略损坏记录。
同一输出文件不能由多个进程同时写入。并行时使用 `--num-shards N --shard-index i`，每个分片独立输出文件；
模型tensor parallel与任务分片是不同概念。

## 4. 可选：同引擎8B对照

运行参数保持相同，只替换本地模型路径、revision、`--model-id Qwen/Qwen3-VL-8B-Instruct`及输出目录。
不要重新生成任何指南或任务输入。历史8B结果已保存在冻结jobs中，
如果只运行30B，可直接与历史8B配对，但正文必须披露推理引擎差异。

## 5. 完成后统计

与原8B历史预测比较：

```bash
python score.py --data-root ./data \
  --outputs ./runs/30b/*.jsonl --report-dir ./reports/30b_vs_historical8b
```

如另跑了同引擎8B：

```bash
python score.py --data-root ./data \
  --outputs ./runs/30b/*.jsonl --reference-outputs ./runs/8b/*.jsonl \
  --report-dir ./reports/30b_vs_matched8b
```

报告目录必须为新目录。不要将同一任务的完整输出和分片输出同时传入；重复ID将导致校验失败。
主产物为 `metrics.json` 和 `paired_predictions.csv`，结果按数据集分别报告。

## 6. 推荐论文表述

> We compared CaseReview backbone configurations while keeping query images, candidate lists,
> retrieved precedents, visual observations, diagnostic memory cards, and task prompts fixed.
> This evaluates the sensitivity of the final review stage to backbone choice, rather than
> end-to-end system scaling. The dense BF16 and MoE FP8 configurations differ in architecture
> and numerical precision as well as total parameter count.

如引用历史8B，再补充其Transformers引擎与新vLLM运行不同。报告真实checkpoint全名，
不能将30B-A3B写成32B，也不能把MoE总参数30B写成每token均激活30B。

## 7. 导出与验证

`export_bundle.py`为原服务器CPU导出入口，检查原manifest、8B候选和配对预测。
`test_protocol.py`运行纯CPU单元测试。不需要新模型，也不启动GPU任务。
安装CPU依赖后，`python -m unittest test_protocol test_scoring` 同时检查配对指标及不完整输出拒绝规则。
`python verify_archive.py --data-root ./data` 可在解包前逐图校验归档与manifest的完整一致性。

本交付的GPU推理入口尚未在目标FP8硬件上实跑，不能称为GPU端到端验证通过。
上传的数据仍受原始图像来源的权利与使用条款约束，工程打包不授予新的图像许可。
