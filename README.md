# PromptEngineering NLP PJ1 — 实验方案

服务器克隆、模型部署、仅运行 M1/M2/M3 三组实验、结果查看与下载的完整步骤见 [SERVER_GUIDE.md](SERVER_GUIDE.md)。

固定使用课程 Qwen3-4B thinking，比对两篇论文在数据记录时的引用量。
当前实现包含 M1、M2、M3、M3r、M4、M5、M6；本机模拟测试验证代码和运行流程，真实模型成绩待 GPU 服务器运行。
课程原始要求在 `student/README.md`。运行器、评测器、模型部署配置及数据保持课程原版。

## 实验计划

| 方法 | 改动 | 正常调用/题 | 检验什么 |
|---|---|---:|---|
| M1 | 直接指令 + 最终 JSON 答案 | 1 | 基线 |
| M2 | M1 + 受众范围、可复用性、贡献范围、日期四个比较维度 | 1 | 原有假设 |
| M3 | M2 + TF-IDF 余弦相似度选例，最多 4 个 | 1 | 检索示例的整体作用 |
| M3r | M3 的示例换成随机 4 例 | 1 | 检索是否优于随机 |
| M4 | M2 的维度换成依据示例池修正的新维度 | 1 | 修正维度是否有效 |
| M5 | M4 + 两种 A/B 顺序各采样两次，多数决，平票取第一票 | 4，含重试最多 5 | 交换和投票的增益 |
| M6 | M5 + 平票或全部失败时由词项分类器裁决 | 同 M5 | 独立信号的增益 |

所有方法固定相同模型、采样参数、数据与解析规则。
M4 和 M6 的结果可以由 M5 的调用记录离线派生，批量运行默认只跑 M1、M2、M3、M3r、M5。
M4 的新维度只依据 `examples.jsonl` 的统计和公开论文，没有使用公开题标签。

**调用与失败规则。** M1 到 M4 每题最多 3 次尝试，三次仍无有效答案时明确报错，由框架计为错误。
M5 和 M6 每题总调用封顶 5 次，一次失败只会少一票。全部失败时 M5 报错，M6 用词项分类器兜底。
每题上限不超过 5 次，所以不会超出课程总预算 5×n。

**示例选择。** M3 仅从传入的 examples 建立词表和 IDF：英文词项、标题双权重、对数 TF、平滑 IDF、余弦相似度，同分按示例 ID 排序。
M3 和 M3r 依次取放得下的示例，答案不符合 A/B/A/B 时，同时交换该示例的两篇论文和答案，事实含义不变，因此配平不会牺牲相似度排名。
单次消息内容目标上限为 18,000 UTF-8 字节，给重试提示留余量；当前题全文保留，当前题本身超过预算时明确报错。
当前题会按“不区分 A/B 顺序”的方式从示例中排除，所以在示例池上调试时不会检索到答案。

**词项分类器。** M6 在 `predict` 内用示例池训练 TF-IDF 差值核岭回归，只用于平票和全部失败的情况。
课程规则没有禁止这种做法，但它不属于提示词工程，正式采用前建议向助教确认，并在报告中说明。

## 本机调试（无需模型）

Python 3.10+，无新增第三方依赖：

```bash
cd student
python -m unittest discover -p "test_*.py"
python experiment.py --mock --limit 3
python experiment.py --mock
python experiment.py --mock --dev
```

模拟模型固定回答 A，仅检查完整流程，任何模拟准确率都不能写成正式成绩。
模拟输出保存在 `student/runs_mock/` 和 `student/runs_mock_dev/`，真实输出保存在 `student/runs/` 和 `student/runs_dev/`，均不提交 Git。

## Linux GPU 服务器运行

上传仓库文件，或克隆后进入目录：

```bash
git clone https://github.com/StevenPang203426/PromptEngineering_NLP_PJ1.git
cd PromptEngineering_NLP_PJ1/student
bash deploy/start.sh
```

等待模型就绪后，在同一服务器新终端进入 `student`：

```bash
source .lab-runtime/venv/bin/activate
python check_server.py
python experiment.py --limit 3
python experiment.py --dev
python experiment.py
```

建议按上面的顺序运行：

1. 先跑 3 题，检查无效返回和结果页中的 `length_finishes`。框架没有设置 `max_tokens`，思考过长会拿不到答案。
2. 用 `--dev` 在示例池上调解析、交换和投票流程，结果保存在 `runs_dev/`。
3. 最后跑公开集。批量运行默认约 800 次模型调用，其中 M1、M2、M3、M3r 各 100 次，M5 400 次，各方法独立使用 500 次预算。

新维度是从示例池归纳的，在示例池上评估会偏乐观，它的效果要看公开集。
出现无效返回时批量入口会继续运行后面的方法，并在最后列出各方法的无效题数。
单独重跑一个方法：`python experiment.py --method m3r`。运行中断后不会自动恢复已完成的方法。
小批运行只检查流程，不用于方法排名。

## 分析结果

`experiment.py` 结束时会打印对应的分析命令，也可以手动传入同一数据集上的运行目录：

```bash
python analyze_votes.py runs/<m2目录> runs/<m5目录> --reference m2
```

输出各方法准确率、A/B 翻转率、平票率、模型选择位置 A 的比例，以及和参考方法的逐题比较（精确 McNemar）。
传入 M5 目录时会自动派生 M4 和 M6；如果同时传入 M4 或 M6 的实际运行目录，以实际运行结果为准。

## 文件与结果

- `student/solution.py`：单文件实现源，默认 M1，尚未根据真实成绩选择最终版本。
- `student/experiment.py`：生成 `experiments/` 下各方法的独立文件，并调用原课程运行器。
- `student/analyze_votes.py`：从运行记录离线计算派生方法、翻转率和配对比较，不调用模型。
- `student/test_experiments.py`、`student/test_analyze_votes.py`：受限执行环境、解析、预算、检索、投票和分析工具的测试。
- `student/runs/reports/`：真实运行 HTML 排行榜；各运行目录保存代码快照、参数、逐题结果、调用及评测记录。

`experiments/` 是自动生成目录。修改公共实现请编辑 `solution.py`，再运行 `python experiment.py --build-only`；不要直接修改生成文件。
服务器完成后保存准确率、正确数、无效返回、调用数、token 用量、耗时和真实截图，分析改对/改错案例。
公开测试题及其标签仅用于评测与误差分析，不能加入示例池或提示材料。
课程数据很可能源自公开的 SciJudgeBench，该数据集带有引用数和答案，不要下载或用它查询答案。
结果接近时不把一两题差异直接视为稳定提升，结合逐题差异、成本和必要复测选择最终方案。

确定最终方法后（下例仅演示选择 M5，并不代表 M5 最优）：

```bash
cp experiments/m5.py solution.py
python run.py --limit 3
python run.py
```

最终报告正文 2—4 页，放真实截图和各方法对照结果，完整提示词和动态构造方式放附录。
最终提交仅需 `report.pdf` 和选定的 `solution.py`，打包为 `学号_姓名.zip`。

## 方法参考

- [Qwen3-4B 官方模型说明](https://huggingface.co/Qwen/Qwen3-4B)：thinking 模式和输出格式。
- [AI Can Learn Scientific Taste](https://arxiv.org/abs/2603.14473)：同形式的论文引用比较任务及基线结果。
- [Judging the Judges](https://arxiv.org/abs/2406.07791)：两两比较中的位置偏置，是 A/B 交换的依据。
- [Self-Consistency](https://arxiv.org/abs/2203.11171)：多次采样后投票。
- [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837)：示例标签正确性的作用有限，是 few-shot 对照实验的动机之一。
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)：相关示例选择的动机；本实现使用词项检索，并非原论文语义编码器的复现。
- [Fantastically Ordered Prompts](https://aclanthology.org/2022.acl-long.556/)：示例顺序敏感性，因此本版固定排序与标签顺序。

课程材料附带的数据许可见 `student/DATA_LICENSE.txt`。
