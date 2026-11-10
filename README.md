# PromptEngineering NLP PJ1 — 第一版实验

固定使用课程 Qwen3-4B thinking，比对两篇论文在数据记录时的引用量。
第一版包含 M1、M2、M3；本机模拟测试验证代码和运行流程，真实模型成绩待 GPU 服务器运行。
课程原始要求在 `student/README.md`。运行器、评测器、模型部署配置及数据保持课程原版。

## 实验计划

| 方法 | 改动 | 正常调用/题 |
|---|---|---:|
| M1 | 直接指令 + 最终 JSON 答案 | 1 |
| M2 | M1 + 受众范围、可复用性、贡献范围、日期四个比较维度 | 1 |
| M3 | M2 + TF-IDF 余弦相似度选例，最多 4 个 | 1 |

所有方法固定相同模型、采样参数、数据与解析规则。M3 相对 M2 的实验检验加入检索示例的整体作用，不能单独证明检索优于随机选例；后者需另加对照。
每题最多 3 次尝试（含失败与格式重试），不超过课程总预算 5×n。
三次仍无有效答案时明确报错，由框架计为错误，不用默认答案伪装成功。

M3 仅从传入的 examples 建立词表和 IDF：英文词项、标题双权重、对数 TF、平滑 IDF、余弦相似度。
同分按示例 ID 排序；按 A/B/A/B 顺序各选相关且能放入预算的未使用示例。
单次消息内容目标上限为 18,000 UTF-8 字节，给重试提示留余量；当前题全文保留。
示例过长则跳过，材料不足时示例数量和标签比例可能减少。当前题本身超过预算时明确报错。
使用的材料及各次调用均由原框架记录，可审计选例来源。

## 本机调试（无需模型）

Python 3.10+，无新增第三方依赖：

```bash
cd student
python -m unittest -v test_experiments
python experiment.py --mock --limit 3
python experiment.py --mock
```

模拟模型固定回答 A，仅检查完整流程，任何模拟准确率都不能写成正式成绩。
模拟输出单独保存在 `student/runs_mock/`，真实输出保存在 `student/runs/`，两者均不提交 Git。

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
python experiment.py
```

最后一条依次运行三种方法，各自独立使用 500 次预算，正常合计约 300 次模型调用。
出现无效返回时批量入口会停止，先查看对应运行目录中的错误记录再继续。
单独重跑一个方法：`python experiment.py --method m3`。
运行中断后不会自动恢复已完成的方法，使用 `--method` 避免不必要的重复运行。
小批运行只检查流程，不用于方法排名。

## 文件与结果

- `student/solution.py`：单文件实现源，默认 M1，尚未根据真实成绩选择最终版本。
- `student/experiment.py`：生成独立的 `experiments/m1.py`、`m2.py`、`m3.py`，并调用原课程运行器。
- `student/test_experiments.py`：受限执行环境、解析、预算、检索与边界测试。
- `student/runs/reports/`：真实运行 HTML 排行榜；各运行目录保存代码快照、参数、逐题结果、调用及评测记录。

`experiments/` 是自动生成目录。修改公共实现请编辑 `solution.py`，再运行 `python experiment.py --build-only`；不要直接修改生成文件。
服务器完成后保存准确率、正确数、无效返回、调用数、token 用量、耗时和真实截图，分析改对/改错案例。
公开测试题及其标签仅用于评测与误差分析，不能加入示例池或提示材料。
结果接近时不把一两题差异直接视为稳定提升，结合逐题差异、成本和必要复测选择最终方案。

确定最终方法后（下例仅演示选择 M3，并不代表 M3 最优）：

```bash
cp experiments/m3.py solution.py
python run.py --limit 3
python run.py
```

最终报告正文 2—4 页，放真实截图和三组对照结果，完整提示词和动态构造方式放附录。
最终提交仅需 `report.pdf` 和选定的 `solution.py`，打包为 `学号_姓名.zip`。

## 方法参考

- [Qwen3-4B 官方模型说明](https://huggingface.co/Qwen/Qwen3-4B)：thinking 模式和输出格式。
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)：相关示例选择的动机；本实现使用词项检索，并非原论文语义编码器的复现。
- [Fantastically Ordered Prompts](https://aclanthology.org/2022.acl-long.556/)：示例顺序敏感性，因此本版固定排序与标签顺序。

课程材料附带的数据许可见 `student/DATA_LICENSE.txt`。
