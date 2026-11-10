# 实验课1：提示词与上下文工程

本实验使用 **Qwen3-4B（thinking）**，根据两篇论文的标题、摘要和日期，判断哪篇在数据记录时的引用数更高。你将在`solution.py`中设计提示词和调用流程，至少尝试3种提示词工程方法，比较效果并分析原因。

## 1. 启动模型（仅供参考）

将学生包上传并解压到Linux GPU服务器，进入`student`目录。支持RTX 4090（24 GB显存）和RTX 5090（32 GB显存）。镜像需包含NVIDIA驱动、Python 3和C编译器，安装前至少留有30 GiB可用磁盘空间。精简GPU镜像建议选择100 GB系统盘；并行智算云的DeepSeek镜像预装文件较多，建议选择200 GB。

在服务器终端运行：

```bash
bash deploy/start.sh
```

脚本会自动识别显卡、安装对应的固定环境，并从ModelScope下载模型。4090需驱动550.142及以上，5090需580.65.06及以上。看到`Application startup complete`表示模型就绪。保持此窗口运行，模型会持续提供服务。

在**同一台服务器新建一个终端**（或再连接一次SSH），进入相同的`student`目录，激活环境：

```bash
source .lab-runtime/venv/bin/activate
```

后续编辑代码和运行实验都在这个新窗口中完成。程序默认连接同机模型。若启动遇到问题，可运行`bash deploy/setup.sh --check`检查显卡、驱动和磁盘。

## 2. 设计提示词与调用流程

`solution.py`只提供`predict(pair, examples, llm)`函数接口。请自行实现提示词构造、模型调用和答案提取，最终返回字符串`"A"`或`"B"`。

|接口|含义|
|---|---|
|`pair`|当前题目。`paper_a`、`paper_b`分别包含`title`、`abstract`、`date`。|
|`examples`|100条带答案的示例，结构与题目相同，另有`id`、`answer`，可用于few-shot或选例。|
|`llm.chat(messages)`|调用模型，返回`content`（最终回答）和`reasoning`（思考文本）。|

模型调用接口如下，消息内容由你设计：

```python
messages = [{"role": "user", "content": "任务要求和论文材料"}]
response = llm.chat(messages)
text = response["content"]
```

消息角色可用`system`、`user`、`assistant`。你负责从模型回答中提取A/B，也可以设计多轮比较、选例或投票。评测使用函数的最终返回值。

将方案及辅助函数写在`solution.py`中，以普通Python函数实现。可使用`json`、`re`、`math`、`statistics`，以及`collections.Counter/defaultdict/deque`、`functools.reduce`和常用的`itertools`、`random`操作。模型访问统一通过`llm.chat`；运行环境提供内存中的题目与示例，调试时通过结果页面查看回答和异常。

**实验规则：** 模型固定为Qwen3-4B thinking。100条公开测试题用于评测和误差分析；few-shot及检索材料从100条示例池中选择。将公开题或标签（含改写内容）用于few-shot、提示材料或检索库，或绕过框架联网、调用其他模型，视为作弊，整个实验记0分。

## 3. 运行与查看结果

完成`predict`函数后，先运行3题，检查调用和答案提取：

```bash
python run.py --limit 3
```

确认流程正常后，运行完整公开集：

```bash
python run.py
```

程序自动检查连接，以16并发运行并评估。每次运行共享`5×n`次调用预算，`n`为本次题数：3题最多15次，100题最多500次。失败与重试计入预算，可自行分配各题的调用次数。`solution.py`最多64 KiB。

结果按时间戳保存在`runs/`中。将该目录下载到本机，打开`runs/reports/`内最新的HTML，查看按准确率排序的结果，点击记录查看逐题回答和错误。小批运行标记为“不完整”；完整公开集成绩以100题计算，漏答、异常和非法返回计错。

修改`solution.py`后重新运行，比较不同方法的结果。可从直接指令、few-shot、分步比较、多次采样投票等方法入手，记录关键改动和观察。

汇总已有结果使用`python evaluate.py`。指定测试集使用`python run.py --data 数据文件.jsonl`。

## 4. 撰写报告

使用`report/main.tex`中的中文NeurIPS模板，将`main.tex`与`neurips_2026.sty`放在同一目录，以XeLaTeX编译。Overleaf中也请选择XeLaTeX；本地编译两次可更新交叉引用。在`report/`中创建`figures/`目录放截图，模板示例使用`run.png`。

1. 正文2—4页，最多4页，图表与截图计入正文；参考文献与附录另起页。
2. 包含真实运行截图，展示自己的实验结果。
3. 至少尝试3种提示词工程方法，记录并分析效果，说明最终选择的理由。
4. 最终选定的完整提示词放入附录，与提交代码一致。多轮调用或动态选例方案同时说明构造方式。

报告可围绕问题分析、提示词调优过程和实验心得组织，按整体质量评分。

## 5. 提交与评分

提交`学号_姓名.zip`，包含`report.pdf`和最终的`solution.py`。

**截止时间以elearning公布时刻为准。**

总分15分：报告10分，公开测试集和私有测试集各2.5分。

|得分|公开集准确率门槛|私有集准确率门槛|
|---:|---:|---:|
|1.5分|≥63%|≥69%|
|2.5分|≥69%|≥75%|

每个集合取达到的最高档，低于首档得0分。助教将在统一环境中运行提交代码，评测100条公开题与200条私有题，测试集成绩以助教评测结果为准。
