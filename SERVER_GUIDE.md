# 服务器运行指南：克隆、部署、三组实验与结果下载

适用于本仓库 `feat/vote-experiments` 分支，三组实验指 M1、M2、M3。
当前 `python experiment.py` 默认运行 M1、M2、M3、M3r、M5 共五组，因此下面逐个指定方法。
以下部署步骤根据仓库脚本核对；本机通过自动化测试，尚未在你的服务器执行部署或测量真实准确率。

## 1. 准备服务器并登录

课程脚本支持 Linux x86_64、RTX 4090（24GB）或 RTX 5090（32GB）。镜像需已有 NVIDIA 驱动、Python 3 和 C 编译器。
仓库要求 4090 驱动不低于 550.142，5090 不低于 580.65.06。首次安装至少留出 30 GiB 空间；建议系统盘 100 GB，预装内容较多的镜像可选 200 GB。
环境和模型会从网络下载，服务器需能访问 GitHub、Python 包源和 ModelScope。

在本机 PowerShell 登录，替换用户名、地址和端口；默认 SSH 端口为 22：

```powershell
ssh -p 22 用户名@服务器地址
```

以下命令均在服务器 Bash 执行，直到第 7 节明确返回本机。

```bash
nvidia-smi
python3 --version
command -v gcc || command -v cc
command -v git
command -v tmux
df -h "$HOME"
```

Ubuntu/Debian 若缺少基础工具，且你有管理员权限，可安装：

```bash
sudo apt-get update
sudo apt-get install -y git tmux build-essential python3 ca-certificates
```

若使用 root 账户且没有 sudo，去掉 sudo。NVIDIA 驱动应由云平台镜像或管理员提供。

## 2. 克隆正确分支

```bash
cd "$HOME"
git clone --branch feat/vote-experiments --single-branch https://github.com/StevenPang203426/PromptEngineering_NLP_PJ1.git
cd "$HOME/PromptEngineering_NLP_PJ1"
git branch --show-current
git log -1 --oneline
cd student
bash deploy/setup.sh --check
```

应显示分支 `feat/vote-experiments`，并通过部署前检查。若你的持久化数据盘不在 HOME，可将克隆位置和后续所有路径统一换到数据盘。

如果之前已经克隆，不要重复克隆；确认没有需要保留的未提交修改后更新：

```bash
cd "$HOME/PromptEngineering_NLP_PJ1"
git status --short
git fetch origin feat/vote-experiments
git switch feat/vote-experiments || git switch -c feat/vote-experiments FETCH_HEAD
git pull --ff-only origin feat/vote-experiments
cd student
```

有本地修改或分支分叉时先处理并保留自己的文件，不要用强制覆盖命令。

## 3. 部署并启动模型

使用 tmux 保持模型在 SSH 断开后继续运行：

```bash
tmux new -s nlp-model
cd "$HOME/PromptEngineering_NLP_PJ1/student"
mkdir -p logs
set -o pipefail
bash deploy/start.sh 2>&1 | tee -a logs/model.log
```

脚本自动识别显卡，创建 `.lab-runtime/venv/`，安装固定依赖，下载并校验模型，然后启动 vLLM。
首次安装可能较久，以日志为准；不要额外安装另一套 vLLM 或修改课程锁定版本。
看到 `Application startup complete` 后，按 **Ctrl+B，松开，再按 D**，离开 tmux 但保留模型进程。

恢复模型窗口：`tmux attach -t nlp-model`。实验期间保持模型运行，勿按 Ctrl+C 停掉它。
模型默认监听服务器本机 `127.0.0.1:8000`，实验客户端也在同一服务器运行，无需将模型端口开放到公网。

## 4. 检查客户端并试跑

在服务器外层终端新建另一个 tmux 会话：

```bash
tmux new -s nlp-exp
cd "$HOME/PromptEngineering_NLP_PJ1/student"
source .lab-runtime/venv/bin/activate
mkdir -p logs
set -o pipefail
python check_server.py
python -m unittest discover -p 'test_*.py'
```

`check_server.py` 只检查健康状态和模型名，不生成答案。之后每种方法先运行 3 题：

```bash
python experiment.py --method m1 --limit 3 2>&1 | tee -a logs/m1-smoke.log
python experiment.py --method m2 --limit 3 2>&1 | tee -a logs/m2-smoke.log
python experiment.py --method m3 --limit 3 2>&1 | tee -a logs/m3-smoke.log
```

每组都应完成 3 题、无效返回为 0。若异常，先查对应日志和结果页再跑完整实验。
小批运行会标记为不完整；它不是完整准确率测试，不能与 100 题成绩直接比较。
不要添加 `--mock`，该选项只使用固定回答 A 的模拟模型。

## 5. 运行三组完整公开集实验

仍在 `nlp-exp` 会话、已激活环境的 `student` 目录中执行：

```bash
set -o pipefail
python experiment.py --method m1 2>&1 | tee -a logs/m1-full.log &&
python experiment.py --method m2 2>&1 | tee -a logs/m2-full.log &&
python experiment.py --method m3 2>&1 | tee -a logs/m3-full.log
```

每组 100 题、独立预算 500 次，正常各调用 100 次；三组合计约 300 次，失败重试会增加调用。
代码默认 16 并发，无需另外开三个实验进程。上述 && 会在命令返回非零状态时停止后续命令。
当前入口遇到逐题无效返回仍可能正常结束，所以必须检查各组最终显示的无效题数。

运行时可以按 Ctrl+B、D 离开；重新 SSH 后执行 `tmux attach -t nlp-exp` 恢复。
也可在另一 SSH 窗口查看 `tail -n 50 ~/PromptEngineering_NLP_PJ1/student/logs/m1-full.log`。
当前框架不支持断点续跑；中断后仅重新运行未完成的方法，每次生成新的结果目录。

## 6. 查看结果

完成后，在 `student` 目录汇总：

```bash
python evaluate.py
ls -lt runs/reports/*.html
find runs -maxdepth 1 -type d
```

- `runs/reports/results_*.html`：总排行榜，打开最新文件，可点击各运行记录。
- `runs/时间戳_随机标识_m1/`（或 m2、m3）：该次运行记录。
- `evaluations/evaluation_*.html`：逐题详情，含模型最终回答、思考文本、请求消息和错误。
- `evaluations/evaluation_*.json`：准确率、无效返回、token 用量、调用次数、`length_finishes` 等指标。
- `predictions.jsonl`、`calls.jsonl`：逐题预测与完整调用记录。
- `config.json`、`solution.py`、`summary.json`：参数与代码快照，以及运行耗时等信息。

正式比较时选择三组 **完整 100 题** 记录，核对无效返回、调用数和环境一致性。
`length_finishes` 非零表示存在生成达到长度限制的调用，应结合空回答与错误记录检查。

需要配对分析时，将下列大写占位路径替换为三组完整运行目录：

```bash
python analyze_votes.py runs/完整M1目录 runs/完整M2目录 runs/完整M3目录 --reference m1 > runs/three-method-analysis.md
```

每种方法只选一次完整运行，不要把小批和完整运行同时传入。此命令离线分析，不调用模型。

## 7. 打包并下载到本机

先在服务器打包整个结果目录，保留 HTML 相对链接及所有实验记录：

```bash
cd "$HOME/PromptEngineering_NLP_PJ1/student"
result_archive="$HOME/nlp-results-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$result_archive" runs logs
ls -lh "$result_archive"
printf '%s\n' "$result_archive"
```

记下打印的绝对路径，例如 `/home/ubuntu/nlp-results-20261110-153000.tar.gz`。
切回**本机 Windows PowerShell**（不要在 SSH 会话里执行下载命令），替换服务器信息和实际文件路径：

```powershell
scp -P 22 用户名@服务器地址:/home/ubuntu/nlp-results-20261110-153000.tar.gz "$env:USERPROFILE\Downloads\"
```

SSH 用小写 `-p`，SCP 用大写 `-P`。如果登录使用密钥，两条命令都可增加 `-i "本机私钥路径"`。
也可以在云平台文件管理器中下载这个 `.tar.gz` 文件。

使用 7-Zip 解压，或在 PowerShell 执行（文件名换成实际下载的名称）：

```powershell
$resultDir = Join-Path $env:USERPROFILE 'Downloads\nlp-results'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
tar -xzf "$env:USERPROFILE\Downloads\nlp-results-20261110-153000.tar.gz" -C $resultDir
Get-ChildItem "$resultDir\runs\reports\*.html" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Invoke-Item
```

浏览器打开排行榜后可点击查看逐题结果。保留解压后的 `runs/` 目录结构，不要只拷贝排行榜 HTML。
截图建议保留三组完整结果的排行榜、调用预算，以及用于报告分析的代表性题目。

## 8. 常见情况与后续

### 4090 旧驱动的实验性试运行

若唯一阻塞是驱动 `550.54.14` 低于课程门槛 `550.142`，可以显式试运行：

```bash
git pull --ff-only origin feat/vote-experiments
bash deploy/setup.sh --check --allow-older-driver
bash deploy/start.sh --allow-older-driver
```

此开关仅允许实际 4090、CUDA 12.x 配置、驱动至少 525.60.13 的情况越过课程驱动门槛；磁盘、编译器和环境版本检查仍执行。
安装固定依赖后、下载模型前会实际执行 BF16 矩阵乘法、逐元素运算、归约和 CUDA 同步，失败立即停止。
`.lab-runtime/driver-check.json` 记录实际驱动、是否放行及基础运算检查状态。
基础运算通过也不等于 vLLM 或全部模型内核兼容。服务启动后仍需 `python check_server.py` 和三题真实试跑。

依据 [NVIDIA CUDA 小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)，CUDA 12.x 可以在满足最低要求的旧驱动上运行部分应用，但 PTX JIT 与需要新驱动的功能存在限制。
因此课程脚本门槛不等同于已经测出的实际运行下限；本开关用于验证，不能保证成功，也不改变课程模型或依赖版本。
若日志出现 `unsupported PTX`、`unsupported toolchain`、`requires newer driver` 等错误，应保留完整日志，依据实际错误处理或升级宿主机驱动。

- 健康检查连接失败：恢复 `nlp-model` 会话看日志，确认服务就绪且模型进程未退出。
- 驱动或磁盘检查失败：按部署检查给出的实际原因调整镜像或空间，然后重新检查。
- 模型下载中断：再次执行 `bash deploy/start.sh`；脚本会检查环境与模型，不要先删除整个运行环境。
- 8000 端口被占用：先确认是否已有课程模型。确需改端口时，模型启动前设 `export COURSE_PORT=8001`，实验终端设 `export COURSE_API_BASE=http://127.0.0.1:8001/v1`。
- tmux 会话已存在：使用 `tmux attach -t nlp-model` 或 `tmux attach -t nlp-exp`，不要重复创建同名会话。
- 运行返回无效：查看对应题目的调用、最终 content、reasoning 和 length 标记；重跑会消耗新的调用。
- 三组基础实验完成后，如需随机选例对照和投票，可分别运行 `python experiment.py --method m3r`、`python experiment.py --method m5`。这是额外实验，不包含在上述三组中。
- 下载核对完成后，确认实验已结束，可在模型 tmux 窗口 Ctrl+C 停止服务，再按云平台流程关闭 GPU 实例。

根据真实结果选择最终 `solution.py`，报告写入真实成绩、截图与提示词附录；当前操作步骤本身不代表任何方法达到评分门槛。
