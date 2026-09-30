# LLM-Beginner 全系列复现指南（Trae 版）

> 本机验证基线：Python 3.14 + torch 2.11.0+cu128（RTX 5070 Ti Laptop, Blackwell sm_120, **12GB 显存**）
> 全局解释器：`C:\Users\32362\AppData\Local\Programs\Python\Python314\python.exe`
> 项目位置：`C:\Users\32362\Documents\llm-beginner-master`

## 0. 通用环境（所有任务共用）

1. **Trae 打开仓库根目录** `llm-beginner-master`（不是某个 task 子目录）。
   原因：`eval/run.py` 依赖仓库根的 `_eval_harness.py`；只开子目录自检会 import 失败。
2. **选解释器**：`Ctrl+Shift+P` → `Python: 选择解释器` → 上面的 Python314 路径。
3. **HF 加速**（下载模型/数据慢时，在终端先执行）：
   ```powershell
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   ```
4. **已装好的依赖**（本机现状，无需重装）：
   torch 2.11.0+cu128 / transformers 5.2.0 / datasets 5.0.1 / tokenizers 0.22.2 /
   sentence-transformers 5.2.3 / huggingface_hub 1.5.0 / openai 2.24.0 / gitpython 3.1.50 /
   numpy / pandas / pyarrow / matplotlib / tqdm / pydantic / requests / pyyaml / scikit-learn
5. **一次补齐缺的**（下面各任务也用得上）：
   ```powershell
   pip install accelerate pytest pypdf faiss-cpu wikipedia-api mcp
   ```
   - `faiss-cpu` 万一没有 Python 3.14 的 wheel：任务四的 KB 只有几千 chunk，
     用 numpy 暴力内积检索（毫秒级）完全够用，不依赖 faiss 也能跑。
   - `datasets` 是 5.x：脚本型数据集（load 时出现 "Dataset scripts are no longer
     supported"）已废弃。后面任务里 TinyStories / SWE-bench / DPO-En-Zh 都是普通
     parquet 数据集，`load_dataset` 没问题；**MOSS 数据**按 README 下 jsonl.zip 自己
     读 JSON，别用 load_dataset。
   - huggingface_hub 1.5 的 CLI 是 `hf download`（旧名 `huggingface-cli` 已更名）。
6. **7B 模型的部署**（任务 4/5/6 都要，12GB 显存放不下 FP16 的 7B 权重 ≈15GB，
   请统一用 **Ollama + Q4_K_M 量化** ≈5GB）：
   - 安装：[https://ollama.com/download](https://ollama.com/download)（Windows 安装包）
   - 拉模型 + 起服务：
     ```powershell
     ollama pull qwen2.5:7b-instruct        # 任务 4/5 用（约 4.7GB）
     ollama pull qwen2.5-coder:7b-instruct  # 任务 6 用（约 4.7GB）
     ollama serve                           # OpenAI 兼容端点 http://localhost:11434/v1
     ```
   - 客户端配置：`OPENAI_BASE_URL=http://localhost:11434/v1`、`OPENAI_API_KEY=ollama`（任意值）。
   - 不想装 Ollama 也可用 vLLM AWQ / llama.cpp GGUF（各任务 `data/download.py` 末尾会打印命令）。

---

## 1. 任务一：Transformer 分类器 — ✅ 已完成

```powershell
cd task-1-transformer
python eval/run.py          # 三项全过（attention_correctness / causal_mask / classifier_accuracy 0.8567）
python train.py             # 重训（约 1 分钟）
python make_figures.py      # 3 张注意力热图
python ablation.py          # S1/S2 消融（7 组，约 3 分钟）→ results/ablation_summary.csv
```

---

## 2. 任务二：从零实现 mini-GPT — ✅ 已完成（本机复现结果见下）

**目标**：手写 BPE 分词 + RoPE + KV cache + 四种采样，decoder-only 预训练，dev 困惑度达标。
**关联**：延续任务一的 causal mask（toy_lm.py 已预热）。
**本机最终结果**：dev ppl **32.32**（阈值 50）；参数量 4.91M，词表 352。

```powershell
cd task-2-mini-gpt
python data/download.py                         # 唐诗 quick-start（离线，无需下载）
# 或 python data/download.py --dataset tinystories   # 英文故事，~100MB（建议 GPU）
# 或 python data/download.py --dataset skypile       # 中文大语料（需手动抽样，见脚本提示）

# 训练（最终配置，约 1 分钟）：
python train.py --vocab_size 352 --d_model 256 --n_heads 8 --n_layers 6 `
  --d_ff 1024 --epochs 300 --dropout 0.25 --batch 64 --block 128
python demo_generate.py                        # 四种采样策略的生成样例 -> results/generation.md
# 自检（在仓库根目录跑！）：
cd ..\..
python task-2-mini-gpt/eval/run.py
```

> ⚠️ 关键发现（消融结论，详情见 `task-2-mini-gpt/EXPERIMENT_NOTES.md`）：
> **极小语料上词表规模是最大杠杆**——固定模型只改词表：v384→39.4 / v512→62.3 /
> v768→141.8 / v1024→241 / char-level→2194。模型容量（0.9M~14.6M）则几乎无影响。
> 训练参数可用 `python train.py --help` 查看（--vocab_size/--level/--d_model/...）。

自检（3 项）：`tokenizer_roundtrip`（BPE 还原中文）/ `kv_cache_equivalence`
（开 cache 与全量 logits 误差 <1e-4）/ `perplexity_on_dev`（唐诗 <50、TinyStories <10）。

**要手写的实现**（`src/` 下）：
| 文件 | 内容 |
|---|---|
| `src/tokenizer.py` | `BPETokenizer`：merge 统计、`encode/decode/vocab_size/from_pretrained` |
| `src/rope.py` | 旋转位置编码（只加 Q/K，**不加 V**） |
| `src/attention.py` | causal 多头注意力 + **KV cache 更新** |
| `src/model.py` | `MiniGPT.forward(ids, kv_cache=None, return_cache=False)`、`generate(...)`、`block_size` 属性；`load_for_eval` |
| `src/sampling.py` | greedy / top-k / top-p / temperature |
| `train.py` | next-token 预训练，输出 `ckpt/best.pt` + `ckpt/tokenizer.json` |

**坑位提醒**：
- RoPE 维度配对（相邻两维一组 vs 前后折半）的实现在模型和 KV cache 里要一致；
- 增量解码时新词元 position 要接历史长度（否则 `kv_cache_equivalence` 挂）；
- top-p 截断后要**重新归一化**；temperature=0 退化为 greedy，别除零；
- 唐诗太小容易过拟合（困惑度先降后升），留早停。

---

## 3. 任务三：SFT + DPO（Qwen2.5-0.5B）— ✅ 已完成

**目标**：手写 LoRA，在 0.5B 基座上 SFT（MOSS 数据）+ DPO（偏好数据）。
**本机最终结果**：`lora_param_count` 0.109% ✅ / `loss_masking` 0.422 ✅ / `sft_vs_base` ✅；
SFT loss 1.32→1.196；DPO win_rate 0.36→**0.89**、margin 5.53。

> ⚠️ 本机网络绕行方案（HF 直连/AWS CDN 均被墙）：
> - 一切模型/数据从 **ModelScope** 下载：`pip install modelscope` 后
>   `from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='models/Qwen2.5-0.5B')`
> - SFT 数据：ModelScope `mxnaxvex/moss-003-sft-data-10w`（MOSS 10 万子集中 1 万条 jsonl）
> - DPO 数据：ModelScope `okwinds/Human-Like-DPO-Dataset`（prompt/chosen/rejected，21768 对）
> - HF 上的 `OpenMOSS-Team/moss-003-sft-data` / `hiyouga/DPO-En-Zh-20k`
>   在 ModelScope 是空壳（只有 README），别用它。

```powershell
cd task-3-sft-dpo
# 训练（顺序不能反：先 SFT 再 DPO）
python train_sft.py --samples 8000         # ~15 分钟 -> ckpt/sft/
python train_dpo.py --pairs 1500 --lr 1e-5 # ~3 分钟 -> ckpt/dpo/
python src/compare.py                      # base/SFT/DPO 对比 -> results/compare.md
# 自检（仓库根目录跑）：
cd ..
python task-3-sft-dpo/eval/run.py
```

**坑位提醒**（详见 `task-3-sft-dpo/EXPERIMENT_NOTES.md`）：
- LoRA 注入后**冻结整个基座**（不是只冻被包裹的 Linear）；
- **SFT/DPO 的 loss 都要 next-token 错位**（logits[t] ↔ labels[t+1]），
  否则模型学"回读当前 token"，loss 虚低、生成烂掉；
- assistant 头部（`<|im_start|>assistant\n`）也要进训练目标，否则生成退化为换行循环；
- **DPO lr 别用 1e-4**：会 margin 爆炸 + 输出坍塌成"！！！"；用 1e-5；
- bf16 算 loss（别 `.float()` 转 15 万类的 logits，会 OOM/拖慢几十倍）。

---

## 4. 任务四：RAG 文档问答（NNDL PDF）— ✅ 已完成

**目标**：PDF chunking → BGE + FAISS →（reranker）→ Qwen 生成；Recall@10 > 0.6。
**本机最终结果**：`chunking_sanity` ✅ / `nndl_gold_recall_at_10` ✅
（**Recall@1 0.50 / @3 0.733 / @5 0.80 / @10 0.867、MRR 0.629**）/ `rag_end_to_end` ✅。
生成器：Qwen2.5-0.5B-Instruct（本地 transformers 加载；Ollama 装了会自动用 7B）。

```powershell
# 依赖 + 数据（PDF 8MB 从 GitHub 直连；BGE 模型从 ModelScope）
pip install pypdf faiss-cpu modelscope
python task-4-rag/data/download_pdf.py
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='task-4-rag/models/bge-small-zh-v1.5')"

# 建索引（首次）-> 自检（仓库根目录）
python task-4-rag/data/build_index_run.py
python task-4-rag/eval/run.py
```

**实现要点**（`src/`）：`chunker.py`（固定尺寸+重叠；页级单元 = 最优）、
`indexer.py`（PDF→页→chunk→BGE→FAISS）、`retriever.py`（BGE query 前缀 + 内积检索）、
`reranker.py`（bge-reranker 精排，加分项）、`generator.py`（OpenAI 兼容服务 →
回退本地 transformes 模型）、`rag.py`（`answer(query) -> {answer, sources}`）。

**坑位提醒**（详见 `task-4-rag/EXPERIMENT_NOTES.md`）：
- 本机 **sentence-transformers 5.2.3 批量编码会原生段错误**（0xC0000005），
  用 `src/embed.py` 的 transformers 手动 mean-pooling + L2 normalize；
- BGE query 侧必须加前缀「为这个句子生成表示以用于检索相关文章：」；
- **chunk 策略是最优搜索点**：page-aware（页≤1200 整页，超页 768/128 细分）最优；
- ModelScope 的 reranker 会连带下载 onnx/bin（3.3GB），ignore_file_pattern 无效，
  只能等或手动删。

---

## 5. 任务五：工具调用 Agent（ReAct）— ✅ 已完成（本机复现结果见下）

**目标**：手写约 200 行 ReAct 循环，4 类工具
（calculator / python_sandbox / file_search / wiki），10 题任务集关键词命中率 >60%。
**本机最终结果**：`tools_individual` ✅（wiki 网络受限按跳过）/
`multi_tool_success_rate` ✅ **1.0（10/10）** / `error_recovery` 跳过（可选）。
模型：**Qwen2.5-1.5B-Instruct**（本地 transformers bf16/CUDA；ModelScope 下载，
不依赖 Ollama）。

> ⚠️ 本机 Ollama 安装包（GitHub CDN）下载被墙，改用本地 transformers 推理：
> `python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='task-5-tool-agent/models')"`
> Agent 自动选择 `models/` 下的模型；`LOCAL_MODEL_PATH` 可覆盖；以后若装好
> Ollama/vLLM，设 `LLM_BASE_URL` 即切回 OpenAI 兼容 HTTP 端点（无需改代码）。

```powershell
# 数据（已生成则跳过）
python task-5-tool-agent/data/download.py
# 自检（在仓库根目录跑！）
python task-5-tool-agent/eval/run.py
# 可选：10 题逐题打印
python task-5-tool-agent/data/run_all_tasks.py
```

自检（3 项）：`tools_individual`（4 工具单测；wiki 依赖网络，离线自动跳过）/
`multi_tool_success_rate`（>60%，按答案关键词校验，**不信 agent 自报 success**）/
`error_recovery`（可选加分，需 `inject_error` 钩子）。

**要手写的实现**：`src/tools/{calculator,python_sandbox,file_search,wiki}.py`
各导出 `TOOL_SCHEMA` + `run(args)`（参数键固定：`{"expression"}` / `{"code"}` /
`{"pattern","dir"}` / `{"query"}`）；`src/agent.py` 的 `ReActAgent.run(task) -> {steps, final_answer, success}`。

**坑位**（详见 `task-5-tool-agent/EXPERIMENT_NOTES.md`）：calculator 别直接 eval
（任意代码执行），限算术白名单；python_sandbox 黑白名单只是教学级防护
（`__globals__` 都能逃）；file_search 先 resolve 防 `..` 越界，**相对路径要锚定到
任务根**（官方 eval 从仓库根运行，按 CWD 解析会找不到 data/agent-fixtures）；
工具抛异常要 catch 后把错误塞回 Observation（M3）；设步数上限防死循环；
第 10 题要真读文件内容返回片段（只匹配文件名不行）；Qwen 原生 `<tool_call>`
解码要用 `skip_special_tokens=False`，还要兼容模型省略标签输出裸 JSON 以及
**缺一个右括号的不平衡 JSON**；模型报错后别在最终答案贴代码块（Agent 层拦截 +
system 规则），要求直接给中文结论（含 True/False、年份、四舍五入后的完整小数）。

---

## 6. 任务六：Mini Coding Agent — ✅ 已完成（本机复现结果见下）

**目标**：MCP server（≥5 工具）+ Skill 加载器（2-3 个 Skill）+ 1-2 个 Subagent +
agent loop，在 `data/toy-repo` 上修好 `calculator.add` 并让 pytest 全绿。
**本机最终结果**：`mcp_server_lists_tools` ✅（7 工具）/ `skill_loader_metadata` ✅
（3 Skills）/ `toy_repo_patch` ✅（3 passed，trace 含 patch）/ `swebench_lite_sample` 跳过（可选 S4）。
模型：**Qwen2.5-Coder-3B-Instruct**（本地 transformers bf16/CUDA；ModelScope 下载，
不依赖 Ollama）。

> ⚠️ 偏差记录：README 默认 Qwen2.5-Coder-7B-Instruct，但 12GB 显存放不下 FP16 7B
> （≈15GB）；Q4_K_M 量化需 llama.cpp 二进制（GitHub 被墙装不了）→ 用 3B Coder 替代。
> toy-repo 上 **6 步一次跑通**（读测试 → 读实现 → 跑测试确认失败 → 重写 calculator.py
> （add 改 a+b、保留 factorial）→ 复测 3 passed → final），全程 16-17s。
> 模型下载：`python -c "from modelscope import snapshot_download;
> snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', local_dir='task-6-coding-agent/models/Qwen2.5-Coder-3B-Instruct')"`

```powershell
# 数据（已生成则跳过；git init 失败是预期，自检不依赖 git）
python task-6-coding-agent/data/download.py
# 自检（在仓库根目录跑！）
python task-6-coding-agent/eval/run.py
# 可选：单次 agent 轨迹 / MCP stdio 握手验证
python task-6-coding-agent/data/run_agent.py
python task-6-coding-agent/data/test_mcp_stdio.py
```

自检（4 项）：`mcp_server_lists_tools`（≥5 工具）/ `skill_loader_metadata`（每 skill
有 name+description）/ `toy_repo_patch`（修好 add、pytest 全绿、trace 含 patch）/
`swebench_lite_sample`（可选加分，本机 GitHub 被墙无法 clone repo，跳过）。

**要手写的实现**：`src/mcp_server.py`（`list_tools()` 顶层导出 + 可独立 stdio 启动；
read_file/write_file/run_tests/git_diff/git_apply，**路径 resolve 防越界、subprocess 用
list 形式防 shell 注入**）、`src/skill_loader.py`（≈50 行：扫描 `src/skills/*/SKILL.md`
YAML front-matter，description 匹配 + 按需加载正文）、`src/skills/`（code-review /
pr-description-writer / test-runner）、`src/subagents/`（隔离 context + 独立步数上限）、
`src/agent.py`（`CodingAgent.run(repo_path, issue) -> Trace`，
Trace 含 `steps/patch/tests_passed`，自检 `trace.get(...)` 取值）。

**坑位**（详见 `task-6-coding-agent/EXPERIMENT_NOTES.md`）：自检每次从
`calculator.py.orig` 恢复 buggy 版，别指望上一轮已修；停机要有明确 done 信号
（测试通过/模型声明），别硬停；上下文爆了做 compaction；subagent 要真隔离
（独立 message 列表）；mcp 2.x 的 FastMCP 已改名 MCPServer；小模型爱输出多个 JSON
对象 / JSON 字符串裸换行 / content 含花括号错位，解析器都要兼容；模型"修测试而非
修实现"的锚定错误要提示 + Agent 层守卫双保险。
内置 git 不可用没问题：`download.py` 的 git init 失败是预期行为（本机未装 git），
自检基于 `calculator.py.orig` 快照，不依赖 git。

---

## 7. 推荐顺序与时间预估（本机）

| 顺序 | 任务 | 主要外部动作 | 预估工时 |
|---|---|---|---|
| 1 | 任务二 mini-GPT | 无（唐诗离线） | 1 周（唐诗跑通即可过长线，TinyStories 视兴趣） |
| 2 | 任务三 SFT/DPO | 下 Qwen0.5B（1GB）+ MOSS 数据 | 1-2 周 |
| 3 | 任务四 RAG | 下 BGE（1.2GB）+ **装 Ollama 拉 7B Q4** | 1-2 周 |
| 4 | 任务五 Tool Agent | 复用 Ollama 7B | 1-2 周 |
| 5 | 任务六 Coding Agent | 再拉 coder-7B（4.7GB） | 3-4 周 |

> 共同外部依赖只有一项：**Ollama**（+ 两次 `ollama pull`）。装好它，任务 4/5/6
> 就只剩写代码。所有任务的提交物都是 `eval/result.json` + 若干图/样例 + 200-500 字
> 实验观察，到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)
> 发帖即可。
