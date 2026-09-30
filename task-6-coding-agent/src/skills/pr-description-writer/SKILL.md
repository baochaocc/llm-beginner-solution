---
name: pr-description-writer
description: 基于代码改动生成简洁的中文 PR/提交描述（改了什么、为什么、怎么验证）；当任务要求提交、汇总改动或生成汇报文本时使用。
---

# PR Description Writer

根据 repo 的改动（用 `git_diff` 或 `search_files`）生成结构化描述：

- **标题**：一句话概括改动（如 "fix: calculator.add 返回值错误"）。
- **背景**：issue 里描述的问题是什么。
- **改动**：逐文件说明改了哪些逻辑（不要贴大段代码，只讲要点）。
- **验证**：`python -m pytest` 的结果（通过多少、失败多少）。
- **风险**：是否影响其他功能，是否有未覆盖的边界情况。

输出格式：Markdown，控制在 150 字以内，重点突出「为什么这样改」。
