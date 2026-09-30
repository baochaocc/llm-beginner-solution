---
name: test-runner
description: 运行 pytest 并解读失败输出、定位失败测试与原因、给出修改方向；当需要验证代码正确性、看到测试失败或改完代码后必须使用。
---

# Test Runner Workflow

1. **运行**：调用 `run_tests`（默认整个 repo；也可指定 `test_path` 缩小范围）。
2. **读输出**：看 `FAILED` 行与断言详情：
   - `assert add(2, 3) == 5` 失败 → 通常是函数返回逻辑错误；
   - `ModuleNotFoundError` / `ImportError` → 模块名或导入路径问题；
   - `E TypeError` → 参数数量/类型不匹配。
3. **定位根因**：每个失败用例读一遍实现，找出函数里与预期不一致的一行。
4. **改完复测**：修改代码后**必须**再跑一次 `run_tests`，确认全绿（`passed` / `no tests ran` 都不是成功）。
5. **全绿标准**：输出出现类似 `3 passed in ...`，即停止改代码、准备收尾。
