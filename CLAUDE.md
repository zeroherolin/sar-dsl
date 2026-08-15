# SAR-DSL 工作约定

## 破坏性操作（必须遵守）

这个工作树长期携带大量未提交改动。以下命令会不可逆地销毁它们，**任何情况下都不要执行**：

- `git checkout <file>` / `git restore <file>` — 回退到 HEAD，未提交的改动无法从
  reflog、stash 或 fsck 恢复
- `git reset --hard`、`git clean -fd`、`git stash drop`

撤销刚做的编辑，用编辑工具改回去；需要对照 HEAD 版本时用 `git show HEAD:<file>`
输出到 `/tmp` 再比较，不要碰工作树里的文件。

这条规则是因为一次真实事故写下的：为撤销一次失败的字符串替换而 `git checkout`
了 `lib/Conversion/SARToLinalg/SARToLinalg.cpp`，丢失了整轮会话的 1024 行改动，
无备份可恢复。

## 提交

除非明确要求，不要提交。

## 验证

改动后跑：`pytest test/python` 与 `cmake --build build --target check-sar-lit`，
两者应全绿。数值改动另需 csim 验证（见 `docs/backends.md`）。
