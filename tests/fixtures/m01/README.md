# M01-A 独立旧行为基准

全部输入为本项目合成配方。28例各两次独立旧v2进程；本目录只保存每例第一轮的完整科学字段和来源哈希，第二轮比较证据在manifest与本机忽略目录。清单指明 `v3_algorithm_parity=not_implemented`，不得将本基准解释为新算法正确性。

- WAV复用 `scripts/baseline_support.py` 的 `SYN-VOWEL-44100` / `SYN-EMPTY` 配方。
- 设置、选择和故障案例由 `scripts/m01_baseline_support.py` 定义。
- 合成TextGrid与PKL仅在新测试目录生成；PKL由测试自身构造，不接受外部任意文件。
- `m01_baseline_worker.py`只在原解释器中读取v2；实际模块hash、依赖版本、recorder hash随每例保存。
- 时间、mask、shape、配置等精确比较，有限值沿用P03容差。JSON不保存非法NaN，使用values与nonfinite两字段。
- 表格公式样文本与空结果后半导出失败是**预期记录的旧问题**，不是通过降低检查获得成功。

最终索引复用了27个不变案例，并单独重跑修正后的REAPER故障注入，详见 [验收报告](../../../docs/testing/m01-baseline-report.md)。早27例recorder哈希 `099f369c3ec8e490f51e24c3a4c5a7d0e1732ca387c0c2d33b7c27f8543b2039` 可由当前worker将下列故障注入一行逆向替换后复原，已逐字节哈希核对；该分支不作用于这27例。最终REAPER案例使用当前修正代码，并有实际退出1与Python回退结果。

```python
# 早版（带完整参数时追加此开关未触发失败，因此该故障案例没有入最终基准）
command.append('--m01-invalid-option') # real native exit 1, then original Python fallback
# 当前（真实二进制返回1，然后由原v2调用Python回退）
command=[command[0],'--m01-invalid-option'] # real native exit 1, then original Python fallback
```

日常测试只读取本目录。新捕获写入 `output/validation/m01/`；首次冻结须显式 `--freeze-from`，已有本目录时拒绝覆盖。改变旧行为、配方或算法版本须另建明确版本，不能自动更新本目录来“修复”parity失败。
