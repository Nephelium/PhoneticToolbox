# P03 独立基线

`golden/*.json.gz` 是原 v2/phonetic_311 对公开合成输入的冻结结果，不是 v3 输出，也不是所有指标的科学真值。`manifest.json` 给出来源、hash、音频头、声道映射和覆盖范围；`parameter-contract.json` 给出各参数单位与重复比较规则。

测试不会更新 expected。真实语料与其数值轨迹只在忽略的本机 output 下，不随本目录提交。详细说明见 [捕获协议](../../docs/baseline/capture-protocol.md) 与 [验证报告](../../docs/testing/p03-baseline-report.md)。
