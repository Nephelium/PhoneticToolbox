# 当前 Windows 安装与免安装交付

当前 MFA-R3 完成构建和 Windows 本机最终字节验收。此次范围为系统/架构说明、分发命名及安装门槛；前轮 MFA-R2 新增卸载时可选删除 MFA 的交互与删除行为仍按当时用户要求未专验。

## 成品

| 路径 | 字节数 | 十进制 MB | SHA-256 |
| --- | ---: | ---: | --- |
| `dist/PhoneticToolbox-v3-MFA-20261011-R3/PhoneticToolbox-windows-x64.exe` | 473722527 | 473.72 | `5c7f0688493e7f926505066b959ffd6c7cf2089a17d3ec81c13975f64f5c0468` |
| `output/release-staging/release-mfa-20261011-r3/artifacts/PhoneticToolbox-3.0.0-preview.1-windows-x64-portable.zip` | 473723103 | 473.72 | `731407d42c49e059db13297e0bcd3e67a3119d56983840992ee9b4c162f0a369` |
| `output/release-staging/release-mfa-20261011-r3/artifacts/PhoneticToolbox-3.0.0-preview.1-windows-x64-setup.exe` | 475639106 | 475.64 | `1201fc0a7809340a5f4b7e1baad1925c652f5ecb264e6a8306faaa11830742a2` |

两类 EXE 与更新 ZIP 使用同一应用字节，均小于 500,000,000 字节。源码快照 SHA-256：`3ed67e87dd3c064c40d41a0f8d44ad1682567be00dd681af36238aa930a4a95c`。证据目录：`output/validation/release-mfa-20261011-r3/`。版本仍为 3.0.0-preview.1，本轮目录 MFA-20261011-R3 唯一。

## 系统与名称

- 当前发行支持 Windows 10 1809 及以上、Windows 11 的 x64 系统（Intel/AMD；也称 AMD64、x86_64），暂不支持 x86 32 位、ARM64、macOS 或 Linux。说明书开始使用章节及 README 已说明范围与系统类型查看方法。
- 免安装下载名为 `PhoneticToolbox-windows-x64.exe`。安装器文件名原本已含 `windows-x64-setup`，安装界面新增 Windows x64 标记，`x64os` 限制原生 x64 Windows，`MinVersion=10.0.17763` 对应 Windows 10 1809。
- 当前 Qt 6.11 的平台要求见 [Qt 官方表](https://doc.qt.io/qt-6.11/supported-platforms.html)。架构匹配语义见 [Inno Setup 官方说明](https://jrsoftware.org/ishelp/topic_archidentifiers.htm)。发行支持范围不等于在所有系统版本、所有硬件上完成实测。
- 安装后及更新 ZIP 内保持 `PhoneticToolbox.exe`，保持已有更新协议与快捷方式。`build-info.json.exe` 记录外部分发名，`application.json.entry` 描述固定的安装/更新布局。安装包源文件名由构建参数传入，安装时写入固定名称；ZIP 同样只重命名条目，EXE SHA 不变。
- 安装路径和 AppId 不变。关闭应用后可在原目录覆盖安装，无需先卸载。用户设置、草稿、研究文件及已配置 MFA 不因本次命名修改而更换存储位置。

## 实际验证

- 说明书严格结构/资源摘要、阅读资源生成、前端类型检查、358 项前端测试、UI 数据一致性、生产构建、生产阅读器解析通过。
- 使用最终新名称 EXE 拷贝到工程外独立目录，移除开发环境绑定并使用 Windows-only PATH。科学/托管任务/帮助、TextGrid、直接 GUI、媒体打开及正常退出通过。
- 最终包内容审计、科学与宿主归档/Qt 身份、真实复制回退与损坏修复、中断/并发恢复、设置页缓存清理通过。
- 专用 AppId、OwnedQA 安装/卸载通过，安装后 EXE 与免安装下载 SHA 相同。安装预热和再次启动复用通过，研究文件与设置标记保留。
- 两类真实辅助进程换版及重启通过。免安装来源使用带 windows-x64 名称的 R2 EXE，验证可读更新 ZIP 的固定内部文件名；安装版继续使用固定安装名。源码启动的测试准备器仍按 application.json 检查固定入口，首轮因测试目录把固定名文件移走而失败。重试在夹具中保留同字节固定名副本，外部分发来源仍为新名称；未改产品代码或放宽校验。实际冻结程序从内嵌描述符识别自身，独立单文件启动另已验证。测试旧版本号为 Preview.0 元数据夹具，不代表全部历史版本实测。
- 最终源码快照、前端资产、成品大小/摘要及更新 ZIP 中 EXE 身份一致，结果见 final-identity.json。构建/封装自动清理旧成品，本轮独立 QA scratch 清理回执见 cleanup.json。

## 保留的边界

本轮没有在 Windows 10、32 位 Windows、ARM64 或另一台干净电脑/虚拟机上实测。安装器拒绝规则已编译，拒绝界面未在这些系统上点击。物理音视频设备、完整科研模块矩阵、长期运行未重新验收。

前轮 MFA 卸载删除选项的正常确认交互与实际删除范围未补验；本轮自动卸载为静默方式，会保留 MFA。独立组件 ZIP 及来源包不变，主包仍排除 MFA 环境、声学模型和词典，公网组件下载尚未因本地构建而发布或验证。未操作用户现有安装、MFA 或研究资料，未上传、push 或公开发布。
