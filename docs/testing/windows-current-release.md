# 当前 Windows 安装与免安装交付

当前为 **3.0.4**，服务器和GitHub公开发布，stable/preview更新频道均为3.0.4。最终成品、安装卸载、科学任务、默认GPU窗口和启动对照均通过，范围见[完整发行验收](release-3.0.4-publish.md)。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `PhoneticToolbox-windows-x64.exe` | 496,270,906 | `f2818b137d20f6f6f9540ff1066e1d326d0cfbeccd50c130e312bd45aacd6af7` |
| `PhoneticToolbox-3.0.4-windows-x64-portable.zip` | 496,271,472 | `ca4b2610c5630caf0c3cd6a9be0541ed0035c6c43e720570f3f7495afc244699` |
| `PhoneticToolbox-3.0.4-windows-x64-setup.exe` | 498,186,108 | `7934caf23dc3fa2cb6f1168600940002d05cbfd7b44d13944dff3b35ca32e1c2` |

免安装版在`dist/PhoneticToolbox-304-20261019-042107-9c1e34`，安装及更新包在`output/release-staging/release-304-20261019-042107/artifacts`。三种分发均小于500,000,000字节，使用同一最终应用。免安装首次启动自行选择缓存位置，替换EXE后继续使用；安装版由安装器准备缓存。设置、草稿和研究数据保留。

Windows 10 1809及以上和Windows 11 x64为产品支持范围。本次在本机工程外独立状态和Windows-only PATH下验证，不代表第二台干净电脑、Windows 10实机、实体设备或所有历史版本升级均已验证。MFA环境/模型/词典不随主包提供。

[GitHub Release](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.4) · [服务器安装包](https://www.phonetictoolbox.com/releases/windows-x64/3.0.4/PhoneticToolbox-3.0.4-windows-x64-setup.exe)
