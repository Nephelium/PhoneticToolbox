# 桌面运行时清单与跨平台验收

2026-10-01，P16 提供清单解析和预检，不代表完整发行包已完成。Windows 当前源码验证与将来的 macOS 本机打包使用同一份代码，各平台原生运行时分别验收。

## 清单

`desktop-bundle.json` 位于应用资源根目录，示例只展示格式，SHA-256 必须来自实际文件：

```json
{
  "schema": "desktop-bundle/1",
  "portable": true,
  "platform": "win32",
  "architecture": "x86_64",
  "runtimes": {
    "PTB_EGG_PYTHON": {
      "path": "runtimes/egg/python.exe",
      "sha256": "<actual SHA-256>"
    },
    "PTB_M05_PYTHON": {
      "path": "runtimes/lip/python.exe",
      "sha256": "<actual SHA-256>"
    }
  }
}
```

运行时路径使用包内相对路径，不接受绝对开发机路径、越界或符号链接。平台和架构必须匹配当前主机，运行时文件需符合清单哈希。现有 `local-preview.json` 仅允许 `portable:false`，继续作为明确依赖开发环境的本机预览。MFA 独立可选组件仍按其组件协议安装，不冒充已嵌入主包。

只读检查：`python scripts/check_desktop_bundle.py <bundle-directory>`。通过只证明这些运行时引用成立，不检查整套依赖文件、不执行科学回归、不验证签名；输出始终保留 `release_ready:false`。打包器尚未自动生成此清单，需在真正封装对应运行时后生成。现有 EXE 不会因源码更改而自动更新。

## 原生声道资源

VTL 加载器按 `platform-architecture` 子目录选择 DLL/dylib/so，例如 `win32-x86_64`、`darwin-arm64`、`linux-x86_64`，并拒绝分析/合成两份原生库内容不一致。只有 Windows x86_64 可兼容已有平铺 DLL。目录选择测试不等于这些平台原生库已经存在或可用。

## 尚需目标设备验收

- Linux：本轮新 POSIX 保存逻辑需实测；科学任务仍依赖既有运行时哈希、资源档与准入证据。本轮未操作实际服务器。
- macOS：原生科学执行适配、VTL/REAPER/MFA 等依赖、Qt WebEngine helper、权限与设备、签名/公证、完整离线包均待实际 Mac。当前不启用未验证的科学进程执行路径。
- Windows：本轮验证源码和前端构建，未重打 EXE，未证明当前预览包可在无开发环境的另一台电脑完整运行。

拿到 Mac 后可以复用本轮修正的路径、资源选择、Command 快捷键和保存适配，仍须完成上述目标平台验收，不能预先保证直接打包即通过。
