# 原生更新子系统交付

2026-10-05。状态：verified，限定此处列明的原生服务、Qt 信号与独立界面验证。产品宿主接线、实际安装/免安装换版和服务器发布由主线继续验收。

## 文件与边界

- `desktop/src/ptb_desktop/updates.py`：SemVer、双来源检查、地区选择、用户目录设置和流式校验下载。
- `desktop/src/ptb_desktop/updates_bridge.py`：独立 QWebChannel 对象。
- `frontend/src/platform/updates.ts`：本机能力客户端，无浏览器联网降级。
- `frontend/src/components/UpdatePanel.vue`：手动检查、来源/通道设置、两来源状态、确认下载、进度、再次确认退出并更新。
- `frontend/src/components/UpdateNotice.vue`：启动提示。
- 定向测试 `desktop/tests/test_updates{,_bridge}.py`、`frontend/tests/updates.test.ts`、实际窗口 `frontend/tests/updates-browser.mjs`。

没有修改 host、AppShell、research_entry、网络拦截策略或科学模块。作者工具另见 `tools/manual-studio/VALIDATION.md` 与 `README.md`。没有安装新依赖、使用服务器凭据、推送、公开发布或删除已有文件。

## 宿主接线

```python
from ptb_desktop.updates_bridge import UpdatesBridge

# Qt 宿主已有的 QWebChannel；package_kind 必须来自发行物安装方式。
self.updates_bridge = UpdatesBridge(
    self,
    current_version='3.0.0-preview.1',
    package_kind='portable',  # 安装版用 installer
    # 默认 user_data_root()/updates；主线如另有缓存约定可显式指定 data_root。
    apply_handler=apply_verified_package,
)
self.channel.registerObject('updates', self.updates_bridge)
# 关闭窗口时先 self.updates_bridge.close()。
```

`apply_verified_package(path: Path, kind: str)` 返回 `{started: bool, message?: str}`。回调运行于更新服务拥有的工作线程，需要退出 Qt 窗口时，应发送 queued signal 回 GUI 线程。回调前已再次检查缓存位置、大小和 SHA-256。此子系统不解压、不启动安装器，不决定换版时的文件替换或清理。

前端握手收到 channel 后调用：

```ts
import {installUpdatesChannel} from '../platform/updates';
installUpdatesChannel(channel.objects.updates);
// 原生连接关闭时 installUpdatesChannel(undefined)。
```

启动组件和更新页面可采用：

```vue
<UpdateNotice @checked="lastUpdateResult = $event" @open="openUpdatesPage()" />
<UpdatePanel :initial-result="lastUpdateResult" @checked="lastUpdateResult = $event" />
```

`UpdateNotice` 可选 `enabled`，默认 true。事件 `checked(UpdateCheck)`、`open(UpdateRelease)`。页面 `initialResult?: UpdateCheck`，事件 `checked(UpdateCheck)`、`downloaded(VerifiedDownload)`、`applied()`。

客户端 API：

```ts
await updater.preferences();
await updater.configure({source:'auto',channel:'preview',autoCheck:true});
await updater.check({manual:true}); // 第二参数可传 AbortSignal
await updater.acknowledge(releaseId); // 仅在提示实际显示后记当天抑制
await updater.download(releaseId,'portable',true,{signal,progress});
await updater.apply(downloadId,true);
```

原生协议：`request(requestId, JSON.stringify({operation,args}))`，`cancel(requestId)`。`ready(requestId,json)` 返回 `{ok:true,value}` 或 `{ok:false,error:{code,message}}`。`progress(requestId,json)` 返回 `{phase,source,received,total,message?}`。能力请求没有任意 URL、进程或磁盘路径字段。前端下载结果仅有不透明 downloadId、文件名、大小、哈希、来源、类型与 verified，不暴露本机绝对路径。

## 发布清单

服务器 preview 地址：`https://www.phonetictoolbox.com/releases/windows-x64/preview/latest.json`，stable 采用相同目录规则。单版本清单为：

```json
{
  "schemaVersion": "ptb-release/1",
  "version": "3.0.0-preview.1",
  "channel": "preview",
  "notes": "版本说明",
  "publishedAt": "2026-10-05T00:00:00Z",
  "packages": [
    {
      "kind": "portable",
      "platform": "windows-x64",
      "name": "PhoneticToolbox-windows-x64-portable.zip",
      "url": "https://www.phonetictoolbox.com/releases/windows-x64/preview/PhoneticToolbox-windows-x64-portable.zip",
      "size": 12345,
      "sha256": "完整64位十六进制SHA256"
    }
  ]
}
```

示例 size/sha256 为占位，发布必须替换成真实文件值。安装包另加 `kind:installer`、`.exe` 名称和对应值。也支持 `ptb-release-index/1` 的 `releases` 数组。所有包均为 windows-x64。

GitHub 使用 `Nephelium/PhoneticToolbox/releases` 列表，最多检查 1000 条，截断返回检查不完整。优先识别发布资产 `ptb-release.json`，其清单结构相同，包地址使用该仓库的 release download 地址。无清单时仅接受明确 windows-x64/win-x64/win64 名称、portable.zip 或 installer/setup.exe，以及 GitHub 资产 `digest:sha256:...`。版本标签与清单必须一致。没有校验信息的已有发布不会被误报为已是最新版本。

## 行为

启动自动检查默认间隔 6 小时，跨重启保留；同版本提示 24 小时内一次，仅实际展示时登记。手动检查强制，关闭启动检查不妨碍手动检查。Preview 包含有效预发布和正式版本，稳定通道排除预发布。最高版本按 SemVer 比较，历史 `3.0.0a1` 映射 `3.0.0-alpha.1`，build metadata 不影响优先级。

自动来源通过原生 HTTPS Cloudflare trace 短超时仅读取 `loc`。正文/IP 不写日志、状态或报告。失败时使用 Windows 国家代码线索，并在界面明确来源。CN 优先国内服务器，其他网络地区优先 GitHub；可手动覆盖。检查仍分别报告两来源结果，一个来源失败不阻止另一来源已发现的新版本。没有新版本且任一来源未知时显示检查未完整。

下载流式校验完整大小与 SHA-256，未通过内容只保留为 .failed；取消和损坏均不产生可应用记录。同版本跨来源只在包类型、大小与哈希完全相同的情况下兜底。换版回调前重验缓存，拒绝被改动的包。用户设置位于 v3 用户数据目录的 updates/state.json，独立于安装/免安装程序目录。没有自动清理，本轮缓存周期清理由主线接入。

## 验证与证据

- `.venv/v3-dev/Scripts/python.exe -c "import sys; sys.path.insert(0,'desktop/src'); import pytest; raise SystemExit(pytest.main(['desktop/tests/test_updates.py','desktop/tests/test_updates_bridge.py','-q','-o','addopts=']))"`：65/65。定向任务未运行历史 phonetic_toolbox coverage 配置。
- `node --test frontend/tests/updates.test.ts`：6/6。
- `npm --prefix frontend run typecheck`：通过。
- `npm --prefix frontend test`：325/325，当时整库源码检查点。
- `node frontend/tests/updates-browser.mjs`：10/10 实际最大化 Chrome 组，窗口在采集前通过 CDP 确认 maximized。浅深截图已回看。报告 `output/updates-ui/2026-10-05T10-33-53-125Z/report.json`，同目录 `updates-{light,dark}-maximized.png`。

Python 覆盖有效/非法 SemVer、完整预发布排序、元数据、Preview/stable、两源失败与404、GitHub分页/清单、检查间隔/重启/手动/提示24小时、地区隐私、可信HTTPS与重定向、确认/取消、截断/超长/响应长度/哈希、匹配与不匹配的跨源兜底、应用前缓存变化拒绝，及真实 Qt 信号、请求额度、重复ID、线程取消与关闭。

Chrome 使用明确的原生通道测试替身，验证启动提示/确认登记、来源失败、两主题、下载两步确认与进度、应用确认、设置、取消/迟到结果、无桌面能力。没有下载100MB发布物或运行安装器。初次测试服务器误将整个仓库作为 Vite 根，扫描历史 HTML 导致导航超时；修复为 frontend 根及单独 fixture，保留失败记录，仅终止此测试拥有且已核实命令的进程。

实际原生联网只读检查，当时服务器清单404、GitHub旧发布缺少校验Windows包，整体返回 incomplete，分别列明HTTP_ERROR/CHECKSUM_MISSING；地区只得到网络loc=US，无IP写入。未公开发布清单前这属于真实状态。此次没有实际产品Qt GUI、冻结成品、安装器、跨版本设置保留或缓存清理验收。主线接线后再构建、验收。

规范依据：[SemVer 2.0.0](https://semver.org/)、[GitHub Releases API](https://docs.github.com/en/rest/releases/releases)。
