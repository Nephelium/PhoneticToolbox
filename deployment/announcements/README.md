# 应用公告

Windows、macOS和网页版3.1.0起在联网打开工作台时读取固定官网地址 `/announcements/v1.json`。每条公告在当前设备只自动显示一次，已读状态仅存本机。用户可从公告入口重新查看。它不承担软件更新、远程执行或用户跟踪。

`public.json` 是正式清单。井井已于2026-10-24明确确认输入法上线，首条通知已准备，随3.1.0统一发布流程启用。`first-announcement.example.json` 保留为独立样例。发布前核对真实官网下载入口，不能只因GitHub存在包就宣布官网已上线。

## 修改与发布

保持同一条公告的ID不变可修正文字，不会重新弹出。新的独立通知使用新ID。成功发布空 `items` 可撤回清单，已读状态无需上传或清除。正文为纯文本，链接仅允许官网HTTPS。可选 `expiresAt`、`minVersion`、`maxVersion` 和 `platforms` 分别控制有效期、版本和平台。

按当前请求取得对外发布授权后，先预览工具输出，再明确指定当前远端文件SHA进行原子替换。首次创建使用 `absent`。SSH辅助脚本由调用者传入，不在仓库保存凭据。

```powershell
py -3.11 -B -X utf8 deployment/announcements/publish.py --feed deployment/announcements/public.json
py -3.11 -B -X utf8 deployment/announcements/publish.py --feed deployment/announcements/public.json --ssh-helper <本机SSH辅助脚本> --expected-sha256 <当前SHA或absent> --apply
```

首次部署由发行流程将 `announcements.nginx.conf` 纳入现有HTTPS站点，经 `nginx -t` 和获准reload后启用。后续公告只替换JSON，不重启应用服务、不迁移数据库、不重新打包。工具执行本地严格校验、远端SHA比较交换和完整文件校验，失败保留原文件。发布后另核实际公网全文及CORS，不能将本地预览当成上线。
