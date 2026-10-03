# P15-STATIC：未发布静态页部署

状态：**verified，限定下列公网静态页、HTTPS 与浏览器验收**。部署日期：2026-10-03（Asia/Shanghai）。

## 授权与范围

井井明确授权通过 SSH 将之前制作的未发布页面部署至服务器，使访问 `www.phonetictoolbox.com` 的用户先看到该页面。本轮只开放静态页，未部署 v3 工作台、API、账号、数据库或科学计算环境。未推送 Git、生成软件发行包或修改其他模块。

原页面为 [temporary-site/index.html](../../temporary-site/index.html)，9294 字节，正文沿用正在开发中及预计 2026 年 10 月底开放。HTML 未改动，无外部字体、脚本、图片或网络依赖。

## 服务器与文件

- 公网地址：`8.134.196.183`；SSH 登录后确认私网地址为 `172.18.62.173`。
- 实际系统：Ubuntu 24.04.2 LTS / x86_64；使用已有 SSH known_hosts 校验连接。
- 部署前只有 SSH 和本机 DNS 监听，没有 Nginx/Caddy/Apache 或既有站点文件；UFW 为 inactive，未修改防火墙。
- 从服务器既有 Ubuntu 阿里云镜像源安装 Nginx `1.24.0-2ubuntu7.18`、Certbot `2.9.0-1` 及其依赖，共 9 个新包。没有升级已有包、重启系统或改 SSH 设置。
- 网页：`/var/www/phonetictoolbox-coming-soon/index.html`。
- Nginx：`/etc/nginx/conf.d/phonetictoolbox-coming-soon.conf`，对应 [HTTPS 配置](../../deployment/temporary-site/nginx-https.conf)。
- 最初的 HTTP 配置保留于 `/etc/nginx/phonetictoolbox-coming-soon.http-bootstrap`，本地对应 [HTTP 配置](../../deployment/temporary-site/nginx-http.conf)。该保留文件不在 Nginx 的 `*.conf` 加载目录内。
- 证书：`/etc/letsencrypt/live/phonetictoolbox.com/`。私钥与 ACME 账号仅存服务器，未下载。
- 续期验证根目录：`/var/www/letsencrypt`；80 端口保留 ACME 验证路径，其余请求跳转 HTTPS。
- 续期部署钩子：`/etc/letsencrypt/renewal-hooks/deploy/phonetictoolbox-nginx.sh`，对应 [本地脚本](../../deployment/temporary-site/renew-nginx.sh)，模式 0755。先检查配置，再 reload Nginx。

Nginx 与 certbot.timer 均为 active / enabled。注册证书时未配置通知邮箱，续期依赖系统定时器。Let’s Encrypt 证书包含两个域名，本次证书到期时间为 2027-01-01 11:29:11 UTC。

## 实际验收

| 检查 | 结果 |
| --- | --- |
| Windows `Resolve-DnsName` 与服务器 `getent ahostsv4` | 两个域名均解析到指定公网 IP；查询未发现 AAAA 地址 |
| `nginx -t` | 配置有效，reload 成功 |
| `http://www.phonetictoolbox.com/` | 301 → `https://www.phonetictoolbox.com/` |
| `http://phonetictoolbox.com/` | 301 → 同上 |
| `https://phonetictoolbox.com/` | 有效 TLS，301 → 同上 |
| `https://www.phonetictoolbox.com/` | 有效 TLS，200，UTF-8 HTML，9294 字节 |
| HTTPS 公网下载回读 | 与原始 HTML 及服务器文件 SHA-256 一致 |
| `/favicon.ico` | 204，避免无图标页面产生重复 404 |
| `/api/health`、`/docs` | 404，未提供应用/API 服务 |
| `/.env` | 403，隐藏路径拒绝访问 |
| `certbot renew --cert-name phonetictoolbox.com --dry-run --run-deploy-hooks --no-random-sleep-on-renew` | 模拟续期成功，部署钩子执行成功 |
| 最终钩子直接执行 | `RENEW_HOOK_OK`，最终改用 `nginx -t -q`，保留失败输出并消除成功检查写 stderr 导致的红色提示 |
| 独立 Chrome / Playwright CLI | 1440×1000 和 390×844，两种布局实际截图检查通过，内容宽度分别为 1440 和 390，无横向溢出 |

公网请求使用 `curl.exe --noproxy '*' --connect-timeout 8 --max-time 20`，没有关闭 TLS 校验。浏览器访问裸域 HTTP 后也实际跳转到 HTTPS 主域。

SHA-256：

```text
index.html        744ece83cdfbcb53e59126193f1de0a7dd1a09b280620b9ab1a0516ee86f69d7
nginx-https.conf  5374fef41a7d94a1e13184ea0d1b233e2533cf00ccf976532fe7557b26ce9ff3
renew-nginx.sh    7792a1d13cda206bdc1c77971d80f5e349b93ecc88b0b85d65c7654d4d9d4f3a
```

证据目录：`output/playwright/p15-static-20261003/`，含公网 HTTP/HTTPS 回读、响应头及 desktop.png / mobile.png。浏览器初测唯一错误是不存在的 favicon 请求，已在 Nginx 单独返回 204；页面内容保持原字节。

## 后续维护

修改占位文案时更新原 HTML，上传到上述专用根目录并核对哈希。更改 Nginx 后必须先 `nginx -t`，再 `systemctl reload nginx`。正式 v3 网站上线仍需独立实施和验收，不能把本次静态站点可访问当作业务服务已上线。

证书续期模拟成功不等于已观察过未来自动续期；未执行服务器重启验证，也未声称所有运营商或全球 DNS 缓存均已同步。
