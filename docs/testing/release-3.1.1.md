# PhoneticToolbox 3.1.1 发行验收

状态：三端最终成品验收、统一公开发布与公网回读全部通过。用户已明确确认完整版本文案，[更新说明](../../release/notes/3.1.1.md)SHA为`e5e896165b258aed320416034bbef17a3424db48258239e705e8b5a7e7c02326`，GitHub与应用频道使用同文。

## 冻结与成品

业务源码冻结`cc1cb2cbc3eba8a6b02b809aedf68ae9b9e41158`，946项来源清单SHA`8fe0a6f87ef085a06c8862c3fb19ade4779030faf56f6b05ab26dfc8f2deb66a`。原始GeoAtlas试用资料2清单`e86ed70d`、930点/510项和科学输入保持；Mac与Windows607前端共享文件逐字节核对。Web科学359项来源与同一冻结对应，使用真实3.1.1轮子及平台浏览器包，原仓库浏览器锁与依赖环境未覆盖。

| 成品 | 字节 | SHA256 |
| --- | ---: | --- |
| Windows免安装EXE | 498165182 | `a7866c539fa2d0b89a4af4a8b35ad0cf41f51a8998c2be14080b9c76a44e7ebe` |
| Windows安装EXE | 497977328 | `1c1ec583e4954612670f3c91ce7e3d6d1600d3070b4c9d0575a0bf690634d350` |
| Windows更新ZIP | 498165748 | `b4c39653f8c6a5c3b79f990da1237213648553a39fcf34c1c943363d9b11eef8` |
| Mac Apple Silicon ZIP | 491915761 | `6b4227705b3cf60c1d3c7cfa5759839f42b279af27d83aa9e02975b05c9514ac` |

## 已执行检查

- 源码：前端789项、typecheck、5项原生全屏状态回归；真实Chrome完整42组及12次拖动/滚轮、4组Canvas/SVG/PNG像素、最新10组筛选/边界/各模式选点/导出/M17输入。全应用6模块和6标签重复计算/读取及后台轮询检查通过。地图实际性能及界面细项见[M19验收](m19-geolinguistics.md#2026-10-25-性能复审与311最终验收)。
- Windows最终字节：`scripts/verify_local_package.py`完整16项含工程外运行、长路径安装/卸载、缓存、两种更新交接及重开、默认GPU可见运行、冗余工作、阅读响应、渲染像素和清理。`verify_geolinguistics_package.py`同EXE隔离重验21项。普通入口外部CDP/Win32真实保存对话框11组及新增10组、手册/音标168项，正常退出0。首次地图测试模式在重载连接时失败，原回执保留，同字节重验通过，不代表长期稳定性或根因已确定。
- Mac实际机器：macOS26.6.2、Apple M4、Qt6.10.2。源运行26项及Windows/Mac231项数值对照，最终应用26项科学、Qt、界面响应、M19、解压迁移后Qt/M19、前后严格签名通过。普通入口六会话覆盖自动省市县、双/四图、来源键盘、全屏恢复、新导出/筛选与M17输入，全部正常退出；补验真实Esc退出后DOM/AX全屏false、原尺寸与视野恢复。完整ZIP6886成员CRC、恢复后SHA与全部证据摘要通过。
- Linux：私有`pcc1cb2`1425发布文件及803运行文件逐SHA，profile`c5eff571`129488B绑定，9项实际科学运算及interactive/VTL通过。公开安全前端排除桌面专用说明书素材，保留现有22章/8图公开读本与17份地图资料，网页最终候选23+10组通过，无生产账号/数据库写入。上线只读预检通过，活动科学任务为零，原版本与回退保留。
- 文案：完整10项经明确确认，应用更新弹窗7组显示检查通过，官网候选3组覆盖1440/768/390、七版本切换和无JS，批准内容和三类下载链接对应本版。

## 证据与范围

本版证据根为`output/validation/release-311-20261025-cc1cb2/`，统一产品收口见`windows-final-acceptance.json`，Windows完整回执`output/validation/local-package-b5cf0fd3/report.json`，Mac回收证据位于证据根`mac-evidence/`，Web见`candidate/report.json`和`private-final-audit.json`。所有候选、失败、重验和实际发布分别记录，不借前代包通过结果扩大本代结论。

Mac为Apple Silicon测试版，ad hoc签名，未公证；macOS14为构建目标，本轮仅实际26.6.2，未新增Intel、Safari、实体麦克风或其他机器验收。浏览器使用隔离合成账号/偏好/工作区配合实际HTTPS资源，不冒充真实生产账号持久化测试。旧构建20处按正式脚本清理，保留输入缓存、原始数据、最终证据与服务器回退。未使用早期候选或未审阅的同期官网源码。

## 公开发布与后验

官网、stable/preview和[GitHub v3.1.1](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.1.1)均已发布；[网页版](https://www.phonetictoolbox.com/app/)激活`pcc1cb2`。实际公网地图23+10组、官网3组通过，两个频道与GitHub正文均和批准文案逐字节相同，四资产大小/digest对应最终成品。四包实际HTTPS完整GET累计1986224019B及HEAD/64KiB Range/SHA通过。服务器1425发布文件/803运行文件与旧1407/803回退逐SHA后验通过，原配置、Nginx、反馈、论文目录和公告保持，未迁移数据库。

Mac完整ZIP与6886成员CRC、全部必要验收证据及输入清单已回收到本机，源码保留在Windows与GitHub，后续不依赖Mac服务器在线；未制作整台Mac系统/开发环境磁盘镜像。旧20处构建/安装暂存已按正式脚本清理，9处既有未完成对象保留。最终索引为证据根`completion.json`，包含所有关键回执的SHA与实际验证范围。
