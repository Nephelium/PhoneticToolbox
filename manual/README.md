# 应用内使用说明

`project.json` 管理章节、小节、参考资料、搜索和素材。`chapters/` 保存可编辑的结构化正文，阅读器与作者编辑器使用同一套渲染组件。

普通用户从应用中的使用说明或模块帮助进入对应章节。作者从 [打开说明书编辑器.cmd](../tools/manual-studio/打开说明书编辑器.cmd) 启动独立工具，选择本目录，即可修改正文、章节、图注、图片、音频、视频、表格、代码、公式、引用和排版。编辑器保存后仍需重建阅读资源，再构建应用。作者工具不随普通应用分发。

## 维护流程

1. 打开独立编辑器并选择 `manual` 工程。
2. 修改内容。跨章引用和帮助跳转依赖稳定的英文 ID，修改标题时保留原 ID。
3. 插入图片后填写图注。例音注明来源类型、处理设置和解释范围。
4. 保存工程。编辑器保留恢复与历史记录，可导出整个工程作为独立备份。
5. 校验并生成应用资源，再运行前端构建。

```powershell
python scripts/manual/validate.py --project manual --strict
python scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
cd frontend
npm run build
```

命令使用项目已经准备的 Python 与 Node 环境。第一次配置作者工具请按 [编辑器 README](../tools/manual-studio/README.md) 操作。统一写作约定见 [AUTHORING.md](AUTHORING.md)。

## 素材分发

自然录音和相关处理结果获准随软件分发，未获准上传公开 GitHub。它们及相应截图保存在忽略的 `assets/software-only/`，素材清单使用 `git:false` 与 `distribution:software-only`。私人原始路径仅进入本机忽略的来源记录。

公开说明书构建应使用新的输出目录，避免把此前软件版输出残留带入公开材料：

```powershell
python scripts/manual/build.py --project manual --output output/manual-public-preview --distribution public
```

公开构建保留说明和素材标识，过滤受限媒体。已有同名输出含不同媒体时，构建会拒绝覆盖，请选择新目录。

生理参数合成章节已有正文，当前工程登记为 reviewed，后续随模块继续修订。唇形视频演示待作者后续录制，未插入虚构示例。
