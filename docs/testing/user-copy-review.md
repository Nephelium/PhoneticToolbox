# 用户文案与使用说明审校

对应任务：`USER-COPY-REVIEW`。按井井的明确要求，正文与界面直接说明功能和操作；开发编号、接口接线、运行资格、迁移及验收诊断留在内部文档与日志。用户无法处理的开发问题由开发者负责。

## 改动范围

- 逐章检查正式说明书的 20 章，微调正文、表格、图注与替代文字，并同步 `docs/manual/` 的用户使用说明。
- 用工具名称替换面向用户的模块编号。导出文件夹示意改用用途描述，实际文件名与保存格式、稳定 ID、引用及媒体身份保留。
- 删除接口未接入、任务宿主、已验证节点等开发状态段落或恢复表行。文件格式、输入上限、单位、参数作用和必要恢复步骤保留。
- 清理设置、合成、录音、文件预览、平台错误提示及声道来源说明中的防御表述。项目页移除开发用流程检查按钮，任务记录与取消、重试继续保留。
- 关于页的来源信息由统一登记生成。原始来源登记保留未决许可和版本审计，界面展示作者、来源、实际版本与已明确的许可信息。
- 素材图注与正文同步，媒体文件未改写。同期新增的设置截图与其登记保留。
- 阅读页隐藏素材制作、原生任务与验收记录，原登记留存；保留自然录音／合成示例类型、时长、采样率和声道数。缺少素材或未知节点提示使用者可理解的状态，未知节点的原文保留。
- 在说明书编写约定中记录用户文案的内容边界。

典型调整：显示轴直接说明图中显示范围；基频提取方法和参数另行说明。合成刺激的说明给出试听和检验步骤。配色介绍保留参考来源、深浅模式与科研曲线的颜色含义，删除背书声明。

## 验证

状态：`verified`，验证范围为 Windows 当前源码与阅读资源。

- 严格说明书校验：20 章、1,010 项素材，零警告、零错误。使用 `scripts/manual/validate.py --project manual --strict`。
- `scripts/manual/build.py --distribution software`、前端 `typecheck`、`ui-data:check` 和 `build` 成功。构建仍有已有大分块提示。
- 定向前端检查最终 55 项通过，零失败。范围包括阅读层、安全渲染、缺少素材、隐藏制作记录、主题、相关功能参数与来源覆盖。执行 `node --test tests/manual-reader*.test.ts tests/appearance.test.ts tests/m05.test.ts tests/m06.test.ts tests/m07.test.ts tests/m08.test.ts tests/m11.test.ts tests/m14.test.ts tests/source-coverage.test.ts`。
- 实际 Qt 阅读器从使用说明入口打开全部 20 章，标题和正文呈现，正文无可见开发模块编号；全文搜索可打开章节，配色说明无背书声明。最终记录：`output/validation/user-copy-review/qt-da1b0a8284a746b4a8effacef25283dd/report.json`。先前检查识别出制作记录中的编号，已修正显示层后重验。
- 阅读源文中的 PCM16、PCM24、FLOAT、dBFS、Praat、REAPER、Griffin、关键采样值及时间步长计数与本轮检查前一致。实际文件名、算法、音频与图片字节未改写。构建登记的 1,030 个章节/素材文件在 public 与 dist 中哈希一致。
- 8 个仅修改错误文案的 Python 文件语法检查通过。最终文档 UTF-8、JSON/Python 格式与本地链接检查覆盖 1,904 个文件，零错误；102 个候选文件的 `git diff --check` 通过。具体记录保存于 `docs-validation-final.json` 与 `diff-check-final.json`。
- 全仓 `scripts/check_architecture.py` 返回同期 `frontend/public/vocal-tract/platform.js` 和 `v3.css` 的资源摘要不一致。两项不属于本次文案改动，保留同期内容。本次修改的 `index.html` 与登记摘要一致，未调整全仓检查标准。

检查脚本、逐处修订记录、生成日志、正文与文件哈希记录、Qt 截图留在 `output/validation/user-copy-review/`。Qt 临时目录在所属进程退出后按项目脚本清理，保留回执。

本次未重验科学算法、媒体录制、实体设备或最终 EXE，没有公开发布或上传。
