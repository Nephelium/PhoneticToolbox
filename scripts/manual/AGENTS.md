# P10 说明书工程工具

- `validate.py` 校验权威工程，`build.py` 生成只读阅读资源。两者只用 Python 标准库。
- `seed_content.py`、`prepare_examples.py` 归主线内容作者维护，编辑器改动不得覆盖它们。
- 结构及支持节点从 `frontend/src/manual/schema-*.json` 读取。源工程不可被构建回写。
- 默认 software 阅读版允许 software-only 素材；public 版排除受限素材并报告占位。
- 不清理源工程。software 构建成功后只清上一份工程索引/构建报告共同登记、当前不再引用且 SHA 未变的生成文件；未知、已变化或含联接的旧输出报错保留。public 构建遇到任何旧受限/未列入文件仍须报错，不自动删除。
- `python scripts/manual/validate.py --project manual --strict` 与 `python scripts/manual/build.py --project manual --output output/manual-public --distribution public`。
- `pack_media.py` 仅在本机压缩发行包临时快照中生成 WebP/MP3，使用已有 Pillow/FFmpeg；源工程 PNG/WAV 不变。标准 `validate.py` / `build.py` 仍只依赖 Python 标准库。
