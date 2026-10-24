# 语言地图

M19 在统一工作台内提供四个入口：井韶子整理的方言分布、方言相似性、语言现象、我的研究地图。Windows、macOS 与网页共用入口、领域代码与导出路径。

首次进入可点“下载资料”，查看实际进度、取消或失败后重试。完整下载经清单、SHA、地图和矩阵校验后一次写入本机存储，自动显示资料；再次进入离线复用。载入后“更新资料”可重新取得服务器版本，“选择资料夹”保留已有文件的导入方式。资料替换与个人工程分别保存，下载或保存失败不覆盖原资料及编辑。

- `model.ts` 保存领域类型、完整分类路径计数、项目校验、原矩阵距离查询和原值覆盖规则。
- `groups.ts` 按逐图审核成员表解析大类，保留待核原码与地点例外，同点同类计数去重。
- `GeoMap.vue` 使用 OpenLayers 离线矢量底图，点选、Shift 框选、分层边界、共同视野、绘图及 SVG 导出。
- `map-paint.ts` 共用几何填充与边界可见性，开放线不填面。按缩放只加载一个行政层，固定territory省界不叠画；其他边界和符号保持。Canvas和SVG逐项处理GeometryCollection，PNG沿同一SVG路径生成。
- `projected-base.ts` 按不可变资料对象共享WeakMap投影缓存，多图复用固定投影。面板销毁时清自己的Source/监听器并释放页面引用，不销毁其他面板共享Feature。
- `GeolinguisticsPage.vue` 负责目录、联动表格、图例、比较、个人工程与导入导出；数据读取与文件保存走 `platform/geolinguistics.ts`。
- `FeatureTitle.vue` 保留中文标题字体，并为方括号内的 IPA 使用专用字体；`export.ts` 组合多幅 SVG 并保持独立裁剪范围。
- `platform/geolinguistics-download.ts` 从官网固定资料前缀顺序读取16个运行文件，省略账号凭据、不跟随重定向，限额内流式接收并共用完整清单验证。进度最多每100ms更新一次，文件/阶段完成时立即更新；下载和资料夹导入按调用顺序串行提交。
- 资料端口按当前版本复用已校验的领域数据、五张地图和五份距离矩阵，重复查询不重新解析或复制矩阵。多个载入请求各自可以取消，最后一位等待者取消后终止实际读取；失败后可重新载入。原子写入成功才发布新资料，迟到的旧请求不能覆盖新版本。
- 工程复用 LocalAssets 的 values/CAS，不迁移现存数据库。原始表格保持只读，校订、类别释义和自建点保存为项目覆盖记录。

资料来源和核查范围见[计划](../../../../docs/plans/2026-10-22-m19-geolinguistics.md)。本地构建通过 `scripts/prepare_geolinguistics.py` 生成完整两套数据，原书核对需要时通过 `prepare_geolinguistics_book.py` 生成本机参考图，参考图不随试用包分发。原始资料路径由命令行传入，生成资料在 Git 忽略目录，不进入公开源码或应用 EXE。

标题构建必须传 `--titles <已核对的 TSV>`，包含 `feature_id/title/toc_pdf_page` 三列和全部 510 项。本机文件为 `local-data/geolinguistics-title-review.tsv`，仅更正标题时加 `--titles-only`，避免重做论文统计。OCR 输出只供对照，不作为直接显示的标题。标题与图例释义分别记录核查范围，疑难音标由用户对照原图确认。

类别释义的私有输入为 `local-data/geolinguistics-category-review.json`，以特征、编码、原书页码和原书 SHA 绑定。`scripts/prepare_geolinguistics_labels.py --review <JSON> --resources local-data/geolinguistics` 只更新目录，文字/IPA 和编码对应两项都核实才显示释义。待核项不能混入正式字典；撤回的释义会从目录移除。全量数据构建用 `--category-review <JSON>` 接入同一字典，已有核对记录而未传该参数时拒绝覆盖。原书只有图形符号，不能按图例排列猜测 A/B 或数字顺序。复合编码按原 `#` 顺序显示，IPA 内的斜线保持原书含义。

大类审核输入为 `local-data/geolinguistics/reviewed/broad-{phonology,lexicon,grammar}.json`，通过 `prepare_geolinguistics_groups.py --review <三卷JSON> --resources local-data/geolinguistics` 接入。全量/标题构建用 `--broad-review <三卷JSON>`，已有大类记录时禁止无审核参数覆盖。每组必须同时核实文字与本图编码对应，保留原书标题/共同特征概括的区别。地点与原图存在冲突时逐原子单列例外，同一地点可保留多项冲突，原值不改；目录兼容旧单项对象和多项数组，地点数与原子数分别统计。缺失图页说明保留在目录与导出中。

界面、地点对照和 CSV/SVG/PNG 共用释义。默认大类视图显示已审核组名，待核地点显示为空心点，部分记录待核用虚线边框；图例仅列已对应大类，不将未对应记录合成语言类别。待核地点可单独筛选，同点只计一次。原码保留于展开记录和细类视图。细类视图保持原字典和个人填写释义；个人细类释义不会自动成为大类。CSV 同时保留原始值和 PDF 页码。SVG 按实际字体测量文本，换行保持组合附标和普通长度的连续数字完整。类别核对仅代表字典关系，不代表全部 930 个原始观测均已对照原图。

中国范围只显示中国/全国入口和省市县。旧世界入口保存的固定视野载入或导入时回到全国，原观测、分类、自建点与绘图保持，普通手工视野不改。DataV官方当前只提供中国选择器，因此未保留世界入口。

地区选择沿真实parent路径列下一级，直属区县可直接出现在省之后；仅实际有子级时显示第三层。台湾在GeoAtlas这一版本只提供省级轮廓，末端市也可整体查看，不伪造下级节点。绘图层可复用真实上级末端面保持缩放连续，map-index及分类统计不因此增加虚构实体。旧三层资料及已保存的地区ID保留兼容。

2026-10-24当前沿用GeoAtlas冻结试用资料2，按井井最新要求停止进一步底图内容核查及采用新细化轮廓，尚未替换公开资料。构建依次使用prepare_geolinguistics_datav.py、normalize_geolinguistics_datav.py、prepare_geolinguistics_datav_runtime.py。原件不改，派生修复、推定坐标适配和逐项批准的岛礁显示补充各有独立审计。行政显示范围不作干陆地掩膜或面积测量依据，辅助断续符号不进入行政统计与地点匹配。学习交流用途告知见third_party/licenses/geolinguistics/DataV-GeoAtlas-use-notice.txt。

开发环境从 `local-data/geolinguistics/` 读取；Windows 本地交付从免安装 EXE 旁的 `geolinguistics-data/` 读取同一套数据。三端也均可选择本地资料夹，`geolinguistics-directory.ts` 校验 16 文件清单、尺寸、SHA 和资料结构后原子替换 LocalAssets 的资料值。目录中未列入的文件不读取。实例与持久资料按研究工作区 scope 隔离；工程只存编辑和视野，资料不上传。原书 PDF 保持外部文件。新版开放底图统一为 EPSG:4326，原始调查点坐标不变；旧资料仍按各自来源保留已有显示转换。行政级别按源数据保留，台湾县市、乡镇区独立标识，不能凭层数解释成统一的行政制度。

开放底图的实体覆盖与几何差异分别验收。OSM/Geofabrik、国土测绘中心、香港政府数据各自保留许可和完整派生数据库，OSM 的 ODbL 不延伸到独立方言观测及个人编辑。地图屏幕保留适用的最短必要署名，地图来源按钮展开完整来源与许可，支持键盘操作及独立多图收起，PNG/SVG 记录完整署名、来源链接、年份和许可。上游重叠、父子范围差异和历史地区代码保留在公开来源审计中，不为显示效果裁界。旧工程中的已撤并地区 ID 保持原值，明确提示重新选择；只有用户选择使用全国才清除旧地区设置。

交付由 `scripts/package_geolinguistics_data.py` 的 16 文件清单决定，只复制交互所需数据和矢量底图，逐文件核对 SHA。书页图片、PDF、书页清单和私有构建审计不分发。界面只显示书名、卷、图号和 PDF 页码，无书页图片请求。

验证：`node --test frontend/tests/geolinguistics.test.ts frontend/tests/geolinguistics-groups.test.ts`、前端 typecheck/build、`scripts/verify_geolinguistics_qt.py` 、`scripts/verify_geolinguistics_browser.py` 及原生静态资源测试。源码 Qt 与最终 EXE 的证据分别记录。


默认进入县级方言分布。语言现象默认“大类”，可选“有大类释义”或全部 510 项；细类仍可选 10 个完整释义现象。大类与细类分别显示审核范围，不把已核部分等同全图完成。`Project.featureScope`、`Project.bookmarks`、`Project.featureDetail` 是可选字段，旧工程缺少 featureDetail 时保持 fine 和原筛选，无需数据库迁移，旧 `atlasView` 仅兼容读取且不再控制显示。旧工程保存的未完整释义特征会自动进入全部特征范围，避免丢失工作位置。

`compareFeatures` 区分完整记录相同、不同与含缺值，保留有序变体，读取项目校订；原始观测保持只读。对照界面按语音、词汇、语法过滤，可替换调查点、跳回特征地图、导出代码/含义/页码。论文距离独立查询原矩阵，不用简单相同项比例替代。MDS、聚类和距离直接读取论文结果。读者可从相近地点直接进入两地对照。
