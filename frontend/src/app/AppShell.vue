<script setup lang="ts">
import { ref,computed,watch,onMounted,onUnmounted,nextTick } from 'vue';import { groups,modules } from './registry.ts';import { host,workspace,states,saveDraft } from '../state/workspace.ts';import { stop } from '../state/audio.ts';
import AudioTransport from '../components/AudioTransport.vue';import AppIcon from '../components/AppIcon.vue';import ModalDialog from '../components/ModalDialog.vue';import MethodReferences from '../components/MethodReferences.vue';import WorkspaceView from './WorkspaceView.vue';import version from '../version.json';import fontLicense from '../assets/Doulos-OFL.txt?raw';import logo from '../assets/k2.png';
import VocalTractPage from '../modules/vocal-tract/VocalTractPage.vue';
import ParameterEstimationPage from '../modules/parameter-estimation/ParameterEstimationPage.vue';
import ParameterDisplayPage from '../modules/parameter-display/ParameterDisplayPage.vue';
import EggAnalysisPage from '../modules/egg-analysis/EggAnalysisPage.vue';
import Spec2WavPage from '../modules/spectrogram-to-audio/Spec2WavPage.vue';
import {m01State,saveM01,forgetM01} from '../modules/parameter-estimation/store.ts';
import {dirty} from '../modules/parameter-estimation/state.ts';
import {previewFiles,type ResearchContext} from '../platform/research.ts';
import {desktopFiles} from '../platform/desktop.ts';
import FontSettings from '../components/FontSettings.vue';
import {selectFontOwner} from '../state/fonts.ts';
const props=defineProps<{research?:ResearchContext}>();const emit=defineEmits<{leaveProject:[]}>();
const defaultContext:ResearchContext={key:'local:M01',label:desktopFiles?'本机目录':'本机预览',files:desktopFiles??previewFiles()};
const researchContext=computed(()=>props.research??defaultContext);
watch(()=>props.research?.ownerId,owner=>{void selectFontOwner(owner);},{immediate:true});
const previewKey=(id:string)=>props.research?researchContext.value.key.replace(/:M01$/,':'+id):id;
const m01=computed(()=>m01State(researchContext.value.key));
const m10Dirty=ref(false);
const m03Page=ref<{save:()=>boolean}>();
const m02Page=ref<{save:()=>boolean}>();
const moduleDirty=(id:string)=>id==='M10'?m10Dirty.value:id==='M01'?dirty(m01.value):!!states[previewKey(id)]?.dirty;
const recordingEntry=location.hash==='#M10';
const active=ref(recordingEntry?'M10':'home'),tabs=ref<string[]>(recordingEntry?['home','M10']:['home']),query=ref('');const collapsed=ref(host.projects.read('collapsed',false));
type Theme='system'|'light'|'dark';const savedTheme=host.projects.read<string>('theme','system');const theme=ref<Theme>(['system','light','dark'].includes(savedTheme)?savedTheme as Theme:'system');
const system=matchMedia('(prefers-color-scheme: dark)');const applyTheme=()=>document.documentElement.dataset.theme=theme.value==='system'?(system.matches?'dark':'light'):theme.value;
watch(theme,()=>{applyTheme();host.projects.write('theme',theme.value);},{immediate:true});watch(collapsed,v=>host.projects.write('collapsed',v));
system.addEventListener('change',applyTheme);onUnmounted(()=>system.removeEventListener('change',applyTheme));
const storedRecent=host.projects.read<unknown>('recent',[]);const recent=ref((Array.isArray(storedRecent)?storedRecent:[]).filter(id=>modules.some(m=>m.id===id)).slice(0,5));
const current=computed(()=>modules.find(m=>m.id===active.value));const visible=computed(()=>modules.filter(m=>(m.title+' '+m.description+' '+m.id).toLowerCase().includes(query.value.trim().toLowerCase())));
const modal=ref(''),closing=ref(''),notice=ref('');const referencesId=ref<string|undefined>();
function open(id:string){stop();if(!tabs.value.includes(id))tabs.value.push(id);active.value=id;if(id!=='home'){if(id!=='M01')workspace(previewKey(id));recent.value=[id,...recent.value.filter(x=>x!==id)].slice(0,5);host.projects.write('recent',recent.value);}}
function remove(id:string){stop();if(id==='M10')m10Dirty.value=false;const index=tabs.value.indexOf(id);tabs.value=tabs.value.filter(x=>x!==id);if(id==='M01')forgetM01(researchContext.value.key);else delete states[previewKey(id)];if(active.value===id)active.value=tabs.value[Math.max(0,index-1)];closing.value='';void nextTick(()=>document.getElementById('tab-'+active.value)?.focus());}
function close(id:string){if(moduleDirty(id))closing.value=id;else remove(id);}
function saveClose(){if(closing.value==='M03'){if(m03Page.value?.save())remove('M03');return;}if(closing.value==='M02'){if(m02Page.value?.save())remove('M02');return;}if(closing.value==='M10'){notice.value='请返回关键帧，保存或取消正在编辑的姿势后再关闭。';return;}if(closing.value==='M01'&&m01.value.drawer){notice.value='请先应用或取消参数/设置对话框中的编辑，再保存关闭。';return;}if(closing.value==='M01'?saveM01(researchContext.value.key):saveDraft(previewKey(closing.value)))remove(closing.value);else notice.value='本机草稿保存失败，标签仍保留。请检查浏览器存储权限。';}
function refs(id?:string){referencesId.value=id;modal.value='references';}
function tabKey(event:KeyboardEvent){let n=tabs.value.indexOf(active.value);if(event.key==='ArrowRight')n=(n+1)%tabs.value.length;else if(event.key==='ArrowLeft')n=(n-1+tabs.value.length)%tabs.value.length;else if(event.key==='Home')n=0;else if(event.key==='End')n=tabs.value.length-1;else return;event.preventDefault();open(tabs.value[n]);void nextTick(()=>document.getElementById('tab-'+active.value)?.focus());}
function beforeUnload(event:BeforeUnloadEvent){if(dirty(m01.value)||tabs.value.some(id=>moduleDirty(id))){event.preventDefault();event.returnValue='';}}
onMounted(()=>window.addEventListener('beforeunload',beforeUnload));onUnmounted(()=>window.removeEventListener('beforeunload',beforeUnload));
const modalTitle=computed(()=>({settings:'工作台设置',help:'使用说明',update:'检查更新',about:'关于 PhoneticToolbox',references:'开源与学术致谢','font-license':'Doulos SIL 字体许可'}[modal.value]||''));
</script>
<template>
<a href="#main-content" class="skip-link">跳到工作区</a>
<div class="app-shell" :class="{collapsed}">
<aside class="sidebar" aria-label="工具导航">
<div class="brand">
<img :src="logo" alt="PhoneticToolbox 波形团子"/>
<span>PhoneticToolbox<small>语音研究工具箱</small>
</span>
<button class="icon-button" :aria-label="collapsed?'展开侧栏':'收起侧栏'" @click="collapsed=!collapsed">
<AppIcon name="menu"/>
</button>
</div>
<label class="search-box">
<AppIcon name="search"/>
<input v-model="query" aria-label="搜索工具" placeholder="搜索工具…" :title="collapsed?'搜索工具':''"/>
</label>
<nav>
<button class="nav-item" :class="{selected:active==='home'}" title="首页" @click="open('home')">
<AppIcon name="home"/>
<span>首页</span>
</button>
<section v-for="(group,index) in groups" :key="group" class="nav-group">
<h2>{{group}}</h2>
<button v-for="m in visible.filter(m=>m.group===index)" :key="m.id" class="nav-item" :class="{selected:active===m.id}" :title="m.title" @click="open(m.id)">
<AppIcon :name="m.icon"/>
<span>{{m.title}}</span>
</button>
</section>
<p v-if="!visible.length" class="empty-small">没有匹配的工具</p>
</nav>
<div class="sidebar-bottom">
<div class="utility-grid">
<button v-for="[key,label,icon] in [['settings','设置','settings'],['help','使用说明','book'],['update','检查更新','update'],['about','关于','info']]" :key="key" :title="label" @click="modal=key">
<AppIcon :name="icon"/>
<span>{{label}}</span>
</button>
</div>
<label class="theme-picker">
<AppIcon name="sun"/>
<select v-model="theme" aria-label="配色主题">
<option value="light">浅色</option>
<option value="dark">深色</option>
<option value="system">跟随系统</option>
</select>
</label>
</div>
</aside>
<div class="main-shell">
<header class="topbar">
<div class="workspace-tabs" role="tablist" aria-label="工作区标签">
<div v-for="id in tabs" :key="id" class="tab-wrap" :class="{active:active===id}">
<button :id="'tab-'+id" role="tab" :aria-selected="active===id" :tabindex="active===id?0:-1" aria-controls="main-content" @click="open(id)" @keydown="tabKey">
<AppIcon v-if="id==='home'" name="home"/>{{id==='home'?'首页':modules.find(m=>m.id===id)?.title}}<span v-if="moduleDirty(id)" aria-label="未保存">•</span>
</button>
<button v-if="id!=='home'" class="tab-close" :aria-label="'关闭 '+modules.find(m=>m.id===id)?.title" @click="close(id)">
<AppIcon name="close"/>
</button>
</div>
</div>
<span class="host-badge">{{research?'网页项目':host.kind==='desktop'?'本地桌面':'浏览器预览'}}</span>
</header>
<button v-if="research" class="project-return" @click="emit('leaveProject')">← 返回项目与文件管理 · {{research.label}}</button>
<main id="main-content" :class="{'pane-workspace':['M01','M02','M03','M09','M10'].includes(active)}" tabindex="-1" role="tabpanel" :aria-labelledby="'tab-'+active">
<div v-if="active==='home'" class="home-page">
<header class="welcome">
<div class="welcome-copy">
<img :src="logo" alt=""/>
<div>
<p class="eyebrow">PHONETIC TOOLBOX 3.0</p>
<h1>从这里开始</h1>
<p>选择一个工具，开始语音分析、合成与实验。</p>
</div>
</div>
<div class="welcome-note">
<span class="status-dot"/>你的语音研究工作台<small>把精力留给声音本身。</small>
</div>
</header>
<div class="section-title">
<h2>全部工具</h2>
<span>15 个工具 · 3 个研究环节</span>
</div>
<div class="tool-groups">
<section v-for="(group,index) in groups" :key="group" class="tool-group" :class="'group-'+index">
<header>
<AppIcon :name="['chart','wave','align'][index]"/>
<h2>{{group}}</h2>
<small>0{{index+1}}</small>
</header>
<button v-for="m in modules.filter(m=>m.group===index)" :key="m.id" @click="open(m.id)">
<AppIcon :name="m.icon"/>
<span>
<strong>{{m.title}}</strong>
<small>{{m.description}}</small>
</span>
<AppIcon name="arrow"/>
</button>
</section>
</div>
<section class="recent-panel">
<div class="section-title">
<h2>
<AppIcon name="clock"/>最近使用</h2>
<span>本机打开的工具</span>
</div>
<div v-if="!recent.length" class="recent-empty">
<AppIcon name="file"/>
<span>还没有最近记录</span>
<small>打开工具后，会在这里留下快捷入口。</small>
</div>
<div v-else class="recent-list">
<button v-for="id in recent" :key="id" @click="open(id)">
<AppIcon :name="modules.find(m=>m.id===id)?.icon"/>{{modules.find(m=>m.id===id)?.title}}<AppIcon name="arrow"/>
</button>
</div>
</section>
<button class="guide-banner" @click="modal='help'">
<AppIcon name="book"/>
<span>第一次使用？<strong>了解工作台的基本操作</strong>
</span>
<span>打开使用说明 →</span>
</button>
<footer class="home-footer">
<span>PhoneticToolbox 3.0 · 界面试用版</span>
<button @click="refs()">开源与学术致谢</button>
</footer>
</div>
<ParameterEstimationPage v-else-if="current?.id==='M01'" :key="researchContext.key" :context="researchContext" @references="refs('M01')"/>
<WorkspaceView v-else-if="current&&!['M02','M03','M09','M10'].includes(current.id)" :key="current.id" :module="current" :state-key="previewKey(current.id)" @references="refs(current?.id)"/>
<ParameterDisplayPage v-if="tabs.includes('M02')" v-show="active==='M02'" ref="m02Page" :key="researchContext.key+':M02'" :active="active==='M02'" :context="researchContext" :state-key="previewKey('M02')" @references="refs('M02')" @close="close('M02')"/>
<EggAnalysisPage v-if="tabs.includes('M03')" v-show="active==='M03'" ref="m03Page" :key="researchContext.key+':M03'" :active="active==='M03'" :context="researchContext" :state-key="previewKey('M03')" @references="refs('M03')" @close="close('M03')"/>
<Spec2WavPage v-if="tabs.includes('M09')" v-show="active==='M09'" :key="researchContext.key+':M09'" :context="researchContext" :state-key="previewKey('M09')" @references="refs('M09')" @close="close('M09')"/>
<VocalTractPage v-if="tabs.includes('M10')" v-show="active==='M10'" :active="active==='M10'" @references="refs('M10')" @close="close('M10')" @dirty="m10Dirty=$event"/>
</main>
<div v-if="current&&!['M03','M10'].includes(current.id)" class="global-transport">
<AudioTransport :state="current.id==='M01'?m01.wave:workspace(previewKey(current.id))" :active="true"/>
</div>
<div class="statusbar">
<span>
<span class="status-dot"/>{{current?.id==='M01'?'参数估计 · 文件、试听与任务':current?.id==='M02'?'参数显示 · 原帧与多图窗':current?.id==='M03'?'EGG · 接触商与声门事件':current?.id==='M09'?'语谱图重建 · 近似相位恢复':current?.id==='M10'?'声道工作台 · VTL 2.4':current?'公共预览就绪 · 分析功能待接入':'就绪 · 选择工具开始'}}</span>
<span>{{research?'当前账号的项目资源':'本机文件 · 无自动上传'}}</span>
</div>
</div>
</div>
<ModalDialog v-if="closing" title="保存参数草稿？" @close="closing=''">
<p>“{{modules.find(m=>m.id===closing)?.title}}”有尚未保存的参数或设置。保存会将参数草稿留在本机；音频不会保存到浏览器存储。</p>
<p v-if="notice" role="alert">{{notice}}</p>
<template #footer>
<button @click="closing=''">取消关闭</button>
<button @click="remove(closing)">放弃草稿并关闭</button>
<button class="primary" @click="saveClose">保存草稿并关闭</button>
</template>
</ModalDialog>
<ModalDialog v-if="modal" :title="modalTitle" :wide="modal==='references'" @close="modal=''">
<template v-if="modal==='settings'">
<h3>外观</h3>
<label class="setting-row">配色主题<select v-model="theme">
<option value="system">跟随系统</option>
<option value="light">浅色</option>
<option value="dark">深色</option>
</select>
</label>
<p class="muted">主题、侧栏宽度与最近工具保存在本机。各工具的参数在工具内单独设置。</p>
<FontSettings/>
</template>
<template v-else-if="modal==='help'">
<h3>工作台的基本操作</h3>
<ol class="help-list">
<li>在左侧搜索工具，或从首页三组入口打开。标签间切换会保留当前文件与参数。</li>
<li>在“参数估计”选择音频目录，网页版先在项目中上传 WAV 与 TextGrid。波形默认显示一个声道，可勾选两个声道；试听声道可单独选择。</li>
<li>拖动波形选区，或输入起止秒数。按 Ctrl 滚轮围绕鼠标缩放，双击恢复全长；普通滚轮滚动当前列，时间窗滑块平移。</li>
<li>点击播放选区；焦点不在控件内时，空格也可播放/暂停。切换标签会停止播放。</li>
<li>“选择输出参数”提供 80 个独立参数键。应用后形成草稿，关闭时可保存、放弃或取消。</li>
</ol>
<h3>参数估计：分析与切分</h3>
<ol class="help-list">
<li>同名 TextGrid 自动关联，也可在音频上方改选。选择层后点击区间即可试听；勾选“显示语谱图（Praat）”查看灰度语谱图。</li>
<li>“开始全列表分析”处理列表里的所有 WAV。先在右列选好 80 项参数和 14 项设置；文件勾选仅控制切分范围。</li>
<li>切分时勾选所需音频，或在文件列按 Ctrl+A 全选，然后选择层并保存。未勾选时只切分当前音频；空白、sil、eps 区间跳过。</li>
<li>勾选“同时切分参数结果”，可使用本应用的同源完整结果，或在切分区逐音频指定历史 XLSX/SQLite。输出标明用户关联、来源未核实；参数沿用原帧，不重新估计。</li>
<li>桌面在旧文件兼容区选择唇形 PKL，点击“转换并保存 .lip.json”。同名伴随时间戳作为起点后备；原文件保留，已有文件不覆盖。网页上传转换后的 .lip.json。</li>
<li>桌面默认保存到 WAV 目录，也可选择独立结果目录；已有不同内容的文件另名保留。网页在处理记录中下载 XLSX、SQLite 和来源 JSON，文件到期前请自行留存。</li>
<li>处理记录显示每个文件的状态。取消保留已完成结果；失败或中断项可单独重试。重开工作台后可查看持久记录，再选择目录保存结果。</li>
</ol>
<p class="notice">长音频可先预览、缩放和切分。当前单次参数估计上限为 200 万采样值（所有声道合计）和 240 秒；显示优化不会改变计算数据。</p>
<h3>参数显示：已有结果与多图窗</h3>
<p>选择 WAV 目录和参数目录，优先关联同名 SQLite，也可手动选择 XLSX。参数沿用原时间和缺失值，读取限 16 MB / 20 万单元格，不执行表内公式。默认所有未校正参数叠加在一个绘图区，用颜色、线型和图例区分；共用纵轴，量级差距较大时沿用 v2 自动双轴。参数图 Ctrl＋滚轮缩放，普通滚轮滚动内容、左键拖动平移、Shift＋拖动选区。</p>
<p>右侧可搜索、勾选多个参数，新建图窗后批量分配。reaper / correction 只筛选候选项；合并图窗保留曲线。时间窗和选区同步，Ctrl+滚轮缩放，波形工具可平移。每张图可放大并保存参数 SVG，也可保存白底300dpi整幅 PNG，包含波形、标注、已开启且完成的语谱图和此参数图。底部播放条试听。</p>
<h3>语谱图转音频：校正与重建</h3>
<p>导入灰度 PNG/JPEG/BMP，桌面也可主动截图。按左上、右上、右下、左下选四点，填写图内时间、频率和灰度标定，设置窗长、迭代和种子后开始重建。网页截图先保存并上传到项目。</p>
<p>结果提供 WAV、校正/重建 PNG 和来源 JSON。固定种子方便复核，非零频率起点采用频带插值并低频补零。输出时长可能因原帧步长取整稍短于标定时长。图像缺少相位，声音仅为近似重建；显示削波样本数，不能视为原录音恢复。</p>
<h3>IPA 字符显示</h3>
<p class="ipa-sample">a ɑ ə ɚ ɤ ɿ ʅ ŋ ɲ ʂ ʐ tʰ ʈʂʰ ˥˩</p>
</template>
<template v-else-if="modal==='update'">
<p>当前版本 {{version.frontend_version}}</p>
<p class="muted">更新服务尚未接入。此试用版没有查询远程版本，也没有自动下载更新。</p>
</template>
<template v-else-if="modal==='about'">
<div class="about-brand">
<img :src="logo" alt="波形团子"/>
<h3>PhoneticToolbox 3.0</h3>
</div>
<p>为语音分析、合成、标注与实验构建的研究工作台。</p>
<p class="muted">界面试用版 · 共用前端与科研核心。各模块的实施范围和引用分别展示。</p>
<button class="primary" @click="refs()">开源与学术致谢</button>
<p>
<button @click="modal='font-license'">Doulos SIL 字体版权与许可</button>
</p>
</template>
<pre v-else-if="modal==='font-license'" class="license-text">{{fontLicense}}</pre>
<MethodReferences v-else-if="modal==='references'" :module-id="referencesId"/>
</ModalDialog>
</template>
