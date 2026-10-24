<script setup lang="ts">
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import AppIcon from '../../components/AppIcon.vue';
import {vResizablePanels} from '../../layout/resizablePanels.ts';
import {computed,nextTick,onBeforeUnmount,onMounted,reactive,ref,shallowRef,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import FontFamilySelect from '../../components/FontFamilySelect.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {candidates,quoteFamily,validFamily} from '../../design/fonts.ts';
import {desktopFontFamilies} from '../../platform/desktop.ts';
import {host} from '../../state/workspace.ts';
import type {MandarinIpaExportSnapshot} from './export.ts';
import PngExportDialog from './PngExportDialog.vue';
import {autoErhuaPairs,convertText,createPageDraft,effectiveIpaSize,moveErhuaPairs,moveVariantChoices,restoreDraft,restoreTableDraft,snapshotDraft,standards,variantKey,variantsFor,type MappedToken} from './state.ts';
import {isUnt} from './unt.ts';
import {resultMetrics} from './layout.ts';
import CustomTableImport from './CustomTableImport.vue';
import TableToneDialog from './TableToneDialog.vue';
import type {ToneSettings} from './tones.ts';
import {addTable,replaceTable,removeTable,CustomTableStore,emptyLibrary,type CustomTable} from './custom-tables.ts';
import type {ConversionToken} from './state.ts';

const props=withDefaults(defineProps<{stateKey?:string}>(),{stateKey:'M13'});
const pngSnapshot=shallowRef<MandarinIpaExportSnapshot>();
const emit=defineEmits<{references:[];help:[];dirty:[value:boolean]}>();
const draftKey=computed(()=>'mandarin-ipa.v1.'+props.stateKey);
const saved=ref(host.projects.read<unknown>(draftKey.value,null)),draft=reactive(createPageDraft(saved.value));
const canRestore=computed(()=>!!restoreDraft(saved.value).text.trim());
const installedFonts=ref<string[]>([]);
const tableStore=new CustomTableStore(props.stateKey),library=ref(emptyLibrary()),tablesReady=ref(false),tableBusy=ref(false),importOpen=ref(false),tableError=ref(''),importError=ref('');
const replaceTarget=ref<CustomTable>(),deleteTarget=ref<CustomTable>(),deleteError=ref(''),tableNotice=ref('');
const toneTarget=ref<CustomTable>(),toneError=ref('');
const customTable=computed(()=>library.value.tables.find(table=>table.id===draft.standard));
const standardName=computed(()=>customTable.value?.name??draft.standard);
let restoringTableSelection=false;
async function loadTables(){
  tableBusy.value=true;tableError.value='';
  try{
    const previous=draft.standard,revision=draft.tableRevision;library.value=await tableStore.load();tablesReady.value=true;restoringTableSelection=true;
    if(library.value.selected)draft.standard=library.value.selected;
    const table=library.value.tables.find(t=>t.id===draft.standard);
    if(table){draft.tableRevision=table.revision??1;if(previous===table.id&&(revision??1)!==draft.tableRevision){draft.selectedVariants={};if(draft.text)dirty.value=true;tableNotice.value='字表已在另一窗口替换，请重新核对读音。';}}
    else if(draft.standard.startsWith('custom:')){draft.standard='Standard Chinese (Beijing)';draft.selectedVariants={};draft.erhuaPairs={};delete draft.tableRevision;if(draft.text)dirty.value=true;tableNotice.value='当前字表已移除，已切回内置标准。';}
  }
  catch(cause){tableError.value=(cause as Error).message;}
  finally{restoringTableSelection=false;tableBusy.value=false;}
}
onMounted(loadTables);
async function selectStandard(value:string){
  if(tableBusy.value)return;
  // Built-in conversion remains available when this device cannot open its table library.
  if((!tablesReady.value||!library.value.selected)&&standards.some(standard=>standard===value)){draft.standard=value as typeof draft.standard;delete draft.tableRevision;tableNotice.value='';return;}
  tableBusy.value=true;tableError.value='';
  const next={...library.value,selected:library.value.tables.find(table=>table.id===value)?.id??null};
  try{await tableStore.save(next);const custom=draft.standard.startsWith('custom:')||value.startsWith('custom:');library.value=next;draft.standard=value as typeof draft.standard;const table=next.tables.find(t=>t.id===value);if(table)draft.tableRevision=table.revision??1;else delete draft.tableRevision;if(custom){draft.selectedVariants={};draft.erhuaPairs={};}tableNotice.value='';}
  catch(cause){tableError.value=(cause as Error).message;}
  finally{tableBusy.value=false;}
}
async function saveTable(table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>){
  tableBusy.value=true;importError.value='';
  try{const next=replaceTarget.value?replaceTable(library.value,replaceTarget.value.id,table):addTable(library.value,{...table,id:`custom:${crypto.randomUUID()}`});await tableStore.save(next);library.value=next;draft.standard=next.selected!;draft.tableRevision=next.tables.find(t=>t.id===next.selected)!.revision??1;draft.selectedVariants={};draft.erhuaPairs={};importOpen.value=false;tableError.value='';tableNotice.value=replaceTarget.value?'字表已替换，请重新核对多音字读音。':'';}
  catch(cause){importError.value=(cause as Error).message;if(/保存失败|已不存在/.test(importError.value))tableError.value=importError.value;}
  finally{tableBusy.value=false;}
}
function openImport(table?:CustomTable){replaceTarget.value=table;importError.value='';importOpen.value=true;}
async function saveTones(tones:ToneSettings){
  const target=toneTarget.value;if(!target||tableBusy.value)return;tableBusy.value=true;toneError.value='';
  try{const next=replaceTable(library.value,target.id,{...target,tones});await tableStore.save(next);library.value=next;draft.tableRevision=next.tables.find(t=>t.id===target.id)!.revision;draft.selectedVariants={};toneTarget.value=undefined;tableError.value='';tableNotice.value='声调映射已保存，请核对转换结果。';}
  catch(cause){toneError.value=(cause as Error).message;tableError.value=toneError.value;}
  finally{tableBusy.value=false;}
}
async function deleteTable(){
  const target=deleteTarget.value;if(!target||tableBusy.value)return;tableBusy.value=true;deleteError.value='';
  try{const next=removeTable(library.value,target.id);await tableStore.save(next);library.value=next;if(draft.standard===target.id){draft.standard='Standard Chinese (Beijing)';draft.selectedVariants={};draft.erhuaPairs={};delete draft.tableRevision;}deleteTarget.value=undefined;tableError.value='';tableNotice.value='字表已删除，当前文本保留。';}
  catch(cause){deleteError.value=(cause as Error).message;tableError.value=deleteError.value;}
  finally{tableBusy.value=false;}
}
const themeColors=ref({ipa:'#a44e2f',hanzi:'#182332'});
let themeObserver:MutationObserver|undefined;
function readThemeColors(){const style=getComputedStyle(document.documentElement);themeColors.value={ipa:style.getPropertyValue('--accent').trim(),hanzi:style.getPropertyValue('--text').trim()};}
const fontNames=computed(()=>installedFonts.value.length?installedFonts.value:candidates.zh);
const fontLabels={SimSun:'宋体',KaiTi:'楷体',FangSong:'仿宋','Microsoft YaHei':'微软雅黑'};
onMounted(async()=>{readThemeColors();themeObserver=new MutationObserver(()=>{readThemeColors();scheduleCellMeasure();});themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','data-palette','style']});if(desktopFontFamilies)try{installedFonts.value=[...new Set((await desktopFontFamilies()).map(name=>name.trim()).filter(name=>name&&validFamily(name)))];}catch{/* Common names and custom input remain available. */}});
const dirty=ref(false),error=ref(''),exporting=ref(false),outputArea=ref<HTMLElement>(),activeVariant=ref<{char:string;index:number}|null>(null);
const tokens=computed(()=>convertText(draft.text,draft.standard,draft.selectedVariants,draft.showTone,draft.erhuaPairs,customTable.value));
const paragraphs=computed(()=>{
  const rows:ConversionToken[][]=[[]];
  for(const token of tokens.value){if(token.kind==='literal'&&token.newline)rows.push([]);else rows.at(-1)!.push(token);}
  return rows;
});
const cellWidth=ref(0),metrics=computed(()=>resultMetrics(draft));
let measureFrame=0,outputResize:ResizeObserver|undefined;
function measureCells(){
  const area=outputArea.value;if(!area)return;
  const zoom=area.getBoundingClientRect().width/area.offsetWidth||1;
  let width=0;
  for(const cell of area.querySelectorAll<HTMLElement>('.m13-token')){
    const style=getComputedStyle(cell),padding=parseFloat(style.paddingLeft)+parseFloat(style.paddingRight);
    const textWidth=Math.max(0,...[...cell.children].map(child=>child.getBoundingClientRect().width/zoom));
    width=Math.max(width,textWidth+padding);
  }
  cellWidth.value=Math.ceil(width);
}
function scheduleCellMeasure(){cancelAnimationFrame(measureFrame);measureFrame=requestAnimationFrame(measureCells);}
watch(()=>[tokens.value,draft.display,draft.hanziSize,draft.ipaSize,draft.ipaSizeUserSet,draft.bold,draft.italic,draft.hanziFont,draft.equalWidth],scheduleCellMeasure,{flush:'post'});
onMounted(()=>{
  outputResize=new ResizeObserver(scheduleCellMeasure);if(outputArea.value)outputResize.observe(outputArea.value);
  document.fonts.addEventListener('loadingdone',scheduleCellMeasure);
  void document.fonts.ready.then(scheduleCellMeasure);
});
const untEnabled=computed(()=>isUnt(draft.standard));
const erhuaCount=computed(()=>tokens.value.filter(token=>token.kind==='mapped'&&token.erhua).length);
const canAutoErhua=computed(()=>untEnabled.value&&Object.keys(autoErhuaPairs(draft.text,draft.standard,draft.selectedVariants)).length>0);
const selectedToken=computed(()=>activeVariant.value?tokens.value.find((token):token is MappedToken=>token.kind==='mapped'&&token.char===activeVariant.value?.char&&token.index===activeVariant.value?.index):undefined);
const options=computed(()=>selectedToken.value?variantsFor(selectedToken.value,draft.standard,draft.showTone):[]);
const outputStyle=computed(()=>({'--m13-hanzi-size':draft.hanziSize+'px','--m13-ipa-size':effectiveIpaSize(draft)+'px','--m13-gap':draft.gap+'px','--m13-row-gap':metrics.value.rowGap+'px','--m13-unit-height':metrics.value.unitHeight+'px','--m13-cell-width':draft.equalWidth&&cellWidth.value?cellWidth.value+'px':'auto','--m13-ipa-color':draft.ipaColor||'var(--accent)','--m13-hanzi-color':draft.hanziColor||'var(--text)','--m13-hanzi-font':draft.hanziFont&&validFamily(draft.hanziFont)?quoteFamily(draft.hanziFont)+', var(--font)':'var(--font)'}));
const hanziStyle=computed(()=>({fontSize:draft.hanziSize+'px',fontWeight:draft.bold?'700':'400',fontStyle:draft.italic?'italic':'normal',textDecoration:draft.underline?'underline':'none'}));

watch(draft,()=>{if(!restoringTableSelection)dirty.value=true;error.value='';},{deep:true,flush:'sync'});
watch(dirty,value=>emit('dirty',value),{immediate:true});
watch(()=>draft.text,(next,previous)=>{draft.selectedVariants=moveVariantChoices(previous,next,draft.selectedVariants);draft.erhuaPairs=moveErhuaPairs(previous,next,draft.erhuaPairs);},{flush:'sync'});
const variantPanel=ref<HTMLElement>(),variantPosition=ref({left:'0px',top:'0px',visibility:'hidden' as 'hidden'|'visible'});
let variantAnchor:HTMLElement|undefined,variantFrame=0,variantResize:ResizeObserver|undefined,variantZoom:MutationObserver|undefined;
function positionVariants(){
  const panel=variantPanel.value,anchor=variantAnchor;
  if(!panel||!anchor?.isConnected)return;
  const rect=anchor.getBoundingClientRect(),box=panel.getBoundingClientRect();
  // Fixed offsets inherit root zoom; DOM rectangles are viewport coordinates.
  const zoom=box.width/panel.offsetWidth||1,margin=8,gap=8;
  const width=window.innerWidth,height=window.innerHeight;
  const clip=outputArea.value?.getBoundingClientRect();
  if(!rect.width||rect.bottom<0||rect.top>height||(clip&&(rect.bottom<clip.top||rect.top>clip.bottom))){closeVariants();return;}
  let left=rect.right+gap,top=rect.top;
  if(left+box.width>width-margin)left=rect.left-box.width-gap;
  if(left<margin){left=rect.left;top=rect.bottom+gap;if(top+box.height>height-margin)top=rect.top-box.height-gap;}
  variantPosition.value={left:Math.max(margin,Math.min(left,width-box.width-margin))/zoom+'px',top:Math.max(margin,Math.min(top,height-box.height-margin))/zoom+'px',visibility:'visible'};
}
function scheduleVariantPosition(){cancelAnimationFrame(variantFrame);variantFrame=requestAnimationFrame(positionVariants);}
function outsideVariants(event:PointerEvent){const target=event.target as Node;if(!variantPanel.value?.contains(target)&&!variantAnchor?.contains(target))closeVariants();}
function variantKeydown(event:KeyboardEvent){if(event.key==='Escape'){event.preventDefault();event.stopPropagation();closeVariants(true);}}
function closeVariants(restoreFocus=false){
  activeVariant.value=null;cancelAnimationFrame(variantFrame);
  variantResize?.disconnect();variantZoom?.disconnect();
  window.removeEventListener('scroll',scheduleVariantPosition,true);window.removeEventListener('resize',scheduleVariantPosition);
  document.removeEventListener('pointerdown',outsideVariants,true);document.removeEventListener('keydown',variantKeydown,true);
  if(restoreFocus)variantAnchor?.focus({preventScroll:true});
  variantAnchor=undefined;
}
async function openVariants(token:MappedToken,event:MouseEvent){
  closeVariants();if(token.variants.length<2&&!token.canCombine&&!token.erhua)return;
  variantAnchor=event.currentTarget as HTMLElement;activeVariant.value={char:token.char,index:token.index};
  variantPosition.value={left:'0px',top:'0px',visibility:'hidden'};
  await nextTick();positionVariants();
  if(!activeVariant.value)return;
  variantResize=new ResizeObserver(scheduleVariantPosition);variantResize.observe(variantAnchor!);variantResize.observe(outputArea.value!);
  variantZoom=new MutationObserver(scheduleVariantPosition);variantZoom.observe(document.documentElement,{attributes:true,attributeFilter:['style']});
  variantPanel.value?.querySelector<HTMLButtonElement>('button[aria-pressed=true]')?.focus({preventScroll:true});
  window.addEventListener('scroll',scheduleVariantPosition,true);window.addEventListener('resize',scheduleVariantPosition);
  document.addEventListener('pointerdown',outsideVariants,true);document.addEventListener('keydown',variantKeydown,true);
}
function chooseVariant(token:MappedToken,index:number){draft.selectedVariants[variantKey(token.baseChar??token.char,token.index)]=index;closeVariants(true);}
function combineErhua(){draft.erhuaPairs=autoErhuaPairs(draft.text,draft.standard,draft.selectedVariants);closeVariants();}
function toggleErhua(token:MappedToken){
  const key=variantKey(token.baseChar??token.char,token.index);
  if(token.erhua)delete draft.erhuaPairs[key];else draft.erhuaPairs[key]=true;
  closeVariants();
}
function splitAllErhua(){draft.erhuaPairs={};closeVariants();}
watch(()=>[draft.text,draft.layout,draft.display,draft.standard],()=>closeVariants());
onBeforeUnmount(()=>{closeVariants();themeObserver?.disconnect();outputResize?.disconnect();cancelAnimationFrame(measureFrame);document.fonts.removeEventListener('loadingdone',scheduleCellMeasure);});
function save(){const snapshot=snapshotDraft(draft),ok=host.projects.write(draftKey.value,snapshot);if(ok){saved.value=snapshot;dirty.value=false;error.value='';}else error.value='草稿保存失败，当前文本和排版仍保留在页面中。';return ok;}
async function loadSaved(){if(dirty.value||!canRestore.value)return;const restored=restoreTableDraft(saved.value,library.value,draft.standard);delete draft.tableRevision;Object.assign(draft,restored.draft);await nextTick();dirty.value=!!restored.notice;tableNotice.value=restored.notice;error.value='';}
function exportImage(){
  if(exporting.value)return;error.value='';closeVariants();
  const element=outputArea.value,style=element?getComputedStyle(element):getComputedStyle(document.documentElement);measureCells();
  pngSnapshot.value={tokens:tokens.value,draft:snapshotDraft(draft),standardName:standardName.value,width:element?.clientWidth??900,cellWidth:draft.equalWidth?cellWidth.value:undefined,uiFont:style.getPropertyValue('--font').trim()||style.fontFamily,ipaColor:style.getPropertyValue('--m13-ipa-color').trim()};
}
defineExpose({save});
// Custom table identity belongs to this device's library; account settings keep built-in choices.
const builtinStandard=ref(draft.standard);
watch(()=>draft.standard,value=>{if(!value.startsWith('custom:'))builtinStandard.value=value;},{flush:'sync'});
const accountStandard=computed({get:()=>builtinStandard.value,set:(value:string)=>{if(standards.some(s=>s===value)){builtinStandard.value=value as typeof draft.standard;if(!draft.standard.startsWith('custom:'))draft.standard=value as typeof draft.standard;}}});
useWebSettings(props.stateKey,{standard:accountStandard,...settingsFields(draft,['display','layout','hanziSize','ipaSize','ipaSizeUserSet','gap','lineHeight','bold','italic','underline','showTone','ipaColor','hanziColor','hanziFont','equalWidth'])});
</script>

<template>
<ModuleFrame unified fit class="mandarin-ipa-page" label="汉字转国际音标 工作区">
  <template #toolbar><ModuleToolbar label="汉字转国际音标 操作">
        <button class="primary" :disabled="exporting||!draft.text.trim()" @click="exportImage">{{exporting?'正在生成…':'保存为 PNG'}}</button>
        <template #actions><button v-if="canRestore" :disabled="dirty" @click="loadSaved">恢复本机草稿</button><button class="primary" :disabled="!dirty" @click="save">保存本机草稿<span v-if="dirty" aria-label="未保存"> *</span></button>
        <button type="button" @click="emit('help')">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button>
      </template></ModuleToolbar></template>
  <template #status>
    <ModuleStatus v-if="error" kind="error" :message="error"><button @click="error=''">收起提示</button></ModuleStatus>
    <ModuleStatus v-else-if="exporting" kind="loading" message="正在用本地 Doulos SIL 字体生成 PNG…"/>
  </template>

  <div v-resizable-panels="{key:stateKey,center:'.m13-result-section',centerMin:240,panels:[{selector:'.m13-input-section',side:'left',label:'汉字输入',initial:300,min:240,max:1600,initialFractionOfRemaining:0.5},{selector:'.m13-settings-section',side:'left',variable:'--panel-right',label:'转换与排版',initial:300,min:240,max:560}]}" :key="draft.layout" class="m13-workspace" :class="{'m13-workspace-stacked':draft.layout==='stacked'}">
    <ModuleSection class="m13-settings-section" label="转换和排版设置" title="转换与排版">

      <p class="m13-local-note">本地转换 · 文本不上传</p><p class="m13-mapping-note">多音字需人工选读音，不处理语流变调。</p>

      <label>转换标准<select :value="draft.standard" aria-label="转换标准" :disabled="tableBusy" @change="selectStandard(($event.target as HTMLSelectElement).value)"><option v-for="standard in standards" :key="standard" :value="standard">{{standard}}</option><optgroup v-if="library.tables.length" label="自定义字表"><option v-for="table in library.tables" :key="table.id" :value="table.id">{{table.name}}</option></optgroup></select></label>
      <div class="m13-table-actions"><button :disabled="!tablesReady||tableBusy" @click="openImport()">导入自定义字表</button><template v-if="customTable"><button :disabled="tableBusy" @click="toneError='';toneTarget=customTable">声调映射</button><button :disabled="tableBusy" @click="openImport(customTable)">替换字表</button><button :disabled="tableBusy" @click="deleteError='';deleteTarget=customTable">删除字表</button><span>{{customTable.kind==='hanzi'?'汉字表':'音节表'}} · {{customTable.rows.length.toLocaleString()}} 条</span></template></div>
      <ModuleStatus v-if="tableError" kind="error" :message="tableError"><button :disabled="tableBusy" @click="loadTables">重新载入已存字表</button></ModuleStatus>
      <ModuleStatus v-if="tableNotice" kind="info" :message="tableNotice"/>
      <fieldset v-if="untEnabled" class="m13-erhua"><legend>儿化组合</legend><button :disabled="!canAutoErhua" @click="combineErhua">自动组合儿化</button><button :disabled="!erhuaCount" @click="splitAllErhua">全部拆分</button>
        <p>已组合 {{erhuaCount}} 处。点击字＋儿可手动组合或拆分；自动组合后请核对独立的儿字。</p>
      </fieldset>
      <fieldset><legend>显示内容</legend><label><input v-model="draft.display" type="radio" value="paired"/>字音同显</label><label><input v-model="draft.display" type="radio" value="ipa-only"/>仅音标</label><button class="m13-tone-toggle" :aria-pressed="draft.showTone" @click="draft.showTone=!draft.showTone">显示声调</button></fieldset>
      <fieldset><legend>输入与结果排布</legend><label><input v-model="draft.layout" type="radio" value="side-by-side"/>左右排布</label><label><input v-model="draft.layout" type="radio" value="stacked"/>上下排布</label></fieldset>
      <fieldset><legend>结果对齐</legend><button :aria-pressed="draft.equalWidth" @click="draft.equalWidth=!draft.equalWidth">等宽排版</button></fieldset>
      <label>汉字字号 <span>{{draft.hanziSize}} px</span><input v-model.number="draft.hanziSize" aria-label="汉字字号" type="range" min="16" max="72"/></label>
      <label>音标字号 <span>{{effectiveIpaSize(draft)}} px</span><input :value="effectiveIpaSize(draft)" aria-label="音标字号" type="range" min="12" max="48" @input="draft.ipaSize=Number(($event.target as HTMLInputElement).value);draft.ipaSizeUserSet=true"/></label>
      <label>字音间距 <span>{{draft.gap}} px</span><input v-model.number="draft.gap" aria-label="字音间距" type="range" min="-12" max="20"/></label>
      <label>行距 <span>{{draft.lineHeight.toFixed(1)}}</span><input :value="Math.round(draft.lineHeight*10)" aria-label="行距" type="range" min="8" max="30" @input="draft.lineHeight=Number(($event.target as HTMLInputElement).value)/10"/></label>
      <fieldset><legend>参考汉字样式</legend><button :aria-pressed="draft.bold" @click="draft.bold=!draft.bold"><strong>B</strong> 粗体</button><button :aria-pressed="draft.italic" @click="draft.italic=!draft.italic"><em>I</em> 斜体</button><button :aria-pressed="draft.underline" @click="draft.underline=!draft.underline"><u>U</u> 下划线</button></fieldset>
      <fieldset class="m13-colors"><legend>文字颜色</legend><label><input type="color" aria-label="音标颜色" :value="draft.ipaColor||themeColors.ipa" @input="draft.ipaColor=($event.target as HTMLInputElement).value"/>音标颜色</label><label><input type="color" aria-label="汉字颜色" :value="draft.hanziColor||themeColors.hanzi" @input="draft.hanziColor=($event.target as HTMLInputElement).value"/>汉字颜色</label><button :disabled="!draft.ipaColor&&!draft.hanziColor" @click="draft.ipaColor='';draft.hanziColor=''">跟随主题</button></fieldset>
      <FontFamilySelect v-model="draft.hanziFont" label="汉字字体" :names="fontNames" :labels="fontLabels"/>
      <p class="m13-font-note">音标字体固定为 Doulos SIL</p>
    </ModuleSection>

    <ModuleSection class="m13-input-section" label="汉字输入" title="汉字输入">
      <template #actions><span class="m13-count">{{[...draft.text].length.toLocaleString()}} 个字符</span></template>
      <textarea v-model="draft.text" aria-label="待转换汉字文本" placeholder="输入汉字、标点或分行文本…" spellcheck="false"/>
    </ModuleSection>

    <ModuleSection class="m13-result-section" label="转换结果" title="转换结果">
      <div ref="outputArea" class="m13-output" :class="{'m13-paired':draft.display==='paired'}" :style="outputStyle" :data-standard="draft.standard">
        <ModuleStatus v-if="!draft.text.trim()" kind="empty" message="输入文本后，这里会实时显示转换结果。"/>
        <template v-else><div v-for="(paragraph,index) in paragraphs" :key="index" class="m13-paragraph">
        <template v-for="token in paragraph" :key="token.index">
          <span v-if="token.kind==='literal'" class="m13-token m13-literal" :class="{'m13-ipa-only-literal':draft.display==='ipa-only'}" :data-index="token.index">
            <span v-if="draft.display==='paired'" class="m13-ipa-placeholder" aria-hidden="true">&nbsp;</span><span class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
          <button v-else-if="token.variants.length>1||token.canCombine||token.erhua" type="button" class="m13-token m13-mapped m13-ambiguous" :class="{'m13-erhua-group':token.erhua}" :data-erhua="token.erhua||undefined" :data-index="token.index" :data-value="token.value" :aria-label="`${token.char}：${token.variants.length} 个读音，当前 ${token.value}${token.erhua?'，可拆分儿化':token.canCombine?'，可组合儿化':''}`" aria-haspopup="dialog" :aria-expanded="activeVariant?.index===token.index" @click="openVariants(token,$event)">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </button>
          <span v-else class="m13-token m13-mapped" :data-index="token.index" :data-value="token.value">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
        </template>
        </div></template>
      </div>
      <Teleport to="body">
      <div v-if="selectedToken" ref="variantPanel" class="m13-variants" :style="variantPosition" role="dialog" :aria-label="'选择 '+selectedToken.char+' 的读音'">
        <div><strong>选择读音：{{selectedToken.char}}</strong><button aria-label="关闭读音选择" @click="closeVariants(true)">×</button></div>
        <button v-if="untEnabled&&(selectedToken.erhua||selectedToken.canCombine)" class="m13-erhua-action" @click="toggleErhua(selectedToken)">{{selectedToken.erhua?'拆分儿化':'组合儿化'}}</button>
        <button v-for="option in options" :key="option.index" :aria-pressed="selectedToken.selectedVariant===option.index" @click="chooseVariant(selectedToken,option.index)">
          <span>{{option.pinyin?option.pinyin+(option.toneLabel==='轻声'?'（轻声）':option.toneLabel):'读音 '+(option.index+1)}}</span><span class="ipa-text">{{option.value}}</span>
        </button>
      </div>
      </Teleport>
    </ModuleSection>


  </div>
  <CustomTableImport v-if="importOpen" :saving="tableBusy" :save-error="importError" :replace-target="replaceTarget" @close="importOpen=false" @save="saveTable"/>
  <PngExportDialog v-if="pngSnapshot" :snapshot="pngSnapshot" @close="pngSnapshot=undefined" @busy="exporting=$event"/>
  <TableToneDialog v-if="toneTarget" :table="toneTarget" :saving="tableBusy" :save-error="toneError" @close="toneTarget=undefined" @save="saveTones"/>
  <ModalDialog v-if="deleteTarget" title="删除自定义字表" variant="confirm" :close-disabled="tableBusy" @close="deleteTarget=undefined">
    <p>确定删除字表 <strong>{{deleteTarget.name}}</strong>？</p>
    <p>删除后会从转换标准中移除。当前文本保留，使用该字表的草稿需重新选择字表。</p>
    <ModuleStatus v-if="deleteError" kind="error" :message="deleteError"/>
    <template #footer><button :disabled="tableBusy" @click="deleteTarget=undefined">取消</button><button class="danger" :disabled="tableBusy" @click="deleteTable">{{tableBusy?'正在删除…':'删除字表'}}</button></template>
  </ModalDialog>
</ModuleFrame>
</template>

<style scoped>
.mandarin-ipa-page{height:100%;overflow:auto}.m13-local-note{color:var(--muted);font-size:var(--support-size)}
.m13-workspace{display:grid;grid-template-columns:var(--panel-right,300px) var(--panel-left,300px) minmax(240px,1fr);grid-template-areas:'settings input result';gap:var(--module-gap);align-items:stretch;min-height:0}.m13-input-section{grid-area:input}.m13-result-section{grid-area:result}.m13-settings-section{grid-area:settings}.m13-workspace.m13-workspace-stacked{grid-template-columns:var(--panel-right,300px) minmax(320px,1fr);grid-template-areas:'settings input' 'settings result'}.m13-workspace-stacked .m13-input-section textarea{min-height:180px}.m13-workspace-stacked .m13-output{min-height:260px}.m13-local-note{margin:10px 0 14px}
.m13-input-section,.m13-result-section,.m13-settings-section{display:flex;flex-direction:column}.m13-input-section textarea{flex:1;width:100%;min-height:360px;resize:vertical;padding:12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);font:2rem/1.8 var(--font);overflow-wrap:anywhere}.m13-count{color:var(--muted);font-size:var(--support-size);text-align:right;white-space:nowrap}
.m13-output{flex:1;min-height:360px;padding:16px;border:1px solid var(--border);border-radius:6px;background:var(--app);color:var(--text);overflow:auto}
.m13-output :deep(.module-status-empty){font:2rem/1.8 var(--font)}
.m13-output :deep(.module-status-empty p){font:inherit}
.m13-paragraph{display:flex;flex-wrap:wrap;align-items:flex-start;align-content:flex-start;gap:var(--m13-row-gap) 4px;min-height:var(--m13-unit-height)}.m13-paragraph+.m13-paragraph{margin-top:var(--m13-row-gap)}
.m13-token{display:inline-flex;flex:none;flex-direction:column;align-items:center;position:relative;width:var(--m13-cell-width);margin:0;padding:2px 4px;border-radius:4px;gap:0;line-height:1;white-space:pre}.m13-ipa{font-family:var(--font-ipa,"PTB-Doulos"),serif;font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0;overflow:visible;color:var(--accent)}.m13-hanzi{font-family:var(--font);line-height:1.2;margin-top:var(--m13-gap)}.m13-ipa-placeholder{font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0}.m13-ipa-only-literal .m13-hanzi{font-size:var(--m13-ipa-size)!important;font-weight:400!important;font-style:normal!important;text-decoration:none!important;margin:0;line-height:1.59}
.m13-ambiguous{min-height:0;border:0;background:transparent;color:inherit}.m13-ambiguous:hover{background:var(--selected)}.m13-ambiguous:after{content:'▼';position:absolute;right:0;bottom:0;color:var(--accent);font-size:0.571429rem;line-height:1}.m13-mapped:not(button):hover{background:var(--selected)}
.m13-variants{position:fixed;z-index:100;width:240px;height:240px;max-width:calc(100vw / var(--page-scale,1) - 16px);max-height:calc(100dvh / var(--page-scale,1) - 16px);overflow:auto;padding:10px;border:1px solid var(--accent);border-radius:var(--radius);background:var(--panel);color:var(--text);font-family:var(--font);font-size:var(--control-size);box-shadow:var(--shadow)}.m13-variants>div{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;position:sticky;top:-10px;background:var(--panel)}.m13-variants>div button{padding:2px 8px;min-height:28px;font-size:1.428571rem}.m13-variants>button{display:flex;width:100%;justify-content:space-between;margin-top:6px;padding:6px 8px;white-space:normal;overflow-wrap:anywhere}.m13-variants>button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}.m13-variants .ipa-text{font-size:1.285714rem;color:var(--accent)}
.m13-settings-section>label:not(.font-family-select){display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 10px;align-items:center;margin-bottom:12px;font-size:var(--control-size)}.m13-settings-section>label:not(.font-family-select) select,.m13-settings-section>label:not(.font-family-select) input{grid-column:1/-1;width:100%}.m13-settings-section>label:not(.font-family-select) span{font-variant-numeric:tabular-nums;color:var(--muted)}fieldset{display:flex;flex-wrap:wrap;gap:8px 12px;margin:0 0 14px;padding:10px;border:1px solid var(--border);border-radius:6px}legend{padding:0 4px;font-size:var(--support-size);color:var(--muted)}fieldset label{display:flex;align-items:center;gap:5px}fieldset button{font-size:var(--support-size)}fieldset button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}

.m13-workspace{flex:1;min-height:0;grid-template-rows:minmax(0,1fr)}
.m13-input-section,.m13-result-section,.m13-settings-section{min-height:0;overflow:auto;overscroll-behavior:contain}
.m13-input-section,.m13-settings-section{background:var(--rail,var(--app));scrollbar-gutter:stable}.m13-input-section textarea,.m13-output{min-height:160px}
.m13-workspace.m13-workspace-stacked{grid-template-rows:minmax(220px,2fr) minmax(280px,3fr)}
.m13-workspace-stacked .m13-input-section textarea,.m13-workspace-stacked .m13-output{min-height:100px}
.m13-settings-section{gap:0;padding:10px}
.m13-settings-section :deep(.module-section-heading){margin-bottom:4px}
.m13-settings-section .m13-local-note{margin:0 0 4px;line-height:1.4}
.m13-mapping-note,.m13-font-note{margin:0 0 8px;color:var(--muted);font-size:var(--support-size);line-height:1.5}
.m13-font-note{margin:4px 0 0}
.m13-settings-section>label:not(.font-family-select){gap:2px 8px;margin-bottom:6px}
.m13-settings-section>label:not(.font-family-select)>select{min-height:30px;height:30px;padding:3px 6px}
.m13-settings-section>label:not(.font-family-select)>input[type=range]{height:18px;min-height:18px;margin:0}
.m13-settings-section fieldset{gap:4px 8px;margin-bottom:6px;padding:5px 7px}
.m13-settings-section fieldset button{min-height:28px;padding:3px 7px}
.m13-erhua p{flex-basis:100%;margin:2px 0;color:var(--muted);font-size:var(--support-size);line-height:1.5}
.m13-erhua-group{box-shadow:inset 0 -2px var(--accent)}
.m13-settings-section .m13-tone-toggle{margin-left:auto}
.m13-table-actions{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin-bottom:6px}.m13-table-actions button,.m13-table-actions span{font-size:var(--support-size)}.m13-table-actions span{color:var(--muted)}
.m13-settings-section .m13-colors{gap:5px 8px}
.m13-colors input[type=color]{width:26px;height:26px;min-height:26px;padding:2px;border-radius:4px;cursor:pointer}
.m13-colors button{margin-left:auto}
.m13-settings-section>.font-family-select{display:flex;flex-direction:column;align-items:stretch;gap:2px;margin-bottom:0}
.m13-settings-section :deep(.font-family-select select),.m13-settings-section :deep(.font-family-select input){height:max(30px,calc(var(--control-size)*1.5 + 8px));min-height:30px;padding:3px 6px}
.m13-hanzi{font-family:var(--m13-hanzi-font,var(--font));color:var(--m13-hanzi-color,var(--text))}
.m13-ipa{color:var(--m13-ipa-color,var(--accent))}
</style>
