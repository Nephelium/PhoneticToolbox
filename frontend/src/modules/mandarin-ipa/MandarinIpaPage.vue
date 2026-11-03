<script setup lang="ts">
import {uiErrorMessage as displayErrorValue} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,message as localizedMessage,messageRef} from '../../i18n/index.ts';
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
import {addTable,replaceTable,removeTable,tableErrorNeedsReload,CustomTableStore,emptyLibrary,type CustomTable} from './custom-tables.ts';
import type {ConversionToken} from './state.ts';

const props=withDefaults(defineProps<{stateKey?:string}>(),{stateKey:'M13'});
const pngSnapshot=shallowRef<MandarinIpaExportSnapshot>();
const emit=defineEmits<{references:[];help:[];dirty:[value:boolean]}>();
const draftKey=computed(()=>'mandarin-ipa.v1.'+props.stateKey);
const saved=ref(host.projects.read<unknown>(draftKey.value,null)),draft=reactive(createPageDraft(saved.value));
const canRestore=computed(()=>!!restoreDraft(saved.value).text.trim());
const installedFonts=ref<string[]>([]);
const tableStore=new CustomTableStore(props.stateKey),library=ref(emptyLibrary()),tablesReady=ref(false),tableBusy=ref(false),importOpen=ref(false),tableError=messageRef(),importError=messageRef();
const replaceTarget=ref<CustomTable>(),deleteTarget=ref<CustomTable>(),deleteError=messageRef(),tableNotice=messageRef();
const toneTarget=ref<CustomTable>(),toneError=messageRef();
const customTable=computed(()=>library.value.tables.find(table=>table.id===draft.standard));
const standardName=computed(()=>customTable.value?.name??draft.standard);
let restoringTableSelection=false;
async function loadTables(){
  tableBusy.value=true;tableError.value='';
  try{
    const previous=draft.standard,revision=draft.tableRevision;library.value=await tableStore.load();tablesReady.value=true;restoringTableSelection=true;
    if(library.value.selected)draft.standard=library.value.selected;
    const table=library.value.tables.find(t=>t.id===draft.standard);
    if(table){draft.tableRevision=table.revision??1;if(previous===table.id&&(revision??1)!==draft.tableRevision){draft.selectedVariants={};if(draft.text)dirty.value=true;tableNotice.value=localizedMessage('modules.mandarin-ipa.text.6574a1e63fa4');}}
    else if(draft.standard.startsWith('custom:')){draft.standard='Standard Chinese (Beijing)';draft.selectedVariants={};draft.erhuaPairs={};delete draft.tableRevision;if(draft.text)dirty.value=true;tableNotice.value=localizedMessage('modules.mandarin-ipa.text.05e853f4ab5c');}
  }
  catch(cause){tableError.value=displayErrorValue(cause);}
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
  catch(cause){tableError.value=displayErrorValue(cause);}
  finally{tableBusy.value=false;}
}
async function saveTable(table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>){
  tableBusy.value=true;importError.value='';
  try{const next=replaceTarget.value?replaceTable(library.value,replaceTarget.value.id,table):addTable(library.value,{...table,id:`custom:${crypto.randomUUID()}`});await tableStore.save(next);library.value=next;draft.standard=next.selected!;draft.tableRevision=next.tables.find(t=>t.id===next.selected)!.revision??1;draft.selectedVariants={};draft.erhuaPairs={};importOpen.value=false;tableError.value='';tableNotice.value=replaceTarget.value?localizedMessage('modules.mandarin-ipa.text.fb5eb7cf230f'):'';}
  catch(cause){importError.value=displayErrorValue(cause);if(tableErrorNeedsReload(cause))tableError.value=displayErrorValue(cause);}
  finally{tableBusy.value=false;}
}
function openImport(table?:CustomTable){replaceTarget.value=table;importError.value='';importOpen.value=true;}
async function saveTones(tones:ToneSettings){
  const target=toneTarget.value;if(!target||tableBusy.value)return;tableBusy.value=true;toneError.value='';
  try{const next=replaceTable(library.value,target.id,{...target,tones});await tableStore.save(next);library.value=next;draft.tableRevision=next.tables.find(t=>t.id===target.id)!.revision;draft.selectedVariants={};toneTarget.value=undefined;tableError.value='';tableNotice.value=localizedMessage('modules.mandarin-ipa.text.4cb60a89ed39');}
  catch(cause){const failure=displayErrorValue(cause);toneError.value=failure;tableError.value=failure;}
  finally{tableBusy.value=false;}
}
async function deleteTable(){
  const target=deleteTarget.value;if(!target||tableBusy.value)return;tableBusy.value=true;deleteError.value='';
  try{const next=removeTable(library.value,target.id);await tableStore.save(next);library.value=next;if(draft.standard===target.id){draft.standard='Standard Chinese (Beijing)';draft.selectedVariants={};draft.erhuaPairs={};delete draft.tableRevision;}deleteTarget.value=undefined;tableError.value='';tableNotice.value=localizedMessage('modules.mandarin-ipa.text.5ed5fad64d43');}
  catch(cause){const failure=displayErrorValue(cause);deleteError.value=failure;tableError.value=failure;}
  finally{tableBusy.value=false;}
}
const themeColors=ref({ipa:'#a44e2f',hanzi:'#182332'});
let themeObserver:MutationObserver|undefined;
function readThemeColors(){const style=getComputedStyle(document.documentElement);themeColors.value={ipa:style.getPropertyValue('--accent').trim(),hanzi:style.getPropertyValue('--text').trim()};}
const fontNames=computed(()=>installedFonts.value.length?installedFonts.value:candidates.zh);
function standardLabel(value:string){return tr(standardKeys[value]??'modules.mandarin-ipa.standard.beijing');}
const standardKeys:Record<string,string>={"Standard Chinese (Beijing)严": "modules.mandarin-ipa.text.338898d4c079", "胡裕树《现代汉语》": "modules.mandarin-ipa.text.bbd94b1a8792", "黄伯荣、廖序东《现代汉语》": "modules.mandarin-ipa.text.ffd92bff3ca6", "钱乃荣《现代汉语》": "modules.mandarin-ipa.text.12dc3c155c74", "吴宗济": "modules.mandarin-ipa.text.c8591479f203", "赵元任《汉语口语语法》": "modules.mandarin-ipa.text.35c4e10d6d24", "《汉语方音字汇》": "modules.mandarin-ipa.text.e82f3449d03f", "UntPhesoca宽": "modules.mandarin-ipa.text.80c37b8ec983", "UntPhesoca严": "modules.mandarin-ipa.text.50befcde04d0", "汉语拼音": "modules.mandarin-ipa.text.e10b52fff358"};
const fontLabels=localized(()=>({SimSun:tr('modules.mandarin-ipa.text.bcce34b035bf'),KaiTi:tr('modules.mandarin-ipa.text.bba1d7806df9'),FangSong:tr('modules.mandarin-ipa.text.0e8f151436f2'),'Microsoft YaHei':tr('modules.mandarin-ipa.text.f18e498de92f')}));
onMounted(async()=>{readThemeColors();themeObserver=new MutationObserver(()=>{readThemeColors();scheduleCellMeasure();});themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','data-palette','style']});if(desktopFontFamilies)try{installedFonts.value=[...new Set((await desktopFontFamilies()).map(name=>name.trim()).filter(name=>name&&validFamily(name)))];}catch{/* Common names and custom input remain available. */}});
const dirty=ref(false),error=messageRef(),exporting=ref(false),outputArea=ref<HTMLElement>(),activeVariant=ref<{char:string;index:number}|null>(null);
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
function save(){const snapshot=snapshotDraft(draft),ok=host.projects.write(draftKey.value,snapshot);if(ok){saved.value=snapshot;dirty.value=false;error.value='';}else error.value=localizedMessage('modules.mandarin-ipa.text.5685e059e8d0');return ok;}
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
<ModuleFrame unified fit class="mandarin-ipa-page" :label="tr('modules.mandarin-ipa.text.70f891a7d8e3')">
  <template #toolbar><ModuleToolbar :label="tr('modules.mandarin-ipa.text.49ab3aef27c5')">
        <button class="primary" :disabled="exporting||!draft.text.trim()" @click="exportImage">{{exporting?tr('modules.mandarin-ipa.text.a8b8730c9611'):tr('modules.mandarin-ipa.text.da64ad7328e0')}}</button>
        <template #actions><button v-if="canRestore" :disabled="dirty" @click="loadSaved">{{tr('modules.mandarin-ipa.text.7fb4b2181a0b')}}</button><button class="primary" :disabled="!dirty" @click="save">{{tr('modules.mandarin-ipa.text.81a4b066ea77')}}<span v-if="dirty" :aria-label="tr('modules.mandarin-ipa.text.6a1be012c99c')"> *</span></button>
        <button type="button" data-action="module-help" @click="emit('help')">{{tr('modules.mandarin-ipa.text.a57cfcb8428d')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.mandarin-ipa.text.7eaa43419522')}}</button>
      </template></ModuleToolbar></template>
  <template #status>
    <ModuleStatus v-if="error" kind="error" :message="error"><button @click="error=''">{{tr('modules.mandarin-ipa.text.fc06733cf134')}}</button></ModuleStatus>
    <ModuleStatus v-else-if="exporting" kind="loading" :message="tr('modules.mandarin-ipa.text.e618cbab4773')"/>
  </template>

  <div v-resizable-panels="{key:stateKey,center:'.m13-result-section',centerMin:240,panels:[{selector:'.m13-input-section',side:'left',label:tr('modules.mandarin-ipa.text.4bfa8eafda56'),initial:300,min:240,max:1600,initialFractionOfRemaining:0.5},{selector:'.m13-settings-section',side:'left',variable:'--panel-right',label:tr('modules.mandarin-ipa.text.fc75344980b3'),initial:300,min:240,max:560}]}" :key="draft.layout" class="m13-workspace" :class="{'m13-workspace-stacked':draft.layout==='stacked'}">
    <ModuleSection class="m13-settings-section" :label="tr('modules.mandarin-ipa.text.03968338e1d4')" :title="tr('modules.mandarin-ipa.text.fc75344980b3')">

      <p class="m13-local-note">{{tr('modules.mandarin-ipa.text.2789321f461d')}}</p><p class="m13-mapping-note">{{tr('modules.mandarin-ipa.text.17755b01bbed')}}</p>

      <label>{{tr('modules.mandarin-ipa.text.0ef0b768da20')}}<select :value="draft.standard" :aria-label="tr('modules.mandarin-ipa.text.0ef0b768da20')" :disabled="tableBusy" @change="selectStandard(($event.target as HTMLSelectElement).value)"><option v-for="standard in standards" :key="standard" :value="standard">{{standardLabel(standard)}}</option><optgroup v-if="library.tables.length" :label="tr('modules.mandarin-ipa.text.c7f4db2345c2')"><option v-for="table in library.tables" :key="table.id" :value="table.id">{{table.name}}</option></optgroup></select></label>
      <div class="m13-table-actions"><button :disabled="!tablesReady||tableBusy" @click="openImport()">{{tr('modules.mandarin-ipa.text.4208e69684fe')}}</button><template v-if="customTable"><button :disabled="tableBusy" @click="toneError='';toneTarget=customTable">{{tr('modules.mandarin-ipa.text.234d5ff49d2b')}}</button><button :disabled="tableBusy" @click="openImport(customTable)">{{tr('modules.mandarin-ipa.text.880987e749b0')}}</button><button :disabled="tableBusy" @click="deleteError='';deleteTarget=customTable">{{tr('modules.mandarin-ipa.text.0bc92967ba8f')}}</button><span>{{customTable.kind==='hanzi'?tr('modules.mandarin-ipa.text.1cd03b9a270b'):tr('modules.mandarin-ipa.text.6713cdab9443')}} · {{customTable.rows.length.toLocaleString()}} {{tr('modules.mandarin-ipa.text.f004f1d84cf9')}}</span></template></div>
      <ModuleStatus v-if="tableError" kind="error" :message="tableError"><button :disabled="tableBusy" @click="loadTables">{{tr('modules.mandarin-ipa.text.60e1ac390bd4')}}</button></ModuleStatus>
      <ModuleStatus v-if="tableNotice" kind="info" :message="tableNotice"/>
      <fieldset v-if="untEnabled" class="m13-erhua"><legend>{{tr('modules.mandarin-ipa.text.fb4321ed87ad')}}</legend><button :disabled="!canAutoErhua" @click="combineErhua">{{tr('modules.mandarin-ipa.text.7a37a9dbf389')}}</button><button :disabled="!erhuaCount" @click="splitAllErhua">{{tr('modules.mandarin-ipa.text.70859fecfe6e')}}</button>
        <p>{{tr('modules.mandarin-ipa.text.8a5ef8f76bc7')}} {{erhuaCount}} {{tr('modules.mandarin-ipa.text.a9379054859c')}}</p>
      </fieldset>
      <fieldset><legend>{{tr('modules.mandarin-ipa.text.de90186fc663')}}</legend><label><input v-model="draft.display" type="radio" value="paired"/>{{tr('modules.mandarin-ipa.text.2381026c3153')}}</label><label><input v-model="draft.display" type="radio" value="ipa-only"/>{{tr('modules.mandarin-ipa.text.c1d197ec5c85')}}</label><button class="m13-tone-toggle" :aria-pressed="draft.showTone" @click="draft.showTone=!draft.showTone">{{tr('modules.mandarin-ipa.text.3751d166b9b0')}}</button></fieldset>
      <fieldset><legend>{{tr('modules.mandarin-ipa.text.f2e51e46ec7b')}}</legend><label><input v-model="draft.layout" type="radio" value="side-by-side"/>{{tr('modules.mandarin-ipa.text.2f56a4dfc5c8')}}</label><label><input v-model="draft.layout" type="radio" value="stacked"/>{{tr('modules.mandarin-ipa.text.8acc3de62ec7')}}</label></fieldset>
      <fieldset><legend>{{tr('modules.mandarin-ipa.text.b91b96f3f5d0')}}</legend><button :aria-pressed="draft.equalWidth" @click="draft.equalWidth=!draft.equalWidth">{{tr('modules.mandarin-ipa.text.a4c10dd667ed')}}</button></fieldset>
      <label>{{tr('modules.mandarin-ipa.text.9bd5d708ab99')}} <span>{{draft.hanziSize}} px</span><input v-model.number="draft.hanziSize" :aria-label="tr('modules.mandarin-ipa.text.9bd5d708ab99')" type="range" min="16" max="72"/></label>
      <label>{{tr('modules.mandarin-ipa.text.4dbb035fe149')}} <span>{{effectiveIpaSize(draft)}} px</span><input :value="effectiveIpaSize(draft)" :aria-label="tr('modules.mandarin-ipa.text.4dbb035fe149')" type="range" min="12" max="48" @input="draft.ipaSize=Number(($event.target as HTMLInputElement).value);draft.ipaSizeUserSet=true"/></label>
      <label>{{tr('modules.mandarin-ipa.text.bb7934f5b438')}} <span>{{draft.gap}} px</span><input v-model.number="draft.gap" :aria-label="tr('modules.mandarin-ipa.text.bb7934f5b438')" type="range" min="-12" max="20"/></label>
      <label>{{tr('modules.mandarin-ipa.text.40026aada6d8')}} <span>{{draft.lineHeight.toFixed(1)}}</span><input :value="Math.round(draft.lineHeight*10)" :aria-label="tr('modules.mandarin-ipa.text.40026aada6d8')" type="range" min="8" max="30" @input="draft.lineHeight=Number(($event.target as HTMLInputElement).value)/10"/></label>
      <fieldset><legend>{{tr('modules.mandarin-ipa.text.e352d9a301ac')}}</legend><button :aria-pressed="draft.bold" @click="draft.bold=!draft.bold"><strong>B</strong> {{tr('modules.mandarin-ipa.text.f6da93ed21e7')}}</button><button :aria-pressed="draft.italic" @click="draft.italic=!draft.italic"><em>I</em> {{tr('modules.mandarin-ipa.text.217901c51a44')}}</button><button :aria-pressed="draft.underline" @click="draft.underline=!draft.underline"><u>U</u> {{tr('modules.mandarin-ipa.text.d3b4a22b4725')}}</button></fieldset>
      <fieldset class="m13-colors"><legend>{{tr('modules.mandarin-ipa.text.f21ef4aba5ab')}}</legend><label><input type="color" :aria-label="tr('modules.mandarin-ipa.text.38dd98a5de9b')" :value="draft.ipaColor||themeColors.ipa" @input="draft.ipaColor=($event.target as HTMLInputElement).value"/>{{tr('modules.mandarin-ipa.text.38dd98a5de9b')}}</label><label><input type="color" :aria-label="tr('modules.mandarin-ipa.text.64f44918815e')" :value="draft.hanziColor||themeColors.hanzi" @input="draft.hanziColor=($event.target as HTMLInputElement).value"/>{{tr('modules.mandarin-ipa.text.64f44918815e')}}</label><button :disabled="!draft.ipaColor&&!draft.hanziColor" @click="draft.ipaColor='';draft.hanziColor=''">{{tr('modules.mandarin-ipa.text.d7fa90d22bb7')}}</button></fieldset>
      <FontFamilySelect v-model="draft.hanziFont" :label="tr('modules.mandarin-ipa.text.6750c6c6a069')" :names="fontNames" :labels="fontLabels"/>
      <p class="m13-font-note">{{tr('modules.mandarin-ipa.text.5e8b45cc4b88')}}</p>
    </ModuleSection>

    <ModuleSection class="m13-input-section" :label="tr('modules.mandarin-ipa.text.4bfa8eafda56')" :title="tr('modules.mandarin-ipa.text.4bfa8eafda56')">
      <template #actions><span class="m13-count">{{[...draft.text].length.toLocaleString()}} {{tr('modules.mandarin-ipa.text.d988c1bfb392')}}</span></template>
      <textarea v-model="draft.text" :aria-label="tr('modules.mandarin-ipa.text.fbbf2631f805')" :placeholder="tr('modules.mandarin-ipa.text.7873b4ef1b09')" spellcheck="false"/>
    </ModuleSection>

    <ModuleSection class="m13-result-section" :label="tr('modules.mandarin-ipa.text.54a8366124bb')" :title="tr('modules.mandarin-ipa.text.54a8366124bb')">
      <div ref="outputArea" class="m13-output" :class="{'m13-paired':draft.display==='paired'}" :style="outputStyle" :data-standard="draft.standard">
        <ModuleStatus v-if="!draft.text.trim()" kind="empty" :message="tr('modules.mandarin-ipa.text.0e330468bc1e')"/>
        <template v-else><div v-for="(paragraph,index) in paragraphs" :key="index" class="m13-paragraph">
        <template v-for="token in paragraph" :key="token.index">
          <span v-if="token.kind==='literal'" class="m13-token m13-literal" :class="{'m13-ipa-only-literal':draft.display==='ipa-only'}" :data-index="token.index">
            <span v-if="draft.display==='paired'" class="m13-ipa-placeholder" aria-hidden="true">&nbsp;</span><span class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
          <button v-else-if="token.variants.length>1||token.canCombine||token.erhua" type="button" class="m13-token m13-mapped m13-ambiguous" :class="{'m13-erhua-group':token.erhua}" :data-erhua="token.erhua||undefined" :data-index="token.index" :data-value="token.value" :aria-label="tr('modules.mandarin-ipa.text.9d3767f78998',{v0:token.char,v1:token.variants.length,v2:token.value,v3:token.erhua?tr('modules.mandarin-ipa.text.e4066f552054'):token.canCombine?tr('modules.mandarin-ipa.text.8257dd15f3e7'):''})" aria-haspopup="dialog" :aria-expanded="activeVariant?.index===token.index" @click="openVariants(token,$event)">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </button>
          <span v-else class="m13-token m13-mapped" :data-index="token.index" :data-value="token.value">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
        </template>
        </div></template>
      </div>
      <Teleport to="body">
      <div v-if="selectedToken" ref="variantPanel" class="m13-variants" :style="variantPosition" role="dialog" :aria-label="tr('modules.mandarin-ipa.reading.select',{character:selectedToken.char})">
        <div><strong>{{tr('modules.mandarin-ipa.text.89ea25e0c5c9')}}{{selectedToken.char}}</strong><button :aria-label="tr('modules.mandarin-ipa.text.16303eb3d9c3')" @click="closeVariants(true)">×</button></div>
        <button v-if="untEnabled&&(selectedToken.erhua||selectedToken.canCombine)" class="m13-erhua-action" @click="toggleErhua(selectedToken)">{{selectedToken.erhua?tr('modules.mandarin-ipa.text.a82eb8d62b2c'):tr('modules.mandarin-ipa.text.42e277c45883')}}</button>
        <button v-for="option in options" :key="option.index" :aria-pressed="selectedToken.selectedVariant===option.index" @click="chooseVariant(selectedToken,option.index)">
          <span>{{option.pinyin?option.pinyin+(option.tone===0?tr('modules.mandarin-ipa.text.f56fd6416d7f'):option.toneLabel):tr('modules.mandarin-ipa.text.2697bfc9fa58')+(option.index+1)}}</span><span class="ipa-text">{{option.value}}</span>
        </button>
      </div>
      </Teleport>
    </ModuleSection>


  </div>
  <CustomTableImport v-if="importOpen" :saving="tableBusy" :save-error="importError" :replace-target="replaceTarget" @close="importOpen=false" @save="saveTable"/>
  <PngExportDialog v-if="pngSnapshot" :snapshot="pngSnapshot" @close="pngSnapshot=undefined" @busy="exporting=$event"/>
  <TableToneDialog v-if="toneTarget" :table="toneTarget" :saving="tableBusy" :save-error="toneError" @close="toneTarget=undefined" @save="saveTones"/>
  <ModalDialog v-if="deleteTarget" :title="tr('modules.mandarin-ipa.text.11da3d944109')" variant="confirm" :close-disabled="tableBusy" @close="deleteTarget=undefined">
    <p>{{tr('modules.mandarin-ipa.text.8b4da7b8aae1')}} <strong>{{deleteTarget.name}}</strong>？</p>
    <p>{{tr('modules.mandarin-ipa.text.ff4db1513183')}}</p>
    <ModuleStatus v-if="deleteError" kind="error" :message="deleteError"/>
    <template #footer><button :disabled="tableBusy" @click="deleteTarget=undefined">{{tr('modules.mandarin-ipa.text.2cd0f3be8738')}}</button><button class="danger" :disabled="tableBusy" @click="deleteTable">{{tableBusy?tr('modules.mandarin-ipa.text.4fd4f40094d4'):tr('modules.mandarin-ipa.text.0bc92967ba8f')}}</button></template>
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
