<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import {vResizablePanels} from '../../layout/resizablePanels.ts';
import {computed,nextTick,onBeforeUnmount,onMounted,reactive,ref,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import FontFamilySelect from '../../components/FontFamilySelect.vue';
import {candidates,quoteFamily,validFamily} from '../../design/fonts.ts';
import {desktopFontFamilies} from '../../platform/desktop.ts';
import {host} from '../../state/workspace.ts';
import {downloadMandarinIpa,renderMandarinIpaPng} from './export.ts';
import {convertText,createPageDraft,effectiveIpaSize,moveVariantChoices,restoreDraft,snapshotDraft,standards,variantKey,variantsFor,type MappedToken} from './state.ts';

const props=withDefaults(defineProps<{stateKey?:string}>(),{stateKey:'M13'});
const emit=defineEmits<{references:[];help:[];dirty:[value:boolean]}>();
const draftKey=computed(()=>'mandarin-ipa.v1.'+props.stateKey);
const saved=ref(host.projects.read<unknown>(draftKey.value,null)),draft=reactive(createPageDraft(saved.value));
const canRestore=computed(()=>!!restoreDraft(saved.value).text.trim());
const installedFonts=ref<string[]>([]);
const themeColors=ref({ipa:'#a44e2f',hanzi:'#182332'});
let themeObserver:MutationObserver|undefined;
function readThemeColors(){const style=getComputedStyle(document.documentElement);themeColors.value={ipa:style.getPropertyValue('--accent').trim(),hanzi:style.getPropertyValue('--text').trim()};}
const fontNames=computed(()=>installedFonts.value.length?installedFonts.value:candidates.zh);
const fontLabels={SimSun:'宋体',KaiTi:'楷体',FangSong:'仿宋','Microsoft YaHei':'微软雅黑'};
onMounted(async()=>{readThemeColors();themeObserver=new MutationObserver(readThemeColors);themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','data-palette','style']});if(desktopFontFamilies)try{installedFonts.value=[...new Set((await desktopFontFamilies()).map(name=>name.trim()).filter(name=>name&&validFamily(name)))];}catch{/* Common names and custom input remain available. */}});
const dirty=ref(false),error=ref(''),exporting=ref(false),outputArea=ref<HTMLElement>(),activeVariant=ref<{char:string;index:number}|null>(null);
const tokens=computed(()=>convertText(draft.text,draft.standard,draft.selectedVariants,draft.showTone));
const selectedToken=computed(()=>activeVariant.value?tokens.value.find((token):token is MappedToken=>token.kind==='mapped'&&token.char===activeVariant.value?.char&&token.index===activeVariant.value?.index):undefined);
const options=computed(()=>selectedToken.value?variantsFor(selectedToken.value,draft.standard,draft.showTone):[]);
const outputStyle=computed(()=>({'--m13-hanzi-size':draft.hanziSize+'px','--m13-ipa-size':effectiveIpaSize(draft)+'px','--m13-gap':draft.gap+'px','--m13-line-height':String(draft.lineHeight),'--m13-ipa-color':draft.ipaColor||'var(--accent)','--m13-hanzi-color':draft.hanziColor||'var(--text)','--m13-hanzi-font':draft.hanziFont&&validFamily(draft.hanziFont)?quoteFamily(draft.hanziFont)+', var(--font)':'var(--font)'}));
const hanziStyle=computed(()=>({fontSize:draft.hanziSize+'px',fontWeight:draft.bold?'700':'400',fontStyle:draft.italic?'italic':'normal',textDecoration:draft.underline?'underline':'none'}));

watch(draft,()=>{dirty.value=true;error.value='';},{deep:true});
watch(dirty,value=>emit('dirty',value),{immediate:true});
watch(()=>draft.text,(next,previous)=>{draft.selectedVariants=moveVariantChoices(previous,next,draft.selectedVariants);},{flush:'sync'});
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
  closeVariants();if(token.variants.length<2)return;
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
function chooseVariant(token:MappedToken,index:number){draft.selectedVariants[variantKey(token.char,token.index)]=index;closeVariants(true);}
watch(()=>[draft.text,draft.layout,draft.display,draft.standard],()=>closeVariants());
onBeforeUnmount(()=>{closeVariants();themeObserver?.disconnect();});
function save(){const snapshot=snapshotDraft(draft),ok=host.projects.write(draftKey.value,snapshot);if(ok){saved.value=snapshot;dirty.value=false;error.value='';}else error.value='草稿保存失败，当前文本和排版仍保留在页面中。';return ok;}
async function loadSaved(){if(dirty.value||!canRestore.value)return;Object.assign(draft,restoreDraft(saved.value));await nextTick();dirty.value=false;error.value='';}
async function exportImage(){
  if(exporting.value)return;exporting.value=true;error.value='';closeVariants();
  try{
    const element=outputArea.value,style=element?getComputedStyle(element):getComputedStyle(document.documentElement);
    const blob=await renderMandarinIpaPng({tokens:tokens.value,draft:snapshotDraft(draft),width:element?.clientWidth??900,uiFont:style.getPropertyValue('--font').trim()||style.fontFamily,ipaColor:style.getPropertyValue('--m13-ipa-color').trim()});
    downloadMandarinIpa(blob);
  }catch(cause){error.value=cause instanceof Error?cause.message:'图片导出失败，当前转换结果仍已保留。';}
  finally{exporting.value=false;}
}
defineExpose({save});
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

  <div v-resizable-panels="{key:stateKey,center:'.m13-result-section',centerMin:360,panels:[{selector:'.m13-input-section',side:'left',label:'汉字输入',initial:300,min:240,max:560},{selector:'.m13-settings-section',side:'left',variable:'--panel-right',label:'转换与排版',initial:300,min:240,max:560}]}" :key="draft.layout" class="m13-workspace" :class="{'m13-workspace-stacked':draft.layout==='stacked'}">
    <ModuleSection class="m13-settings-section" label="转换和排版设置" title="转换与排版">

      <p class="m13-local-note">本地逐字转换 · 文本不上传</p><p class="m13-mapping-note">按单字映射，不处理语流音变。多音字需人工选读音；标准名称沿用旧字表，规范来源尚待核验。</p>

      <label>转换标准<select v-model="draft.standard" aria-label="转换标准"><option v-for="standard in standards" :key="standard" :value="standard">{{standard}}</option></select></label>
      <fieldset><legend>显示内容</legend><label><input v-model="draft.display" type="radio" value="paired"/>字音同显</label><label><input v-model="draft.display" type="radio" value="ipa-only"/>仅音标</label><button class="m13-tone-toggle" :aria-pressed="draft.showTone" @click="draft.showTone=!draft.showTone">显示声调</button></fieldset>
      <fieldset><legend>输入与结果排布</legend><label><input v-model="draft.layout" type="radio" value="side-by-side"/>左右排布</label><label><input v-model="draft.layout" type="radio" value="stacked"/>上下排布</label></fieldset>
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
      <textarea v-model="draft.text" aria-label="待转换汉字文本" placeholder="输入汉字、标点或分行文本…" spellcheck="false"/>
      <p class="m13-count">{{[...draft.text].length.toLocaleString()}} 个字符</p>
    </ModuleSection>

    <ModuleSection class="m13-result-section" label="转换结果" title="转换结果">
      <div ref="outputArea" class="m13-output" :class="{'m13-paired':draft.display==='paired'}" :style="outputStyle" :data-standard="draft.standard">
        <ModuleStatus v-if="!draft.text.trim()" kind="empty" message="输入文本后，这里会实时显示转换结果。"/>
        <template v-else v-for="token in tokens" :key="token.index">
          <br v-if="token.kind==='literal'&&token.newline"/>
          <span v-else-if="token.kind==='literal'" class="m13-token m13-literal" :class="{'m13-ipa-only-literal':draft.display==='ipa-only'}">
            <span v-if="draft.display==='paired'" class="m13-ipa-placeholder" aria-hidden="true">&nbsp;</span><span class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
          <button v-else-if="token.variants.length>1" type="button" class="m13-token m13-mapped m13-ambiguous" :data-index="token.index" :data-value="token.value" :aria-label="`${token.char}：${token.variants.length} 个读音，当前 ${token.value}`" aria-haspopup="dialog" :aria-expanded="activeVariant?.index===token.index" @click="openVariants(token,$event)">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </button>
          <span v-else class="m13-token m13-mapped" :data-index="token.index" :data-value="token.value">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
        </template>
      </div>
      <Teleport to="body">
      <div v-if="selectedToken" ref="variantPanel" class="m13-variants" :style="variantPosition" role="dialog" :aria-label="'选择 '+selectedToken.char+' 的读音'">
        <div><strong>选择读音：{{selectedToken.char}}</strong><button aria-label="关闭读音选择" @click="closeVariants(true)">×</button></div>
        <button v-for="option in options" :key="option.index" :aria-pressed="selectedToken.selectedVariant===option.index" @click="chooseVariant(selectedToken,option.index)">
          <span>{{option.pinyin}}{{option.toneLabel==='轻声'?'（轻声）':option.toneLabel}}</span><span class="ipa-text">{{option.value}}</span>
        </button>
      </div>
      </Teleport>
    </ModuleSection>


  </div>
</ModuleFrame>
</template>

<style scoped>
.mandarin-ipa-page{height:100%;overflow:auto}.m13-local-note{color:var(--muted);font-size:var(--support-size)}
.m13-workspace{display:grid;grid-template-columns:var(--panel-right,300px) var(--panel-left,300px) minmax(360px,1fr);grid-template-areas:'settings input result';gap:var(--module-gap);align-items:stretch;min-height:0}.m13-input-section{grid-area:input}.m13-result-section{grid-area:result}.m13-settings-section{grid-area:settings}.m13-workspace.m13-workspace-stacked{grid-template-columns:var(--panel-right,300px) minmax(320px,1fr);grid-template-areas:'settings input' 'settings result'}.m13-workspace-stacked .m13-input-section textarea{min-height:180px}.m13-workspace-stacked .m13-output{min-height:260px}.m13-local-note{margin:10px 0 14px}
.m13-input-section,.m13-result-section,.m13-settings-section{display:flex;flex-direction:column}.m13-input-section textarea{flex:1;width:100%;min-height:360px;resize:vertical;padding:12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);font:2rem/1.8 var(--font);overflow-wrap:anywhere}.m13-count{margin-top:8px;color:var(--muted);font-size:var(--support-size);text-align:right}
.m13-output{flex:1;min-height:360px;padding:16px;border:1px solid var(--border);border-radius:6px;background:var(--app);color:var(--text);overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;line-height:var(--m13-line-height)}
.m13-token{display:inline-block;position:relative;vertical-align:bottom;margin:0 2px;padding:2px 4px;border-radius:4px;gap:0}.m13-paired .m13-token{display:inline-flex;flex-direction:column;align-items:center}.m13-ipa{font-family:var(--font-ipa,"PTB-Doulos"),serif;font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0;overflow:visible;color:var(--accent)}.m13-hanzi{font-family:var(--font);line-height:1.2;margin-top:var(--m13-gap)}.m13-ipa-placeholder{font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0}.m13-ipa-only-literal .m13-hanzi{font-size:var(--m13-ipa-size)!important;font-weight:400!important;font-style:normal!important;text-decoration:none!important;margin:0}
.m13-ambiguous{min-height:0;border:0;background:transparent;color:inherit;white-space:normal}.m13-ambiguous:hover{background:var(--selected)}.m13-ambiguous:after{content:'▼';position:absolute;right:0;bottom:-1px;color:var(--accent);font-size:0.571429rem}.m13-mapped:not(button):hover{background:var(--selected)}
.m13-variants{position:fixed;z-index:100;width:240px;height:240px;max-width:calc(100vw / var(--page-scale,1) - 16px);max-height:calc(100dvh / var(--page-scale,1) - 16px);overflow:auto;padding:10px;border:1px solid var(--accent);border-radius:var(--radius);background:var(--panel);color:var(--text);font-family:var(--font);font-size:var(--control-size);box-shadow:var(--shadow)}.m13-variants>div{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;position:sticky;top:-10px;background:var(--panel)}.m13-variants>div button{padding:2px 8px;min-height:28px;font-size:1.428571rem}.m13-variants>button{display:flex;width:100%;justify-content:space-between;margin-top:6px;padding:6px 8px;white-space:normal;overflow-wrap:anywhere}.m13-variants>button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}.m13-variants .ipa-text{font-size:1.285714rem;color:var(--accent)}
.m13-settings-section>label:not(.font-family-select){display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 10px;align-items:center;margin-bottom:12px;font-size:var(--control-size)}.m13-settings-section>label:not(.font-family-select) select,.m13-settings-section>label:not(.font-family-select) input{grid-column:1/-1;width:100%}.m13-settings-section>label:not(.font-family-select) span{font-variant-numeric:tabular-nums;color:var(--muted)}fieldset{display:flex;flex-wrap:wrap;gap:8px 12px;margin:0 0 14px;padding:10px;border:1px solid var(--border);border-radius:6px}legend{padding:0 4px;font-size:var(--support-size);color:var(--muted)}fieldset label{display:flex;align-items:center;gap:5px}fieldset button{font-size:var(--support-size)}fieldset button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}

.m13-workspace{flex:1;min-height:0;grid-template-rows:minmax(0,1fr)}
.m13-input-section,.m13-result-section,.m13-settings-section{min-height:0;overflow:auto;overscroll-behavior:contain}
.m13-input-section,.m13-settings-section{background:var(--app);scrollbar-gutter:stable}.m13-input-section textarea,.m13-output{min-height:160px}
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
.m13-settings-section .m13-tone-toggle{margin-left:auto}
.m13-settings-section .m13-colors{gap:5px 8px}
.m13-colors input[type=color]{width:26px;height:26px;min-height:26px;padding:2px;border-radius:4px;cursor:pointer}
.m13-colors button{margin-left:auto}
.m13-settings-section>.font-family-select{display:flex;flex-direction:column;align-items:stretch;gap:2px;margin-bottom:0}
.m13-settings-section :deep(.font-family-select select),.m13-settings-section :deep(.font-family-select input){height:max(30px,calc(var(--control-size)*1.5 + 8px));min-height:30px;padding:3px 6px}
.m13-hanzi{font-family:var(--m13-hanzi-font,var(--font));color:var(--m13-hanzi-color,var(--text))}
.m13-ipa{color:var(--m13-ipa-color,var(--accent))}
</style>
