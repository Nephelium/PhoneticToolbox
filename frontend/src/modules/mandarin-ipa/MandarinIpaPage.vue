<script setup lang="ts">
import {computed,reactive,ref,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import {host} from '../../state/workspace.ts';
import {downloadMandarinIpa,renderMandarinIpaPng} from './export.ts';
import {convertText,createDraft,effectiveIpaSize,restoreDraft,snapshotDraft,standards,variantKey,variantsFor,type MappedToken} from './state.ts';

const props=withDefaults(defineProps<{stateKey?:string}>(),{stateKey:'M13'});
const emit=defineEmits<{references:[];dirty:[value:boolean]}>();
const draftKey=computed(()=>'mandarin-ipa.v1.'+props.stateKey);
const loaded=host.projects.read<unknown>(draftKey.value,null);const draft=reactive(loaded?restoreDraft(loaded):createDraft());
const dirty=ref(false),error=ref(''),exporting=ref(false),outputArea=ref<HTMLElement>(),activeVariant=ref<{char:string;index:number}|null>(null);
const tokens=computed(()=>convertText(draft.text,draft.standard,draft.selectedVariants));
const selectedToken=computed(()=>activeVariant.value?tokens.value.find((token):token is MappedToken=>token.kind==='mapped'&&token.char===activeVariant.value?.char&&token.index===activeVariant.value?.index):undefined);
const options=computed(()=>selectedToken.value?variantsFor(selectedToken.value,draft.standard):[]);
const outputStyle=computed(()=>({'--m13-hanzi-size':draft.hanziSize+'px','--m13-ipa-size':effectiveIpaSize(draft)+'px','--m13-gap':draft.gap+'px','--m13-line-height':String(draft.lineHeight)}));
const hanziStyle=computed(()=>({fontSize:draft.hanziSize+'px',fontWeight:draft.bold?'700':'400',fontStyle:draft.italic?'italic':'normal',textDecoration:draft.underline?'underline':'none'}));

watch(draft,()=>{dirty.value=true;error.value='';},{deep:true});
watch(dirty,value=>emit('dirty',value),{immediate:true});
function openVariants(token:MappedToken){if(token.variants.length>1)activeVariant.value={char:token.char,index:token.index};}
function chooseVariant(token:MappedToken,index:number){draft.selectedVariants[variantKey(token.char,token.index)]=index;activeVariant.value=null;}
function save(){const ok=host.projects.write(draftKey.value,snapshotDraft(draft));if(ok){dirty.value=false;error.value='';}else error.value='草稿保存失败，当前文本和排版仍保留在页面中。';return ok;}
async function exportImage(){
  if(exporting.value)return;exporting.value=true;error.value='';activeVariant.value=null;
  try{
    const element=outputArea.value,style=element?getComputedStyle(element):getComputedStyle(document.documentElement);
    const blob=await renderMandarinIpaPng({tokens:tokens.value,draft:snapshotDraft(draft),width:element?.clientWidth??900,uiFont:style.getPropertyValue('--font').trim()||style.fontFamily});
    downloadMandarinIpa(blob);
  }catch(cause){error.value=cause instanceof Error?cause.message:'图片导出失败，当前转换结果仍已保留。';}
  finally{exporting.value=false;}
}
defineExpose({save});
</script>

<template>
<ModuleFrame class="mandarin-ipa-page" label="普通话转 IPA 工作区">
  <template #toolbar>
    <ModuleToolbar label="普通话转 IPA 操作">
      <button class="primary" :disabled="exporting||!draft.text.trim()" @click="exportImage">{{exporting?'正在生成…':'保存为 PNG'}}</button>
      <button :disabled="!dirty" @click="save">保存本机草稿<span v-if="dirty" aria-label="未保存"> *</span></button>
      <template #actions><span class="m13-local-note">本地逐字转换 · 文本不上传</span><button @click="emit('references')">帮助与来源</button></template>
    </ModuleToolbar>
  </template>
  <template #status>
    <ModuleStatus v-if="error" kind="error" :message="error"><button @click="error=''">收起提示</button></ModuleStatus>
    <ModuleStatus v-else-if="exporting" kind="loading" message="正在用本地 Doulos SIL 字体生成 PNG…"/>
    <ModuleStatus v-else kind="info" message="结果按单字映射，不处理语流音变。多音字默认使用旧数据中第一条读音，需要人工选择；标准名称是旧数据列名，不代表规范来源已核验。"/>
  </template>

  <div class="m13-workspace" :class="{'m13-workspace-stacked':draft.layout==='stacked'}">
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
          <button v-else-if="token.variants.length>1" type="button" class="m13-token m13-mapped m13-ambiguous" :data-index="token.index" :data-value="token.value" :aria-label="`${token.char}：${token.variants.length} 个读音，当前 ${token.value}`" @click="openVariants(token)">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </button>
          <span v-else class="m13-token m13-mapped" :data-index="token.index" :data-value="token.value">
            <span class="m13-ipa ipa-text">{{token.value}}</span><span v-if="draft.display==='paired'" class="m13-hanzi" :style="hanziStyle">{{token.char}}</span>
          </span>
        </template>
      </div>
      <div v-if="selectedToken" class="m13-variants" role="dialog" :aria-label="'选择 '+selectedToken.char+' 的读音'">
        <div><strong>选择读音：{{selectedToken.char}}</strong><button aria-label="关闭读音选择" @click="activeVariant=null">关闭</button></div>
        <button v-for="option in options" :key="option.index" :aria-pressed="selectedToken.selectedVariant===option.index" @click="chooseVariant(selectedToken,option.index)">
          <span>{{option.pinyin}}{{option.toneLabel==='轻声'?'（轻声）':option.toneLabel}}</span><span class="ipa-text">{{option.value}}</span>
        </button>
      </div>
    </ModuleSection>

    <ModuleSection class="m13-settings-section" label="转换和排版设置" title="转换与排版">
      <label>转换标准<select v-model="draft.standard" aria-label="转换标准"><option v-for="standard in standards" :key="standard" :value="standard">{{standard}}</option></select></label>
      <fieldset><legend>显示内容</legend><label><input v-model="draft.display" type="radio" value="paired"/>字音同显</label><label><input v-model="draft.display" type="radio" value="ipa-only"/>仅音标</label></fieldset>
      <fieldset><legend>输入与结果排布</legend><label><input v-model="draft.layout" type="radio" value="side-by-side"/>左右排布</label><label><input v-model="draft.layout" type="radio" value="stacked"/>上下排布</label></fieldset>
      <label>汉字字号 <span>{{draft.hanziSize}} px</span><input v-model.number="draft.hanziSize" aria-label="汉字字号" type="range" min="16" max="72"/></label>
      <label>音标字号 <span>{{draft.ipaSize}} px</span><input v-model.number="draft.ipaSize" aria-label="音标字号" type="range" min="12" max="48" @input="draft.ipaSizeUserSet=true"/></label>
      <label>字音间距 <span>{{draft.gap}} px</span><input v-model.number="draft.gap" aria-label="字音间距" type="range" min="-12" max="20"/></label>
      <label>行距 <span>{{draft.lineHeight.toFixed(1)}}</span><input :value="Math.round(draft.lineHeight*10)" aria-label="行距" type="range" min="8" max="30" @input="draft.lineHeight=Number(($event.target as HTMLInputElement).value)/10"/></label>
      <fieldset><legend>参考汉字样式</legend><button :aria-pressed="draft.bold" @click="draft.bold=!draft.bold"><strong>B</strong> 粗体</button><button :aria-pressed="draft.italic" @click="draft.italic=!draft.italic"><em>I</em> 斜体</button><button :aria-pressed="draft.underline" @click="draft.underline=!draft.underline"><u>U</u> 下划线</button></fieldset>
    </ModuleSection>
  </div>
</ModuleFrame>
</template>

<style scoped>
.mandarin-ipa-page{height:100%;overflow:auto}.m13-local-note{color:var(--muted);font-size:var(--support-size)}
.m13-workspace{display:grid;grid-template-columns:minmax(250px,.85fr) minmax(360px,1.45fr) minmax(240px,.8fr);gap:var(--module-gap);align-items:stretch;min-height:0}.m13-workspace-stacked{grid-template-columns:minmax(0,1fr)}.m13-workspace-stacked .m13-settings-section{grid-row:2}.m13-workspace-stacked .m13-result-section{grid-row:3}
.m13-input-section,.m13-result-section,.m13-settings-section{display:flex;flex-direction:column}.m13-input-section textarea{flex:1;width:100%;min-height:360px;resize:vertical;padding:12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);font:28px/1.8 var(--font);overflow-wrap:anywhere}.m13-count{margin-top:8px;color:var(--muted);font-size:var(--support-size);text-align:right}
.m13-output{flex:1;min-height:360px;padding:16px;border:1px solid var(--border);border-radius:6px;background:var(--app);color:var(--text);overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;line-height:var(--m13-line-height)}
.m13-token{display:inline-block;position:relative;vertical-align:bottom;margin:0 2px;padding:2px 4px;border-radius:4px;gap:0}.m13-paired .m13-token{display:inline-flex;flex-direction:column;align-items:center}.m13-ipa{font-family:var(--font-ipa,"PTB-Doulos"),serif;font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0;overflow:visible;color:var(--accent)}.m13-hanzi{font-family:var(--font);line-height:1.2;margin-top:var(--m13-gap)}.m13-ipa-placeholder{font-size:var(--m13-ipa-size);line-height:1.35;padding:.12em 0}.m13-ipa-only-literal .m13-hanzi{font-size:var(--m13-ipa-size)!important;font-weight:400!important;font-style:normal!important;text-decoration:none!important;margin:0}
.m13-ambiguous{min-height:0;border:0;background:transparent;color:inherit;white-space:normal}.m13-ambiguous:hover{background:var(--selected)}.m13-ambiguous:after{content:'▼';position:absolute;right:0;bottom:-1px;color:var(--accent);font-size:8px}.m13-mapped:not(button):hover{background:var(--selected)}
.m13-variants{margin-top:12px;padding:12px;border:1px solid var(--accent);border-radius:var(--radius);background:var(--panel);box-shadow:var(--shadow)}.m13-variants>div{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.m13-variants>button{display:flex;width:100%;justify-content:space-between;margin-top:6px}.m13-variants>button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}.m13-variants .ipa-text{font-size:18px;color:var(--accent)}
.m13-settings-section>label{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 10px;align-items:center;margin-bottom:12px;font-size:var(--control-size)}.m13-settings-section>label select,.m13-settings-section>label input{grid-column:1/-1;width:100%}.m13-settings-section>label span{font-variant-numeric:tabular-nums;color:var(--muted)}fieldset{display:flex;flex-wrap:wrap;gap:8px 12px;margin:0 0 14px;padding:10px;border:1px solid var(--border);border-radius:6px}legend{padding:0 4px;font-size:var(--support-size);color:var(--muted)}fieldset label{display:flex;align-items:center;gap:5px}fieldset button{font-size:var(--support-size)}fieldset button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}
@container module (max-width:1000px){.m13-workspace{grid-template-columns:minmax(240px,.8fr) minmax(360px,1.2fr)}.m13-settings-section{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}.m13-settings-section>:deep(.module-section-heading){grid-column:1/-1}.m13-settings-section fieldset{align-self:start}}
@container module (max-width:680px){.m13-workspace{grid-template-columns:1fr}.m13-settings-section{grid-column:auto;display:flex}.m13-input-section textarea,.m13-output{min-height:240px}}
</style>
