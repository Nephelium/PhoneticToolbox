<script setup lang="ts">
import {ref,computed,watch} from 'vue';
import {preferences,setFonts,prepareFonts,fontError} from '../state/fonts.ts';
import {defaults,normalizeFonts,candidates} from '../design/fonts.ts';
import {desktopFontFamilies} from '../platform/desktop.ts';
const emit=defineEmits<{dirty:[value:boolean]}>();
const draft=ref(normalizeFonts(preferences.value)),message=ref(''),busy=ref(false),extra=ref<string[]>([]),preview=ref<Record<string,string>>({});
watch(preferences,value=>{draft.value=normalizeFonts(value);});
const fontLabels:Record<string,string>={SimSun:'宋体',KaiTi:'楷体','Source Han Serif SC':'思源宋体','Microsoft YaHei':'微软雅黑'};
const names=computed(()=>[...new Set([...candidates.zh,...candidates.latin,...candidates.mono,...extra.value])].sort((a,b)=>a.localeCompare(b)));
const customMono=ref(false);
const monoNames=computed(()=>[...new Set([...candidates.mono,...(draft.value.mono?[draft.value.mono]:[]),...extra.value])]);
function selectMono(event:Event){const value=(event.target as HTMLSelectElement).value;customMono.value=value==='__custom__';if(!customMono.value)draft.value.mono=value;}
async function list(){busy.value=true;message.value='';try{
 if(desktopFontFamilies)extra.value=await desktopFontFamilies();
 else{const api=(window as unknown as {queryLocalFonts?:()=>Promise<{family:string}[]>}).queryLocalFonts;if(!api)throw Error('此浏览器不支持字体列表读取，可直接填写本机已安装的字体名称。');extra.value=(await api()).map(f=>f.family);}
 message.value=`已读取 ${extra.value.length} 个字体名称，应用时会检查所选字体。`;
}catch(e){message.value=e instanceof Error?e.message:'无法读取字体列表。';}finally{busy.value=false;}}
async function show(){busy.value=true;message.value='';try{const p=await prepareFonts(normalizeFonts(draft.value));preview.value={fontFamily:p.ui,'--preview-mono':p.mono,'--preview-figure':p.figure,'--preview-size':p.size+'px'};message.value='预览已更新，点击应用字体后生效。';}catch(e){message.value=(e as Error).message;}finally{busy.value=false;}}
async function apply(){if(busy.value)return false;busy.value=true;message.value='';try{if(await setFonts(draft.value)){draft.value=normalizeFonts(preferences.value);preview.value={};message.value='字体已应用到工作台与图表，并保存在本机。';return true;}return false;}catch(e){message.value=(e as Error).message;return false;}finally{busy.value=false;}}
function cancel(){draft.value=normalizeFonts(preferences.value);customMono.value=false;preview.value={};message.value='已取消本次字体编辑。';}
watch(()=>busy.value||JSON.stringify(normalizeFonts(draft.value))!==JSON.stringify(normalizeFonts(preferences.value)),value=>emit('dirty',value),{immediate:true});
defineExpose({save:apply});
</script>
<template><section class="font-settings settings-card" aria-label="字体设置">
<div class="section-title settings-heading"><h3>字体</h3><button @click="list" :disabled="busy">读取本机字体列表</button></div>
<p class="muted">默认宋体与 Times New Roman。可填写已安装的字体名称。</p>
<datalist id="ptb-font-options"><option v-for="name in names" :key="name" :value="name" :label="fontLabels[name]||name"/></datalist>
<div class="font-fields"><label>中文字体<input v-model="draft.zh" list="ptb-font-options" aria-label="中文字体" placeholder="系统默认"/></label><label>英文与数字字体<input v-model="draft.latin" list="ptb-font-options" aria-label="英文与数字字体" placeholder="系统默认"/></label><div class="mono-field"><label for="mono-choice">代码与等宽字体</label><span class="font-inline-hint">JetBrains Mono 已内置，无需安装</span><select id="mono-choice" :value="customMono?'__custom__':draft.mono" aria-label="代码与等宽字体" @change="selectMono"><option v-for="name in monoNames" :key="name" :value="name">{{name==='JetBrains Mono'?'JetBrains Mono（内置）':name}}</option><option value="">系统默认</option><option value="__custom__">自定义字体…</option></select><input v-if="customMono" v-model="draft.mono" list="ptb-font-options" aria-label="自定义代码字体" placeholder="填写已安装的字体名称"/></div><label>所有 IPA 音标<input value="Doulos SIL（固定）" aria-label="IPA 字体" readonly/></label></div>
<label class="font-follow"><input v-model="draft.figure.follow" type="checkbox"/>图表与导出跟随全局字体</label>
<p class="muted">导出优先使用所选字体，缺失时使用兼容字体并记录实际字体。</p>
<div v-if="!draft.figure.follow" class="font-fields"><label>图表中文字体<input v-model="draft.figure.zh" list="ptb-font-options" aria-label="图表中文字体" placeholder="系统默认"/></label><label>图表英文字体<input v-model="draft.figure.latin" list="ptb-font-options" aria-label="图表英文字体" placeholder="系统默认"/></label></div>
<label class="setting-row">图表基础字号（px）<input v-model.number="draft.figure.size" type="number" min="10" max="24" step="1" aria-label="图表基础字号"/></label>
<div class="font-preview" :style="preview"><span>声学参数 · 中文字体预览</span><span>PhoneticToolbox 0123456789 −12.5 Hz</span><span class="ipa-text">[aː tʰ ɕ ŋ ə ã n̩ ɑ²]</span><code class="font-code-preview" :style="{fontFamily:preview['--preview-mono']}">f0 = signal[0:10]  # Hz != ms</code><svg viewBox="0 0 420 72" role="img" aria-label="图表字体预览" :style="{fontFamily:preview['--preview-figure']||'var(--font-figure)',fontSize:preview['--preview-size']||'var(--figure-size)'}"><path d="M40 5V42H405" fill="none" stroke="currentColor"/><text x="48" y="24" fill="currentColor">基频 F0 · −12.5 Hz</text><text x="240" y="64" fill="currentColor">时间 Time (s)</text></svg></div>
<p class="muted">IPA 使用内置 Doulos SIL。字体设置应用于页面与新生成的图像。</p>
<p v-if="message||fontError" role="status">{{message||fontError}}</p>
<div class="font-actions"><button @click="draft=defaults();customMono=false;message='已填入默认设置，点击应用字体后生效。'" :disabled="busy">恢复默认</button><button @click="cancel" :disabled="busy">取消字体编辑</button><button @click="show" :disabled="busy">预览字体</button><button class="primary" @click="apply" :disabled="busy">{{busy?'检查字体…':'应用字体'}}</button></div>
</section></template>
<style scoped>
.font-fields .mono-field{display:flex;flex-direction:column;gap:5px;min-width:0}.mono-field select{width:100%;min-width:0;margin-top:auto}
.font-settings{display:flex;flex-direction:column;gap:12px;align-self:start}.font-settings .section-title{margin:0}.font-settings .section-title button{font-size:12px;min-height:28px;padding:4px 8px}.font-fields{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}.font-fields label{display:flex;flex-direction:column;gap:5px;min-width:0;position:relative}.font-fields input{width:100%;min-width:0}.font-inline-hint{font-size:11px;color:var(--muted);margin-top:-3px}.font-fields label:has(.font-inline-hint) input{margin-top:auto}.font-follow{display:flex;align-items:center;gap:7px;margin:2px 0}.font-preview{display:grid;grid-template-columns:1fr;gap:6px;padding:12px 14px;background:var(--app);border:1px solid var(--border);border-radius:7px;line-height:1.65;overflow-wrap:anywhere}.font-preview svg{width:100%;height:65px;max-height:65px}.font-code-preview{font-family:var(--preview-mono,var(--font-mono));font-size:12px;font-variant-ligatures:none}.font-actions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;border-top:1px solid var(--border);padding-top:12px}.font-actions button{font-size:12px;padding:5px 9px}.font-settings .muted{line-height:1.6;font-size:12px}.font-settings .setting-row{margin:0;gap:12px}.font-settings .setting-row input{width:80px}.font-settings [role=status]{font-size:12px;color:var(--accent)}@container module (max-width:520px){.font-fields{grid-template-columns:1fr}}
</style>
