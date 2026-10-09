<script setup lang="ts">
import {ref,computed,watch} from 'vue';
import {preferences,setFonts,prepareFonts,fontError} from '../state/fonts.ts';
import {defaults,normalizeFonts,candidates} from '../design/fonts.ts';
import {desktopFontFamilies} from '../platform/desktop.ts';
import FontFamilySelect from './FontFamilySelect.vue';
const emit=defineEmits<{dirty:[value:boolean]}>();
const draft=ref(normalizeFonts(preferences.value)),message=ref(''),busy=ref(false),extra=ref<string[]>([]),preview=ref<Record<string,string>>({});
watch(preferences,value=>{draft.value=normalizeFonts(value);});
const fontLabels:Record<string,string>={SimSun:'宋体',KaiTi:'楷体','Source Han Serif SC':'思源宋体','Microsoft YaHei':'微软雅黑'};
const names=computed(()=>[...new Set([...candidates.zh,...candidates.latin,...candidates.mono,...extra.value])].sort((a,b)=>a.localeCompare(b)));
const resetToken=ref(0);
const monoNames=computed(()=>[...new Set([...candidates.mono,...(draft.value.mono?[draft.value.mono]:[]),...extra.value])]);
async function list(){busy.value=true;message.value='';try{
 if(desktopFontFamilies)extra.value=await desktopFontFamilies();
 else{const api=(window as unknown as {queryLocalFonts?:()=>Promise<{family:string}[]>}).queryLocalFonts;if(!api)throw Error('此浏览器不支持字体列表读取，可直接填写本机已安装的字体名称。');extra.value=(await api()).map(f=>f.family);}
 message.value=`已读取 ${extra.value.length} 个字体名称，应用时会检查所选字体。`;
}catch(e){message.value=e instanceof Error?e.message:'无法读取字体列表。';}finally{busy.value=false;}}
async function show(){busy.value=true;message.value='';try{const p=await prepareFonts(normalizeFonts(draft.value));preview.value={fontFamily:p.ui,fontSize:p.bodySize+'px','--preview-mono':p.mono,'--preview-figure':p.figure,'--preview-size':p.size+'px'};message.value='预览已更新，点击应用字体后生效。';}catch(e){message.value=(e as Error).message;}finally{busy.value=false;}}
async function apply(){if(busy.value)return false;busy.value=true;message.value='';try{if(await setFonts(draft.value)){draft.value=normalizeFonts(preferences.value);resetToken.value++;preview.value={};message.value='字体已应用到工作台与图表，并保存在本机。';return true;}return false;}catch(e){message.value=(e as Error).message;return false;}finally{busy.value=false;}}
function cancel(){draft.value=normalizeFonts(preferences.value);resetToken.value++;preview.value={};message.value='已取消本次字体编辑。';}
watch(()=>busy.value||JSON.stringify(normalizeFonts(draft.value))!==JSON.stringify(normalizeFonts(preferences.value)),value=>emit('dirty',value),{immediate:true});
defineExpose({save:apply});
</script>
<template><section class="font-settings settings-card" aria-label="字体设置">
<div class="section-title settings-heading"><h3>字体</h3><button @click="list" :disabled="busy">读取本机字体列表</button></div>
<p class="settings-note">默认宋体与 Times New Roman。选择自定义字体可填写本机已安装的名称。</p>
<div class="settings-section"><h4>界面字体</h4><div class="font-fields">
<FontFamilySelect v-model="draft.zh" label="中文字体" :names="names" :labels="fontLabels" :reset-token="resetToken"/>
<FontFamilySelect v-model="draft.latin" label="英文与数字字体" :names="names" :labels="fontLabels" :reset-token="resetToken"/>
<FontFamilySelect id="mono-choice" v-model="draft.mono" label="代码与等宽字体" custom-label="自定义代码字体" :names="monoNames" :labels="{'JetBrains Mono':'内置'}" :reset-token="resetToken"/>
<label class="fixed-font"><span>所有 IPA 音标</span><input value="Doulos SIL（固定）" aria-label="IPA 字体" readonly/></label>
</div><p class="settings-note">JetBrains Mono 与 IPA 字体已内置，无需安装。</p></div>
<label class="setting-row body-size-row">正文基础字号（px）<input v-model.number="draft.bodySize" type="number" min="10" max="24" step="1" aria-label="正文基础字号"/></label>
<div class="settings-section"><div class="font-chart-heading"><h4>图表与导出</h4><label class="font-follow"><input v-model="draft.figure.follow" type="checkbox" aria-label="图表与导出跟随全局字体"/>跟随全局字体</label></div>
<div v-if="!draft.figure.follow" class="font-fields"><FontFamilySelect v-model="draft.figure.zh" label="图表中文字体" :names="names" :labels="fontLabels" :reset-token="resetToken"/><FontFamilySelect v-model="draft.figure.latin" label="图表英文字体" :names="names" :labels="fontLabels" :reset-token="resetToken"/></div>
<label class="setting-row">图表基础字号（px）<input v-model.number="draft.figure.size" type="number" min="10" max="24" step="1" aria-label="图表基础字号"/></label>
<p class="settings-note">导出优先使用所选字体，缺失时使用兼容字体并记录实际字体。</p></div>
<div class="settings-section font-preview-section"><h4>字体预览</h4>
<div class="font-preview" :style="preview"><span>声学参数 · 中文字体预览</span><span>PhoneticToolbox 0123456789 −12.5 Hz</span><span class="ipa-text">[aː tʰ ɕ ŋ ə ã n̩ ɑ²]</span><code class="font-code-preview" :style="{fontFamily:preview['--preview-mono']}">f0 = signal[0:10]  # Hz != ms</code><svg viewBox="0 0 420 72" role="img" aria-label="图表字体预览" :style="{fontFamily:preview['--preview-figure']||'var(--font-figure)',fontSize:preview['--preview-size']||'var(--figure-size)'}"><path d="M40 5V42H405" fill="none" stroke="currentColor"/><text x="48" y="24" fill="currentColor">基频 F0 · −12.5 Hz</text><text x="240" y="64" fill="currentColor">时间 Time (s)</text></svg></div>
<p class="settings-note">预览后点击应用字体，设置将保存到本机。</p></div>
<p v-if="message||fontError" role="status">{{message||fontError}}</p>
<div class="font-actions"><button @click="draft=defaults();resetToken++;message='已填入默认设置，点击应用字体后生效。'" :disabled="busy">恢复默认</button><button @click="cancel" :disabled="busy">取消字体编辑</button><button @click="show" :disabled="busy">预览字体</button><button class="primary" @click="apply" :disabled="busy">{{busy?'检查字体…':'应用字体'}}</button></div>
</section></template>
<style scoped>
.font-settings{display:flex;flex-direction:column;gap:14px;align-self:start}.font-settings .section-title{margin:0}.font-settings .section-title button{font-size:0.857143rem;min-height:28px;padding:4px 8px}.font-fields{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}.fixed-font{display:flex;flex-direction:column;gap:7px;min-width:0}.fixed-font input{width:100%;height:36px;background:var(--app);color:var(--muted)}.font-chart-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.font-follow{display:flex;align-items:center;gap:7px;font-size:0.857143rem}.font-preview{display:grid;grid-template-columns:1fr;gap:6px;padding:12px 14px;background:var(--app);border:1px solid var(--border);border-radius:7px;line-height:1.65;overflow-wrap:anywhere}.font-preview svg{width:100%;height:65px;max-height:65px}.font-code-preview{font-family:var(--preview-mono,var(--font-mono));font-size:0.857143rem;font-variant-ligatures:none}.font-actions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;border-top:1px solid var(--border);padding-top:14px}.font-actions button{font-size:0.857143rem;padding:5px 9px}.font-settings .setting-row{display:flex;align-items:center;justify-content:space-between;margin:0;gap:12px}.font-settings .setting-row input{width:80px}.font-settings [role=status]{font-size:0.857143rem;color:var(--accent)}@container module (max-width:760px){.font-fields{grid-template-columns:1fr}}
</style>
