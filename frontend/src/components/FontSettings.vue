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
async function list(){busy.value=true;message.value='';try{
 if(desktopFontFamilies)extra.value=await desktopFontFamilies();
 else{const api=(window as unknown as {queryLocalFonts?:()=>Promise<{family:string}[]>}).queryLocalFonts;if(!api)throw Error('此浏览器不支持字体列表读取，可直接填写本机已安装的字体名称。');extra.value=(await api()).map(f=>f.family);}
 message.value=`已读取 ${extra.value.length} 个字体名称，应用时会检查所选字体。`;
}catch(e){message.value=e instanceof Error?e.message:'无法读取字体列表。';}finally{busy.value=false;}}
async function show(){busy.value=true;message.value='';try{const p=await prepareFonts(normalizeFonts(draft.value));preview.value={fontFamily:p.ui,'--preview-mono':p.mono,'--preview-figure':p.figure,'--preview-size':p.size+'px'};message.value='预览已更新，点击应用字体后生效。';}catch(e){message.value=(e as Error).message;}finally{busy.value=false;}}
async function apply(){if(busy.value)return false;busy.value=true;message.value='';try{if(await setFonts(draft.value)){draft.value=normalizeFonts(preferences.value);message.value='字体已应用到工作台与图表，并保存在本机。';return true;}return false;}catch(e){message.value=(e as Error).message;return false;}finally{busy.value=false;}}
function cancel(){draft.value=normalizeFonts(preferences.value);preview.value={};message.value='已取消本次字体编辑。';}
watch(()=>busy.value||JSON.stringify(normalizeFonts(draft.value))!==JSON.stringify(normalizeFonts(preferences.value)),value=>emit('dirty',value),{immediate:true});
defineExpose({save:apply});
</script>
<template><section class="font-settings" aria-label="字体设置">
<div class="section-title"><h3>字体</h3><button @click="list" :disabled="busy">读取本机字体列表</button></div>
<p class="muted">留空使用系统默认。可选择候选字体或填写已安装字体的名称，应用时检查可用性。</p>
<datalist id="ptb-font-options"><option v-for="name in names" :key="name" :value="name" :label="fontLabels[name]||name"/></datalist>
<div class="font-fields"><label>中文字体<input v-model="draft.zh" list="ptb-font-options" aria-label="中文字体" placeholder="系统默认"/></label><label>英文与数字字体<input v-model="draft.latin" list="ptb-font-options" aria-label="英文与数字字体" placeholder="系统默认"/></label><label>代码与等宽字体<input v-model="draft.mono" list="ptb-font-options" aria-label="代码与等宽字体" placeholder="系统默认"/></label><label>所有 IPA 音标<input value="Doulos SIL（固定）" aria-label="IPA 字体" readonly/></label></div>
<label class="font-follow"><input v-model="draft.figure.follow" type="checkbox"/>图表与导出跟随全局字体</label>
<p class="muted">后台导出优先使用所选字体；计算设备缺少该字体时使用可用的兼容字体，并在结果中记录实际字体。IPA 保持 Doulos SIL。</p>
<div v-if="!draft.figure.follow" class="font-fields"><label>图表中文字体<input v-model="draft.figure.zh" list="ptb-font-options" aria-label="图表中文字体" placeholder="系统默认"/></label><label>图表英文字体<input v-model="draft.figure.latin" list="ptb-font-options" aria-label="图表英文字体" placeholder="系统默认"/></label></div>
<label class="setting-row">图表基础字号（px）<input v-model.number="draft.figure.size" type="number" min="10" max="24" step="1" aria-label="图表基础字号"/></label>
<div class="font-preview" :style="preview"><span>中文字体预览 · 声学参数</span><span>PhoneticToolbox 0123456789 −12.5 Hz</span><span class="ipa-text">[aː tʰ ɕ ŋ ə ã n̩ ɑ²]</span><code :style="{fontFamily:preview['--preview-mono']}">f0 = signal[0:10]  # Hz != ms</code><svg viewBox="0 0 420 72" role="img" aria-label="图表字体预览" :style="{fontFamily:preview['--preview-figure']||'var(--font-figure)',fontSize:preview['--preview-size']||'var(--figure-size)'}"><path d="M40 5V42H405" fill="none" stroke="currentColor"/><text x="48" y="24" fill="currentColor">基频 F0 · −12.5 Hz</text><text x="240" y="64" fill="currentColor">时间 Time (s)</text></svg></div>
<p class="muted">IPA 始终使用 Doulos SIL。字体应用于应用页面与新生成的图像，系统文件选择框和已有图片保留原样。</p>
<p v-if="message||fontError" role="status">{{message||fontError}}</p>
<div class="font-actions"><button @click="draft=defaults();message='已填入默认设置，点击应用字体后生效。'" :disabled="busy">恢复默认</button><button @click="cancel" :disabled="busy">取消字体编辑</button><button @click="show" :disabled="busy">预览字体</button><button class="primary" @click="apply" :disabled="busy">{{busy?'检查字体…':'应用字体'}}</button></div>
</section></template>
<style scoped>
.font-settings{border-top:1px solid var(--border);margin-top:20px;padding-top:8px}.font-fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.font-fields label{display:grid;gap:6px;min-width:0}.font-fields input{width:100%;min-width:0}.font-follow{display:flex;align-items:center;gap:8px;margin:16px 0}.font-preview{display:grid;gap:10px;padding:16px;background:var(--app);border:1px solid var(--border);border-radius:8px;line-height:1.7;overflow-wrap:anywhere}.font-preview svg{width:100%;height:auto;max-height:90px}.font-actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.font-settings .muted{line-height:1.7;font-size:12px}@media(max-width:600px){.font-fields{grid-template-columns:1fr}}
</style>
