<script setup lang="ts">
import {vTimePrecision} from '../../design/time-precision.ts';
import {eggDefaults,type Extended} from './state.ts';
const props=defineProps<{value:Extended;mono:boolean;selected:boolean;override:number|undefined}>();
const emit=defineEmits<{override:[value:number|undefined]}>();
function enable(event:Event){props.value.egg=(event.target as HTMLInputElement).checked?eggDefaults():null;if(props.value.egg)props.value.audio_channel=1;}
function channel(){if(props.value.egg)props.value.egg.egg_channel=1-(props.value.audio_channel??1);}
</script>
<template><section class="m01-joint" aria-label="音频与EGG联合分析">
<h2>分析声道</h2>
<label>默认音频声道<select v-model="value.audio_channel" aria-label="默认分析声道" @change="channel"><option v-if="!value.egg" :value="null">全部声道混合</option><option :value="0">声道 1（左）</option><option :value="1">声道 2（右）</option></select></label>
<label class="check"><input type="checkbox" :checked="!!value.egg" @change="enable"/>联合计算 EGG</label>
<p class="hint">最长 30 分钟，自动分段计算。分析声道与试听选择分别设置。</p>
<template v-if="value.egg">
<p class="hint">默认 EGG：声道 {{(value.egg.egg_channel??0)+1}}。单声道文件仅计算声学参数。</p>
<label v-if="selected&&!mono">当前文件声道<select :value="override??''" aria-label="当前文件分析声道" @change="emit('override',($event.target as HTMLSelectElement).value===''?undefined:Number(($event.target as HTMLSelectElement).value))"><option value="">跟随默认</option><option value="0">音频 1 / EGG 2</option><option value="1">音频 2 / EGG 1</option></select></label>
<p v-if="mono" class="hint">当前文件为单声道，将跳过 EGG。</p>
<label>EGG 参数表<select v-model="value.egg.storage" aria-label="EGG参数保存方式"><option value="aligned">插值到主表，并保留原始周期表</option><option value="cycles">原始周期独立表（同一文件）</option></select></label>
<label v-if="value.egg.storage==='aligned'">插值后平滑（ms，0 关闭）<input v-time-precision="'ms'" v-model.number="value.egg.smooth_ms" aria-label="EGG平滑毫秒" type="number" min="0" max="1000" step="5"/></label>
<details><summary>EGG 计算设置</summary>
<label>高通（Hz）<input v-model.number="value.egg.highpass_cutoff" type="number" min="1" max="10000"/></label>
<label>低通（Hz）<input v-model.number="value.egg.lowpass_cutoff" type="number" min="1" max="48000"/></label>
<label>GCI 方法<select v-model="value.egg.gci_method"><option value="slope">最大斜率</option><option value="scale">比例阈值</option></select></label>
<label>GOI 方法<select v-model="value.egg.goi_method"><option value="slope">最大斜率</option><option value="scale">比例阈值</option></select></label>
<label class="check"><input v-model="value.egg.auto_prominence" type="checkbox"/>自动峰谷突出度</label>
<template v-if="!value.egg.auto_prominence"><label>峰突出度<input v-model.number="value.egg.peak_prominence" type="number" min="0" max="1" step=".001"/></label><label>谷突出度<input v-model.number="value.egg.valley_prominence" type="number" min="0" max="1" step=".001"/></label></template>
<label>音频静音阈值<input v-model.number="value.egg.silence_threshold" type="number" min="0" max="1" step=".001"/></label>
<label>最大插值间隔（ms）<input v-time-precision="'ms'" v-model.number="value.egg.max_gap_ms" type="number" min="1" max="1000"/></label>
<label class="check"><input v-model="value.egg.derived" type="checkbox"/>计算 GCI F0 派生声学参数</label>
<p class="hint">添加 gF0 谐波、CPP、HNR 等列。pF0 / rF0 列保留，Jitter / Shimmer 仍沿用现有方法。无效周期和长间断保持空缺。</p>
</details></template>
</section></template>
<style scoped>
.m01-joint{border-top:1px solid var(--border);margin-top:12px;padding-top:10px;min-width:0}.m01-joint h2{font-size:1rem;margin:0 0 8px}.m01-joint label{display:grid;gap:4px;margin:8px 0}.m01-joint .check{display:flex;align-items:center;gap:6px}.m01-joint select,.m01-joint input:not([type=checkbox]){width:100%;min-width:0}.m01-joint summary{cursor:pointer;margin:8px 0}.m01-joint .hint{font-size:0.857143rem;line-height:1.5}
</style>
