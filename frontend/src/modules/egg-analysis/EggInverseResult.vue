<script setup lang="ts">
import {computed,ref} from 'vue';
import ScientificPlot from '../../components/ScientificPlot.vue';
import type {components} from '../../../../contracts/generated/api';
import {scientificPlotsPng,downloadPng} from '../../design/plot-export.ts';
import {inverseLayouts,inversePanels,type InverseLayout} from './inverse-plots.ts';
import {plotTickLabel} from '../../platform/plotTicks.ts';
const props=defineProps<{data:components['schemas']['EggInverseData']}>();
const layout=ref<InverseLayout>('audio-if'),grid=ref<HTMLElement>(),saving=ref(false),error=ref('');
const panels=computed(()=>inversePanels(props.data,layout.value));
const axisLabelWidth=computed(()=>Math.max(6,...panels.value.flatMap(p=>[p.y,...(p.right?[p.right]:[])].flatMap(r=>Array.from({length:5},(_,i)=>plotTickLabel(r[0]+i*(r[1]-r[0])/4).length)))));
const reserveRightAxis=computed(()=>panels.value.some(p=>p.right));
async function save(index?:number){
  if(!grid.value||saving.value)return;
  saving.value=true;error.value='';
  const selectedLayout=layout.value;
  try{
    const elements=[...grid.value.querySelectorAll<HTMLElement>(':scope>section')];
    const selected=index==null?elements:[elements[index]];
    downloadPng(await scientificPlotsPng(selected),`EGG-逆滤波-${selectedLayout}${index==null?'':`-${index+1}`}.png`);
  }catch(e){error.value=String(e);}finally{saving.value=false;}
}
defineExpose({save});
</script>
<template>
  <div class="inverse-actions"><button class="primary" @click="save()" :disabled="saving">保存当前 {{panels.length}} 图 PNG</button><slot name="actions"/><span v-if="saving" role="status">正在生成图片…</span><span v-if="error" role="alert">{{error}}</span></div>
  <div class="inverse-layouts" role="group" aria-label="逆滤波图形组合"><button v-for="item in inverseLayouts" :key="item.id" :aria-pressed="layout===item.id" :disabled="saving" @click="layout=item.id">{{item.label}}</button></div>
  <p v-if="data.fixed_window_crossings" class="hint" role="status">此结果使用 {{data.lp_order}} 阶 LPC。{{data.gci_count}} 个 GCI 起点中，{{data.fixed_window_crossings}} 个固定 3 ms 窗覆盖下一检测 GCI，闭相取窗假设不满足。IF 仅作为当前简化算法的探索性结果。</p>
  <div ref="grid" class="inverse-grid">
    <section v-for="(panel,index) in panels" :key="panel.id" :data-panel-id="panel.id">
      <h3>{{panel.title}}</h3>
      <ScientificPlot :height="300" :title="panel.title" :x="panel.x" :y="panel.y" :right="panel.right" :axis-label-width="axisLabelWidth" :reserve-right-axis="reserveRightAxis" :unit="panel.unit" :right-unit="panel.rightUnit" :x-unit="panel.xUnit" :traces="panel.traces"/>
      <button class="primary" @click="save(index)" :disabled="saving">保存此图 PNG</button>
    </section>
  </div>
  <p class="hint">音频实线、IF 虚线、EGG 点线。双信号波形使用左右轴，三信号波形各自按中心窗峰值归一化，仅用于形状比较。频谱保留原幅度标尺，各谱峰下 80 dB 截底。</p>
</template>
<style scoped>
.inverse-actions,.inverse-layouts{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.inverse-actions{position:sticky;top:0;z-index:2;background:var(--panel);padding:6px 0}.inverse-layouts{margin-top:8px}.inverse-layouts button[aria-pressed=true]{background:var(--selected);border-color:var(--accent);color:var(--accent)}.inverse-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;margin-top:12px}.inverse-grid section{min-width:0;border:1px solid var(--border);border-radius:var(--radius);padding:8px}.inverse-grid h3{text-align:center;font-size:var(--figure-size,12px);margin:2px 0 6px}@media(max-width:850px){.inverse-grid{grid-template-columns:minmax(0,1fr)}}
</style>
