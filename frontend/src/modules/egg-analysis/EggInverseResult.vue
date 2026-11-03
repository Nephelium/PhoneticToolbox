<script setup lang="ts">
import {uiErrorMessage} from '../module-localization.ts';
import {tr,message,messageRef} from '../../i18n/index.ts';
import {computed,ref} from 'vue';
import ScientificPlot from '../../components/ScientificPlot.vue';
import type {components} from '../../../../contracts/generated/api';
import {scientificPlotsPng,downloadPng} from '../../design/plot-export.ts';
import {inverseLayouts,inversePanels,type InverseLayout} from './inverse-plots.ts';
import {plotTickLabel} from '../../platform/plotTicks.ts';
const props=defineProps<{data:components['schemas']['EggInverseData']}>();
const layout=ref<InverseLayout>('audio-if'),grid=ref<HTMLElement>(),saving=ref(false),error=messageRef('');
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
  }catch(e){error.value=uiErrorMessage(e);}finally{saving.value=false;}
}
defineExpose({save});
</script>
<template>
  <div class="inverse-actions"><button class="primary" @click="save()" :disabled="saving">{{tr('modules.egg-analysis.save_current')}} {{panels.length}} {{tr('modules.egg-analysis.plots_as_png')}}</button><slot name="actions"/><span v-if="saving" role="status">{{tr('modules.egg-analysis.generating_image')}}</span><span v-if="error" role="alert">{{error}}</span></div>
  <div class="inverse-layouts" role="group" :aria-label="tr('modules.egg-analysis.inverse_filtering_plot_layout')"><button v-for="item in inverseLayouts" :key="item.id" :aria-pressed="layout===item.id" :disabled="saving" @click="layout=item.id">{{item.label}}</button></div>
  <p v-if="data.fixed_window_crossings" class="hint" role="status">{{tr('modules.egg-analysis.inverseWindowWarning',{order:data.lp_order,gci:data.gci_count,crossings:data.fixed_window_crossings})}}</p>
  <div ref="grid" class="inverse-grid">
    <section v-for="(panel,index) in panels" :key="panel.id" :data-panel-id="panel.id">
      <h3>{{panel.title}}</h3>
      <ScientificPlot :height="300" :title="panel.title" :x="panel.x" :y="panel.y" :right="panel.right" :axis-label-width="axisLabelWidth" :reserve-right-axis="reserveRightAxis" :unit="panel.unit" :right-unit="panel.rightUnit" :x-unit="panel.xUnit" :traces="panel.traces"/>
      <button class="primary" @click="save(index)" :disabled="saving">{{tr('modules.egg-analysis.save_this_plot_as_png')}}</button>
    </section>
  </div>
  <p class="hint">{{tr('modules.egg-analysis.audio_solid_if_dashed_egg_dotted_two_signal_waveforms')}}</p>
</template>
<style scoped>
.inverse-actions,.inverse-layouts{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.inverse-actions{position:sticky;top:0;z-index:2;background:var(--panel);padding:6px 0}.inverse-layouts{margin-top:8px}.inverse-layouts button[aria-pressed=true]{background:var(--selected);border-color:var(--accent);color:var(--accent)}.inverse-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;margin-top:12px}.inverse-grid section{min-width:0;border:1px solid var(--border);border-radius:var(--radius);padding:8px}.inverse-grid h3{text-align:center;font-size:var(--figure-size,12px);margin:2px 0 6px}@media(max-width:850px){.inverse-grid{grid-template-columns:minmax(0,1fr)}}
</style>
