<script setup lang="ts">
import {computed,ref,useId,useSlots} from 'vue';
import {vResizablePanels} from '../layout/resizablePanels.ts';
import {layoutKey} from '../layout/panelWidths.ts';
import {projects} from '../platform/browser.ts';

// Layout only. Modules retain ownership of inputs, jobs, playback and exports.
const props=withDefaults(defineProps<{
 stateKey:string;leftLabel?:string;centerLabel?:string;rightLabel?:string;unified?:boolean;
 leftWidth?:number;rightWidth?:number;leftMin?:number;rightMin?:number;centerMin?:number;leftWidthAlias?:string;legacyStateKey?:string;
}>(),{leftLabel:'输入与参数',centerLabel:'工作区',rightLabel:'任务与记录',leftWidth:240,rightWidth:300,leftMin:200,rightMin:240,centerMin:360});
const key=layoutKey(props.stateKey)+'.workbench-right-collapsed';
const collapsed=ref(projects.read<boolean>(key,false)===true);
const rightId='workbench-right-'+useId();
const slots=useSlots();
function toggle(){collapsed.value=!collapsed.value;projects.write(key,collapsed.value);}
const resize=computed(()=>({key:props.stateKey+'.workbench',legacyKey:props.legacyStateKey?props.legacyStateKey+'.workbench':undefined,center:':scope > .workbench-center',centerMin:props.centerMin,
 panels:[
  ...(slots.left?[{selector:':scope > .workbench-left',side:'left' as const,label:props.leftLabel,initial:props.unified?300:props.leftWidth,min:props.leftMin,max:520,savedVariable:props.leftWidthAlias}]:[]),
  ...(slots.right?[{selector:':scope > .workbench-right',side:'right' as const,label:props.rightLabel,initial:props.unified?300:props.rightWidth,min:props.rightMin,max:560}]:[]),
 ]}));
</script>

<template>
 <div v-resizable-panels="resize" class="module-workbench" :class="{'workbench-unified':unified,'right-collapsed':collapsed,'without-left':!$slots.left,'without-right':!$slots.right}">
  <aside v-if="$slots.left" class="workbench-left" :aria-label="leftLabel"><slot name="left"/></aside>
  <div class="workbench-center" :aria-label="centerLabel"><slot/></div>
  <aside v-if="$slots.right" class="workbench-right" :aria-label="rightLabel">
   <div class="workbench-right-heading">
    <strong v-if="!collapsed">{{rightLabel}}</strong>
    <button type="button" :aria-expanded="!collapsed" :aria-controls="rightId" :aria-label="(collapsed?'展开':'收起')+rightLabel" :title="(collapsed?'展开':'收起')+rightLabel" @click="toggle">{{collapsed?'‹':'›'}}</button>
   </div>
   <div v-show="!collapsed" :id="rightId" class="workbench-right-body"><slot name="right"/></div>
  </aside>
 </div>
</template>

<style scoped>
.module-workbench{display:grid;grid-template-columns:var(--panel-left,240px) minmax(0,1fr) var(--panel-right,300px);gap:var(--module-gap);flex:1;min-width:0;min-height:0;align-items:stretch}
.workbench-left,.workbench-center,.workbench-right-body{display:flex;flex-direction:column;gap:var(--module-gap);min-width:0;min-height:0;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin}
.workbench-left,.workbench-center{height:100%}
.workbench-left>:deep(*),.workbench-center>:deep(*),.workbench-right-body>:deep(*){min-width:0;flex-shrink:0}
.workbench-right{display:flex;flex-direction:column;gap:var(--control-gap);min-width:0;min-height:0;border-left:1px solid var(--border);padding-left:var(--module-gap)}
.workbench-right-heading{display:flex;align-items:center;justify-content:space-between;gap:var(--control-gap);flex:none;color:var(--text)}
.workbench-right-heading button{width:30px;padding:2px;flex:none;font-size:20px;line-height:1}
.workbench-right-body{flex:1}
.right-collapsed{grid-template-columns:var(--panel-left,240px) minmax(0,1fr) 42px}
.without-left{grid-template-columns:minmax(0,1fr) var(--panel-right,300px)}
.without-left.right-collapsed{grid-template-columns:minmax(0,1fr) 42px}
.without-right{grid-template-columns:var(--panel-left,240px) minmax(0,1fr)}
.without-left.without-right{grid-template-columns:minmax(0,1fr)}
.workbench-unified>.workbench-left,.workbench-unified>.workbench-right{border:1px solid var(--border);border-radius:var(--radius);background:var(--app);padding:var(--workbench-pane-padding,8px)}
.workbench-unified>.workbench-left,.workbench-unified .workbench-right-body{scrollbar-gutter:stable}
.workbench-unified .workbench-right-heading{min-height:var(--control-height);padding-bottom:var(--control-gap);border-bottom:1px solid var(--border);font-size:var(--control-size)}
.workbench-unified .workbench-right-heading button{min-height:var(--control-height)}
.workbench-unified :deep(.workbench-card){background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:10px}
.workbench-unified>.workbench-left :deep(button),.workbench-unified>.workbench-right :deep(button){max-width:100%;white-space:normal;overflow-wrap:anywhere}
.workbench-unified>.workbench-left :deep(select),.workbench-unified>.workbench-right :deep(select){min-width:0;max-width:100%}
.workbench-unified .workbench-right-body :deep(button.primary){width:100%}
.workbench-unified.right-collapsed>.workbench-right{padding:4px}
.workbench-unified.right-collapsed .workbench-right-heading{border-bottom:0;padding:0;justify-content:center}
@container module (max-width:1060px){
 .module-workbench{grid-template-columns:minmax(180px,var(--panel-left,220px)) minmax(280px,1fr);flex:none;min-height:min-content;align-items:start}
 .workbench-left,.workbench-center{height:auto;overflow:visible}
 .workbench-right{grid-column:1/-1;border-left:0;border-top:1px solid var(--border);padding:var(--module-gap) 0 0}
 .workbench-right-body{max-height:360px}
 .module-workbench.without-left{grid-template-columns:minmax(0,1fr)}
}
@container module (max-width:720px){
 .module-workbench{display:flex;flex-direction:column}
 .workbench-left,.workbench-center,.workbench-right{width:100%}
 .workbench-right-body{max-height:360px}
}
</style>
