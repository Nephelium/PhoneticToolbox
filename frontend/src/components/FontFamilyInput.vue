<script setup lang="ts">
import {ref,computed,nextTick,useId} from 'vue';
import {useFloatingPicker} from './floating-picker.ts';
const value=defineModel<string>({required:true});
const props=defineProps<{label:string;names:string[];labels?:Record<string,string>}>();
const anchor=ref<HTMLElement>(),panel=ref<HTMLElement>(),input=ref<HTMLInputElement>();
const {open,position}=useFloatingPicker(anchor,panel),id=useId(),active=ref(0);
const options=computed(()=>[...new Set(['',...props.names,...(value.value?[value.value]:[])])]);
function optionLabel(name:string){return name?(props.labels?.[name]?name+' · '+props.labels[name]:name):'系统默认';}
async function show(){open.value=true;active.value=Math.max(0,options.value.indexOf(value.value));await nextTick();reveal();}
function reveal(){panel.value?.querySelectorAll<HTMLElement>('[role=option]')[active.value]?.scrollIntoView({block:'nearest'});}
function choose(name:string){value.value=name;open.value=false;input.value?.focus();}
async function keyboard(event:KeyboardEvent){
 if(event.key==='ArrowDown'||event.key==='ArrowUp'){
  event.preventDefault();if(!open.value){await show();return;}
  active.value=(active.value+(event.key==='ArrowDown'?1:-1)+options.value.length)%options.value.length;reveal();
 }else if(event.key==='Enter'&&open.value){event.preventDefault();choose(options.value[active.value]);}
 else if(event.key==='Tab')open.value=false;
}
</script>
<template><div class="font-family-field"><label :for="id">{{label}}</label><div ref="anchor" class="font-family-control">
 <input ref="input" :id="id" v-model="value" :aria-label="label" role="combobox" :aria-expanded="open" :aria-controls="id+'-list'" aria-autocomplete="none" :aria-activedescendant="open?id+'-'+active:undefined" placeholder="系统默认" @keydown="keyboard"/>
 <button type="button" :aria-label="'展开'+label+'列表'" :aria-expanded="open" :aria-controls="id+'-list'" @click="open?open=false:show()">⌄</button>
 </div><Teleport to="body"><div v-if="open" ref="panel" :id="id+'-list'" class="font-family-list" role="listbox" :aria-label="label+'完整列表'" :style="position" @keydown="keyboard">
 <button v-for="(name,index) in options" :id="id+'-'+index" :key="name" type="button" role="option" :aria-selected="name===value" :class="{highlight:index===active}" @pointerenter="active=index" @pointerdown.prevent @click="choose(name)"><span>{{optionLabel(name)}}</span><span v-if="name===value" aria-hidden="true">✓</span></button>
 </div></Teleport></div></template>
<style scoped>
.font-family-field{display:grid;gap:5px;min-width:0}.font-family-control{display:flex;min-width:0}.font-family-control input{flex:1;width:0;border-radius:6px 0 0 6px}.font-family-control button{flex:none;width:34px;padding:0;border-left:0;border-radius:0 6px 6px 0}
.font-family-list{position:fixed;z-index:180;overflow:auto;overscroll-behavior:contain;padding:5px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow);scrollbar-width:thin;font:var(--control-size,14px) var(--font)}
.font-family-list button{display:flex;justify-content:space-between;width:100%;min-height:30px;border:0;border-radius:4px;padding:5px 8px;text-align:left;white-space:normal;overflow-wrap:anywhere;font:inherit;box-shadow:none}.font-family-list button span:first-child{min-width:0}.font-family-list button.highlight,.font-family-list button[aria-selected=true]{background:var(--selected);color:var(--accent)}
</style>
