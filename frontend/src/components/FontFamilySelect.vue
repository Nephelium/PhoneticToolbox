<script setup lang="ts">
import {computed,ref,watch} from 'vue';
const props=defineProps<{modelValue:string;label:string;names:string[];labels?:Record<string,string>;id?:string;customLabel?:string;resetToken?:number}>();
const emit=defineEmits<{'update:modelValue':[value:string]}>();
const custom=ref(false);
const options=computed(()=>[...new Set([...props.names,...(props.modelValue?[props.modelValue]:[])])]);
watch(()=>props.resetToken,()=>{custom.value=false;});
function choose(event:Event){const value=(event.target as HTMLSelectElement).value;custom.value=value==='__custom__';if(!custom.value)emit('update:modelValue',value);}
</script>
<template><label class="font-family-select"><span>{{label}}</span>
<select :id="id" :value="custom?'__custom__':modelValue" :aria-label="label" @change="choose">
 <option v-for="name in options" :key="name" :value="name">{{name}}{{labels?.[name]?' · '+labels[name]:''}}</option>
 <option value="">系统默认</option><option value="__custom__">自定义字体…</option>
</select>
<input v-if="custom" :value="modelValue" :aria-label="customLabel||'自定义'+label" placeholder="填写已安装的字体名称" @input="emit('update:modelValue',($event.target as HTMLInputElement).value)"/>
</label></template>
<style scoped>
.font-family-select{display:flex;flex-direction:column;gap:7px;min-width:0}.font-family-select>span{font-size:var(--control-size);line-height:1.5}.font-family-select select,.font-family-select input{width:100%;min-width:0;height:36px}
</style>
