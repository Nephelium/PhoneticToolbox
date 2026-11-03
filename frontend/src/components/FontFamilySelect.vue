<script setup lang="ts">
import {tr} from '../i18n/index.ts';
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
 <option value="">{{tr('components.system_default')}}</option><option value="__custom__">{{tr('components.custom_font')}}</option>
</select>
<input v-if="custom" :value="modelValue" :aria-label="customLabel||tr('components.custom')+label" :placeholder="tr('components.enter_the_name_of_an_installed_font')" @input="emit('update:modelValue',($event.target as HTMLInputElement).value)"/>
</label></template>
<style scoped>
.font-family-select{display:flex;flex-direction:column;gap:7px;min-width:0}.font-family-select>span{font-size:var(--control-size);line-height:1.5}.font-family-select select,.font-family-select input{width:100%;min-width:0;height:36px}
</style>
