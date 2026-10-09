<script setup lang="ts">
import type {Tier} from '../platform/research.ts';
import {intervalLayout} from './interval-layout.ts';
withDefaults(defineProps<{tiers:Tier[];start:number;end:number;selected:number;selectionStart:number;selectionEnd:number;selectedTierLabel?:string}>(),{selectedTierLabel:'当前切分层'});
const emit=defineEmits<{select:[tier:number,start:number,end:number]}>();
</script>
<template><section class="textgrid-timeline" aria-label="TextGrid 时间对齐标注">
<div v-for="(tier,tierIndex) in tiers" :key="tierIndex" class="textgrid-tier" :class="{active:tierIndex===selected}">
<header><strong>{{tier.name}}</strong><small>{{tierIndex===selected?selectedTierLabel+' · ':''}}点击片段选区试听，缩放后可查看短标签</small></header>
<div class="textgrid-lane" :aria-label="tier.name+' 标注轨'">
<button v-for="interval in intervalLayout(tier.intervals,start,end)" :key="interval.index" class="textgrid-interval" :class="{selected:interval.xmin===selectionStart&&interval.xmax===selectionEnd}" :style="{left:interval.left+'%',width:interval.width+'%'}" :data-xmin="interval.xmin" :data-xmax="interval.xmax" :title="(interval.text||'（空标签）')+' · '+interval.xmin.toFixed(6)+'–'+interval.xmax.toFixed(6)+' s'" :aria-label="tier.name+'：'+(interval.text||'空标签')+'，'+interval.xmin+' 到 '+interval.xmax+' 秒'" @click="emit('select',tierIndex,interval.xmin,interval.xmax)"><span class="ipa-text">{{interval.text}}</span></button>
</div></div>
<div class="time-axis"><span v-for="i in 5" :key="i">{{(start+(i-1)*(end-start)/4).toFixed(end-start<.1?4:3)}}{{i===5?' s':''}}</span></div>
</section></template>
<style scoped>
.textgrid-timeline{margin-top:12px;min-width:0}.textgrid-tier{margin-top:7px}.textgrid-tier header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:4px;font-size:0.857143rem}.textgrid-tier.active header strong{color:var(--accent)}.textgrid-lane{position:relative;height:54px;width:100%;overflow:hidden;background:var(--panel);border-block:1px solid var(--border)}.textgrid-lane .textgrid-interval{position:absolute;top:0;bottom:0;min-width:0;min-height:0;max-width:none;padding:0;border:0;border-right:1px solid var(--muted);border-radius:0;overflow:hidden;background:transparent;color:var(--text)}.textgrid-interval span{display:block;overflow:hidden;white-space:nowrap;text-overflow:clip;font-family:var(--font-figure-ipa,var(--font-ipa));font-size:1rem;padding:0 2px}.textgrid-lane .textgrid-interval:hover,.textgrid-lane .textgrid-interval.selected{background:var(--selected)}.textgrid-lane .textgrid-interval:focus-visible{outline-offset:-2px;z-index:1}
</style>
