<script setup lang="ts">
import {computed} from 'vue';
import 'katex/dist/katex.min.css';
import {renderFormula} from './formula.ts';
const props=defineProps<{latex:unknown;display?:unknown;caption?:unknown}>();
const rendered=computed(()=>renderFormula(props.latex,props.display));
</script>
<template>
  <component :is="display?'figure':'span'" class="manual-formula" :class="{'manual-formula-inline':!display}">
    <span v-if="rendered.html" class="manual-formula-rendered" :aria-label="'公式：'+rendered.source" v-html="rendered.html"/>
    <template v-else><code>{{rendered.source}}</code><small role="status">{{rendered.error}}</small></template>
    <figcaption v-if="caption&&display">{{caption}}</figcaption>
  </component>
</template>
