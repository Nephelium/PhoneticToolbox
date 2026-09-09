<script setup lang="ts">
import { computed,ref } from 'vue';import type { Module } from './registry.ts';import { workspace,host,assignAsset,states } from '../state/workspace.ts';import { stop } from '../state/audio.ts';
import AppIcon from '../components/AppIcon.vue';import WaveformViewport from '../components/WaveformViewport.vue';import ParameterDrawer from '../components/ParameterDrawer.vue';import TaskPanel from '../components/TaskPanel.vue';
const props=defineProps<{module:Module}>();const emit=defineEmits<{references:[]}>();const state=computed(()=>workspace(props.module.id));const drawer=ref(false),picker=ref<HTMLInputElement>();
async function load(file:File){const id=props.module.id,s=states[id];if(!s)return;s.loading=true;s.error='';stop();try{const asset=await host.files.load(file);if(states[id]!==s)return;if(id==='M03'&&asset.channels.length!==2)throw Error('EGG 需要双声道 WAV：一个声道为 EGG，另一个为音频。');assignAsset(id,asset);}catch(e){s.error=e instanceof Error?e.message:'文件读取失败，请重新选择。';}finally{s.loading=false;}}
function pick(e:Event){const input=e.target as HTMLInputElement;if(input.files?.[0])void load(input.files[0]);input.value='';}
async function demo(){const currentState=state.value;currentState.loading=true;try{const response=await fetch(new URL('../assets/SYN-EGG-44100.wav',import.meta.url).href);if(!response.ok)throw Error('测试音频读取失败');await load(new File([await response.arrayBuffer()],'公开测试音频 · 双声道.wav'));}catch(e){currentState.error=String(e);}finally{currentState.loading=false;}}
</script>
<template>
<section class="workspace-page" :aria-label="module.title+' 工作区'">
<header class="page-heading">
<div>
<p class="eyebrow">{{module.id}} · 研究工具</p>
<h1>{{module.title}}</h1>
<p class="muted">{{module.description}}</p>
</div>
<button @click="emit('references')">
<AppIcon name="book"/>方法与引用</button>
</header>
<div class="notice">
<span class="badge">待接入</span>
<span>本模块的分析功能尚未迁入。下方可试用共用的文件预览、选区和试听。</span>
</div>
<div class="workspace-toolbar">
<input ref="picker" type="file" accept=".wav,audio/wav" class="visually-hidden" aria-label="选择 WAV 音频" @change="pick"/>
<button class="primary" :disabled="state.loading" @click="picker?.click()">
<AppIcon name="folder"/>{{state.loading?'正在读取…':'打开 WAV'}}</button>
<button :disabled="state.loading" @click="demo">载入公开测试音频</button>
<span class="muted toolbar-note">{{host.kind==='desktop'?'本机文件':'本地预览 · 不上传文件'}}</span>
</div>
<div v-if="state.error" role="alert" class="error-banner">
<AppIcon name="info"/>
<span>{{state.error}} {{state.asset?'已加载的音频仍保留。':'请选择有效的音频文件。'}}</span>
<button @click="state.error=''">收起提示</button>
</div>
<div class="workbench-grid">
<aside class="file-panel">
<div class="panel-heading">
<h2>当前文件</h2>
<small>{{state.asset?1:0}} 个文件</small>
</div>
<div v-if="state.asset" class="file-row selected">
<AppIcon name="file"/>
<span>{{state.asset.name}}<small>WAV · 已加载</small>
</span>
</div>
<p v-else class="empty-small">尚未选择文件</p>
<p class="file-note">文件保留在本机。切换工具会保留当前预览，关闭标签会释放音频。</p>
</aside>
<div class="signal-panel">
<template v-if="state.asset">
<div class="signal-heading">
<h2>{{state.asset.name}}</h2>
<p class="mono muted">{{state.asset.sampleRate.toLocaleString()}} Hz · {{state.asset.channels.length}} 声道 · {{state.asset.duration.toFixed(3)}} s</p>
</div>
<label class="channel-picker">{{module.id==='M03'?'音频声道（用于试听）':'试听声道'}} <select v-model.number="state.channel" @change="stop">
<option v-for="(_,i) in state.asset.channels" :key="i" :value="i">声道 {{i+1}}{{i===0?' · 左':i===1?' · 右':''}}</option>
</select>
</label>
<p v-if="module.id==='M03'" class="hint">EGG：声道 {{state.channel===0?2:1}}；音频：声道 {{state.channel+1}}。请按实际录制顺序选择。</p>
<WaveformViewport :state="state"/>
</template>
<div v-else class="wave-empty">
<AppIcon name="wave"/>
<h2>让声音进入工作台</h2>
<p>打开一段 WAV，查看真实波形与时间选区。</p>
<p class="muted">也可以载入公开测试音频体验操作。</p>
</div>
</div>
<aside class="parameter-summary">
<h2>输出参数</h2>
<p class="muted">下一次分析的草稿</p>
<div class="parameter-count">
<strong>{{state.parameters.length}}</strong>
<span>/ 80 项</span>
</div>
<p class="hint">参数选择不会改变当前波形，也不会自动启动分析。</p>
<button @click="drawer=true">
<AppIcon name="sliders"/>选择全部参数</button>
<span v-if="state.dirty" class="draft-indicator">● 草稿尚未保存</span>
<hr/>
<h2>分析结果</h2>
<p class="empty-small">还没有分析结果</p>
</aside>
</div>
<TaskPanel/>
<ParameterDrawer v-if="drawer" :key="module.id" :selected="state.parameters" @close="drawer=false" @apply="state.parameters=$event;state.dirty=true;drawer=false"/>
</section>
</template>
