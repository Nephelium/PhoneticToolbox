<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
// Frontend view model only; the P06 JobClient will map actual task contracts here.
interface TaskView {id:string;title:string;status:'queued'|'running'|'cancel_requested'|'cancelled'|'succeeded'|'failed'|'interrupted';progress?:number;error?:string;canCancel?:boolean;canRetry?:boolean}
const props=withDefaults(defineProps<{tasks?:TaskView[];emptyText?:string;emptyTitle?:string}>(),{tasks:()=>[]});
const emit=defineEmits<{cancel:[id:string];retry:[id:string]}>();
const labels=localized(()=>({queued:tr('components.queued'),running:tr('components.running'),cancel_requested:tr('components.cancelling'),cancelled:tr('components.cancelled_443'),succeeded:tr('components.completed'),failed:tr('components.failed'),interrupted:tr('components.interrupted')}));
</script>
<template>
<section :aria-label="tr('components.analysis_tasks')" class="task-panel">
<template v-if="!tasks.length">
<span class="status-dot"/>
<strong>{{props.emptyTitle??tr('components.no_analysis_tasks_yet')}}</strong>
<span class="muted">{{props.emptyText??tr('components.you_can_preview_files_and_listen_to_audio_use')}}</span>
</template>
<article v-for="task in tasks" :key="task.id" class="task-row" :data-job-state="task.status">
<strong>{{task.title}}</strong>
<span role="status" :data-job-state="task.status">{{labels[task.status]}}</span>
<progress v-if="task.status==='running'&&task.progress!==undefined&&Number.isFinite(task.progress)&&task.progress>=0&&task.progress<=1" :value="task.progress" max="1" :aria-label="task.title+tr('components.progress')"/>
<p v-if="task.error" class="error-text" role="alert">{{task.error}}</p>
<button v-if="task.canCancel&&['queued','running'].includes(task.status)" @click="emit('cancel',task.id)">{{tr('components.cancel_this_task')}}</button>
<button v-if="task.canRetry&&['failed','interrupted','cancelled'].includes(task.status)" @click="emit('retry',task.id)">{{tr('components.retry_this_task')}}</button>
</article>
</section>
</template>
