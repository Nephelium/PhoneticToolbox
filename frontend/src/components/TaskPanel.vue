<script setup lang="ts">
// Frontend view model only; the P06 JobClient will map actual task contracts here.
interface TaskView {id:string;title:string;status:'queued'|'running'|'cancelling'|'cancelled'|'succeeded'|'failed';progress?:number;error?:string;canCancel?:boolean;canRetry?:boolean}
withDefaults(defineProps<{tasks?:TaskView[]}>(),{tasks:()=>[]});
const emit=defineEmits<{cancel:[id:string];retry:[id:string]}>();
const labels={queued:'排队中',running:'运行中',cancelling:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败'};
</script>
<template>
<section aria-label="分析任务" class="task-panel">
<template v-if="!tasks.length">
<span class="status-dot"/>
<strong>暂无分析任务</strong>
<span class="muted">分析服务尚未接入。文件预览与试听可用。</span>
</template>
<article v-for="task in tasks" :key="task.id" class="task-row">
<strong>{{task.title}}</strong>
<span role="status">{{labels[task.status]}}</span>
<progress v-if="task.status==='running'&&task.progress!==undefined&&Number.isFinite(task.progress)&&task.progress>=0&&task.progress<=1" :value="task.progress" max="1" :aria-label="task.title+' 进度'"/>
<p v-if="task.error" class="error-text" role="alert">{{task.error}}</p>
<button v-if="task.canCancel&&['queued','running'].includes(task.status)" @click="emit('cancel',task.id)">取消此任务</button>
<button v-if="task.canRetry&&task.status==='failed'" @click="emit('retry',task.id)">重试此任务</button>
</article>
</section>
</template>
