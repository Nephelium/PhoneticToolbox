<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import {nextTick,ref} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import sources from '../../generated/sources.json';
import {copyText} from '../../platform/clipboard.ts';
const emit=defineEmits<{references:[]}>();
const upstream=sources.find(source=>source.id==='SRC-ZAIWA')!;
const paper=sources.find(source=>source.id==='REF-ZAIWA')!;
const message=ref('');
const details=ref(false);
async function references(){details.value=false;await nextTick();emit('references');}
async function copy(){
 try{await copyText(paper.citation!);message.value='引用已复制';}
 catch{message.value='无法访问剪贴板，请选中论文引用复制。';}
}
</script>
<template>
 <footer class="m07-attribution" aria-label="发声类型合成来源与致谢">
  <p class="attribution-line">{{upstream.attribution}}<span class="permission-date">（{{upstream.permission_date}} 作者邮件许可）</span></p>
  <div class="citation-line">
   <p class="paper-citation">{{paper.citation}}</p>
   <div class="attribution-actions">
    <a :href="paper.urls.doi" target="_blank" rel="noopener noreferrer">论文</a>
    <a :href="upstream.urls.repository" target="_blank" rel="noopener noreferrer">原始仓库</a>
    <button type="button" @click="copy">复制引用</button>
    <button type="button" @click="details=true">改写说明</button>
    <span v-if="message" role="status">{{message}}</span>
   </div>
  </div>
 </footer>
 <ModalDialog v-if="details" title="发声类型合成改写说明" @close="details=false">
  <p>{{upstream.adaptation_note}}</p>
  <p>{{upstream.license}}</p>
  <p>{{upstream.adaptation_caution}} <a :href="upstream.urls.repository" target="_blank" rel="noopener noreferrer">尝试作者团队的官方开源代码</a></p>
  <p>本模块用于制作 LPC 残差操纵的实验刺激。上述适配不表示精确复现原论文全部流程，也未验证发声类别知觉效度或连续统的知觉等距。原论文、配套录音与统计数据的许可分别处理，Parselmouth、REAPER 等组件保留各自来源与许可。</p>
  <p class="paper-citation">{{paper.citation}}</p>
  <button type="button" @click="references"><AppIcon name="book"/>方法与引用</button>
 </ModalDialog>
</template>
<style scoped>
.m07-attribution{flex:none;min-width:0;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel);font-size:var(--support-size);line-height:1.45;color:var(--text)}
.m07-attribution p{margin:0;overflow-wrap:anywhere}.permission-date{color:var(--muted);white-space:nowrap}.citation-line{display:flex;align-items:center;flex-wrap:wrap;gap:2px 14px;margin-top:3px}.paper-citation{flex:1 1 700px}.attribution-actions{display:flex;align-items:center;flex-wrap:wrap;gap:4px 10px}.attribution-actions a{color:var(--accent);text-decoration:underline;text-underline-offset:2px}.attribution-actions button{min-height:24px;padding:2px 7px;font-size:inherit}.attribution-actions [role=status]{color:var(--muted)}
</style>
