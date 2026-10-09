<script setup lang="ts">
import {ref,computed,watch} from 'vue';
import type {Settings,Preview} from './port.ts';
import {clone,select,move,merge,undoMerge,resolve,label} from './state.ts';
const props=defineProps<{settings:Settings;preview:Preview;busy:boolean}>();
const emit=defineEmits<{save:[Settings];dirty:[boolean]}>();
const draft=ref(clone(props.settings)),error=ref('');
const picked=ref<{initial:string[];final:string[]}>({initial:[],final:[]});
const anchors:{initial?:string;final?:string}={},queries=ref({initial:'',final:''});
const dragKind=ref<'initial'|'final'>();
const target=ref<{kind:'initial'|'final';source:string;target:string}>();
const symbolKeys=['initial_order','final_order','initial_map','final_map'] as const;
const edited=computed(()=>symbolKeys.some(k=>JSON.stringify(draft.value[k])!==JSON.stringify(props.settings[k])));
watch([edited,()=>!!target.value],()=>emit('dirty',edited.value||!!target.value));
function changed(){emit('dirty',edited.value);}
function choose(kind:'initial'|'final',value:string,e:MouseEvent|KeyboardEvent){
 const out=select(draft.value[`${kind}_order`],picked.value[kind],anchors[kind],value,e.ctrlKey||e.metaKey,e.shiftKey);picked.value[kind]=out.selected;anchors[kind]=out.anchor;
}
function visible(kind:'initial'|'final'){const q=queries.value[kind].normalize('NFD');return draft.value[`${kind}_order`].filter(v=>label(v).normalize('NFD').includes(q));}
function prepare(kind:'initial'|'final'){
 if(picked.value[kind].length!==1){error.value='归并前请单选一个源音标。';return;}
 const source=picked.value[kind][0],to=draft.value[`${kind}_order`].find(v=>v!==source);
 if(to===undefined){error.value='至少保留一个不同的目标音标。';return;}
 target.value={kind,source,target:to};error.value='';
}
const affected=computed(()=>target.value?props.preview.analysis.rows.filter(r=>resolve(r[target.value!.kind],draft.value[`${target.value!.kind}_map`])===target.value!.source):[]);
function confirmMerge(){const value=target.value;if(!value)return;try{merge(draft.value,value.kind,value.source,value.target);target.value=undefined;picked.value[value.kind]=[];changed();}catch(e){error.value=(e as Error).message;}}
function drop(kind:'initial'|'final',value:string){draft.value[`${kind}_order`]=move(draft.value[`${kind}_order`],picked.value[kind],value);changed();}
function nudge(kind:'initial'|'final',delta:number){const order=draft.value[`${kind}_order`],selected=picked.value[kind];if(!selected.length)return;const next=[...order];const indices=order.map((v,i)=>selected.includes(v)?i:-1).filter(i=>i>=0);if(delta>0)indices.reverse();for(const i of indices){const j=i+delta;if(j>=0&&j<next.length&&!selected.includes(next[j]))[next[i],next[j]]=[next[j],next[i]];}draft.value[`${kind}_order`]=next;changed();}
function undo(kind:'initial'|'final',source:string){undoMerge(draft.value,kind,source,kind==='initial'?props.preview.config.initial_order:props.preview.config.final_order);changed();}
function restore(){draft.value=clone(props.settings);target.value=undefined;error.value='';picked.value={initial:[],final:[]};changed();}
function save(){if(target.value){error.value='请先确认或取消当前归并。';return;}emit('save',clone(draft.value));emit('dirty',false);}
defineExpose({value:()=>clone(draft.value),pending:()=>!!target.value,restore});
</script>
<template>
<section class="symbol-editor" aria-label="声韵页内编辑">
 <div class="editor-heading"><h2>声韵排序与归并</h2><span>{{edited?'有未保存修改':'设置已保存'}}</span></div>
 <p class="hint">Ctrl / ⌘ 离散多选，Shift 连续选择，拖动整组排序，也可使用上移和下移。</p>
 <p v-if="error" class="error" role="alert">{{error}}</p>
 <div class="symbol-columns"><div v-for="kind in (['initial','final'] as const)" :key="kind" class="symbol-panel">
  <div class="symbol-heading"><strong>{{kind==='initial'?'声母':'韵母'}}</strong><small>{{draft[`${kind}_order`].length}} 项 · 已选 {{picked[kind].length}}</small></div>
  <input v-model="queries[kind]" :aria-label="'搜索'+(kind==='initial'?'声母':'韵母')" :placeholder="'搜索'+(kind==='initial'?'声母':'韵母')"/>
  <ul class="symbol-list" role="listbox" aria-multiselectable="true" :aria-label="kind==='initial'?'声母列表':'韵母列表'"><li v-for="value in visible(kind)" :key="value" role="option" :aria-selected="picked[kind].includes(value)" :aria-disabled="busy" tabindex="0" :draggable="!busy" class="ipa-text" @click="!busy&&choose(kind,value,$event)" @keydown.enter.prevent="!busy&&choose(kind,value,$event)" @keydown.space.prevent="!busy&&choose(kind,value,$event)" @dragstart="dragKind=kind;!picked[kind].includes(value)&&(picked[kind]=[value])" @dragend="dragKind=undefined" @dragover.prevent @drop.prevent="!busy&&dragKind===kind&&drop(kind,value)">{{label(value)}}</li></ul>
  <div class="symbol-actions"><button :disabled="busy||!picked[kind].length" @click="nudge(kind,-1)">上移</button><button :disabled="busy||!picked[kind].length" @click="nudge(kind,1)">下移</button><button :disabled="busy" @click="prepare(kind)">{{kind==='initial'?'归并声母':'归并韵母'}}</button></div>
  <ul class="merge-list"><li v-for="(to,from) in draft[`${kind}_map`]" :key="from"><span class="ipa-text">{{label(String(from))}} → {{label(to)}}<template v-if="resolve(to,draft[`${kind}_map`])!==to"> → {{label(resolve(to,draft[`${kind}_map`]))}}</template></span><button :disabled="busy" :aria-label="'撤销归并 '+label(String(from))" @click="undo(kind,String(from))">撤销</button></li></ul>
 </div></div>
 <div v-if="target" class="merge-confirm" aria-label="归并确认"><div><strong>归并{{target.kind==='initial'?'声母':'韵母'}}</strong><span class="ipa-text">{{label(target.source)}} → </span><select v-model="target.target" aria-label="归并目标" :disabled="busy" class="ipa-text"><option v-for="v in draft[`${target.kind}_order`].filter(v=>v!==target!.source)" :key="v" :value="v">{{label(v)}}</option></select></div><p>影响 {{affected.length}} 条记录 · 例字 {{affected.slice(0,8).map(r=>r.character).join('、')}}。原始字音与备注保留。</p><div><button :disabled="busy" @click="target=undefined">取消本次归并</button><button class="primary" :disabled="busy" @click="confirmMerge">确认归并</button></div></div>
 <footer class="editor-footer"><button :disabled="busy||!edited" @click="restore">还原本次编辑</button><button class="primary" :disabled="busy" @click="save">保存声韵设置并继续</button></footer>
</section>
</template>
<style scoped>
.symbol-editor{border:1px solid var(--border);border-radius:var(--radius);background:var(--panel);padding:14px}.editor-heading,.symbol-heading{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}.editor-heading span{font-size:.9em;color:var(--muted)}h2{font-size:1.05em}.symbol-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--module-gap);margin-top:12px}.symbol-panel{padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--app);min-width:0}.symbol-panel>input{width:100%;margin-bottom:10px}.symbol-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(65px,1fr));gap:6px;align-content:start;list-style:none;margin:0;padding:0;min-height:200px;max-height:48vh;overflow:auto}.symbol-list li{border:1px solid var(--border);border-radius:6px;min-height:40px;padding:7px 6px;text-align:center;cursor:grab;background:var(--panel);overflow-wrap:anywhere}.symbol-list li[aria-selected=true]{background:var(--selected);border-color:var(--accent);color:var(--accent)}.symbol-actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.merge-list{list-style:none;padding:0;margin:12px 0 0}.merge-list li{display:flex;align-items:center;justify-content:space-between;border-top:1px solid var(--border);padding:6px 0;gap:8px}.merge-list button{font-size:.85em;min-height:26px}.merge-confirm{margin-top:12px;padding:12px;border:1px solid var(--accent);border-radius:var(--radius);background:var(--selected)}.merge-confirm>div{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.merge-confirm p{margin:8px 0;font-size:.9em}.editor-footer{display:flex;justify-content:space-between;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid var(--border)}.error{color:var(--danger)}
@media(max-width:820px){.symbol-columns{grid-template-columns:1fr}.symbol-list{min-height:120px;max-height:260px}}
</style>
