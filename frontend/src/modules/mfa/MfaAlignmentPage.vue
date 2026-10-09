<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import {computed,onMounted,onUnmounted,ref,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import type {ResearchContext,JobView} from '../../platform/research.ts';
import {projects} from '../../platform/browser.ts';
import {m11Message} from '../../platform/m11.ts';
import {defaults,beamChanged,validate,type TranscriptSource} from './state.ts';
import type {Catalog,CorpusItem,Grant} from './port.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active?:boolean}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const port=computed(()=>props.context.files.m11);
const key='m11:'+props.stateKey;
const saved=projects.read<{beam:number;retry_beam:number;runtime:string;model:string;transcriptSource?:TranscriptSource}>(key,{...defaults(),runtime:'',model:''});
const config=ref({beam:saved.beam,retry_beam:saved.retry_beam}),runtime=ref(saved.runtime),model=ref(saved.model);
const transcriptSource=ref<TranscriptSource>(saved.transcriptSource??'auto'),loadedSource=ref<TranscriptSource>();
const catalog=ref<Catalog>(),corpus=ref<CorpusItem[]>([]);
const importedDictionaries=ref<{id:string;name:string;asset:{asset_id:string;sha256:string};grant?:Grant}[]>([]),customDictionaryId=ref('');
const dictionary=computed(()=>importedDictionaries.value.find(d=>d.id===customDictionaryId.value)?.asset);
const output=ref<Grant>(),grants=ref<Record<string,Grant>>({}),manifestHash=ref('');
const error=ref(''),message=ref(''),busy=ref(false),dirty=ref(false),help=ref(false),componentOpen=ref(false);
const fileInput=ref<HTMLInputElement>(),dictInput=ref<HTMLInputElement>();
const log=ref({text:'',truncated:false});
const jobs=ref<JobView[]>([]),selected=ref<JobView>(),events=ref<{sequence:number;code:string;created_at:number}[]>([]);
const running=computed(()=>!!selected.value&&['queued','running','cancel_requested'].includes(selected.value.state));
const runtimeInfo=computed(()=>catalog.value?.runtimes.find(r=>r.id===runtime.value));
const modelInfo=computed(()=>catalog.value?.models.find(m=>m.id===model.value));
const models=computed(()=>catalog.value?.models.filter(m=>m.validated_runtime===runtime.value)??[]);
const acousticModels=computed(()=>models.value.filter((m,i,all)=>all.findIndex(other=>other.model_sha256===m.model_sha256)===i));
const acousticSelection=computed(()=>grants.value.model?'pending-model':acousticModels.value.find(m=>m.model_sha256===modelInfo.value?.model_sha256)?.id??'');
const registeredDictionaries=computed(()=>models.value.filter(m=>m.model_sha256===modelInfo.value?.model_sha256));
const dictionarySelection=computed(()=>customDictionaryId.value||model.value);
const pendingResources=computed(()=>Object.keys(grants.value).length>0);
const canCheck=computed(()=>!!(port.value?.component&&!grants.value.archive&&!grants.value.manifest&&(grants.value.runtime||runtimeInfo.value?.validated)&&(grants.value.model||modelInfo.value)&&(grants.value.dictionary||(!customDictionaryId.value&&modelInfo.value))));
const canImport=computed(()=>!!(grants.value.archive&&grants.value.manifest&&/^[a-f0-9]{64}$/.test(manifestHash.value)&&(grants.value.model||modelInfo.value)&&(grants.value.dictionary||(!customDictionaryId.value&&modelInfo.value))));
const selectedDictionaryName=computed(()=>importedDictionaries.value.find(d=>d.id===customDictionaryId.value)?.name||modelInfo.value?.dictionary_name||'模型登记词典');
const sourceChanged=computed(()=>!!corpus.value.length&&loadedSource.value!==transcriptSource.value);
const resultFiles=computed(()=>selected.value?.result_manifest&&'files'in selected.value.result_manifest?selected.value.result_manifest.files:[]);
let timer:ReturnType<typeof setInterval>|undefined,alive=true,refreshing=false;
const controller=new AbortController();
watch([config,runtime,model,transcriptSource,customDictionaryId,grants],()=>{dirty.value=true;},{deep:true});
watch(()=>dirty.value||busy.value||running.value,value=>emit('dirty',value),{immediate:true});
function save(){if(busy.value){error.value='请等待当前组件或文件操作完成。';return false;}try{const values=validate(config.value);if(!projects.write(key,{...values,runtime:runtime.value,model:model.value,transcriptSource:transcriptSource.value}))throw Error('草稿保存失败。');dirty.value=false;message.value='参数草稿已保存；任务继续在后台运行，可从任务记录恢复。';emit('dirty',running.value);return true;}catch(e){error.value=(e as Error).message;return false;}}
defineExpose({save});
async function operation(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;error.value='';message.value='';try{await fn();}catch(e){if(alive)error.value=m11Message(e instanceof Error?e.message:'操作失败。');}finally{if(alive)busy.value=false;}}
async function refresh(){
 if(!port.value||refreshing)return;refreshing=true;const selectedId=selected.value?.id;
 try{
  const [c,j]=await Promise.all([port.value.catalog(),port.value.history()]);if(!alive)return;
  catalog.value=c;jobs.value=j;
  if(!runtime.value&&c.runtimes.length)runtime.value=c.runtimes[0].id;
  if(!model.value&&models.value.length)model.value=models.value[0].id;
  if(selectedId&&selected.value?.id===selectedId){
   const updated=j.find(v=>v.id===selectedId);if(updated)selected.value=updated;
   const [eventList,nativeLog]=await Promise.all([port.value.events(selectedId),port.value.log?.(selectedId)]);
   if(alive&&selected.value?.id===selectedId){events.value=eventList.events;if(nativeLog)log.value=nativeLog;}
  }
 }catch(e){if(alive&&selected.value?.id===selectedId)error.value=m11Message((e as Error).message);}finally{refreshing=false;}
}
watch(runtime,()=>{if(catalog.value&&!models.value.some(m=>m.id===model.value))model.value=models.value[0]?.id??'';});

onMounted(()=>{void refresh();timer=setInterval(()=>{if(!busy.value&&props.active!==false)void refresh();},2000);});
onUnmounted(()=>{alive=false;controller.abort();if(timer)clearInterval(timer);});
function beamInput(event:Event){try{config.value=beamChanged(config.value,Number((event.target as HTMLInputElement).value));error.value='';}catch(e){error.value=(e as Error).message;}}
async function imported(event:Event){const input=event.target as HTMLInputElement,files=Array.from(input.files??[]);input.value='';if(!files.length||!port.value)return;await operation(async()=>{corpus.value=await port.value!.import(files,controller.signal,transcriptSource.value);loadedSource.value=transcriptSource.value;dirty.value=true;message.value=`已导入 ${corpus.value.length} 组音频与已有转写。`;});}
function chooseAcoustic(event:Event){const select=event.target as HTMLSelectElement,id=select.value;if(id==='import'){select.value=acousticSelection.value;void pick('model');}else if(id!=='pending-model'){model.value=id;customDictionaryId.value='';delete grants.value.model;delete grants.value.dictionary;}}
function chooseDictionary(event:Event){const select=event.target as HTMLSelectElement,id=select.value;if(id==='import'){select.value=dictionarySelection.value;if(port.value?.local&&port.value.pick&&port.value.dictionaryFromGrant)void importNativeDictionary();else dictInput.value?.click();}else{const imported=importedDictionaries.value.find(d=>d.id===id);if(imported){customDictionaryId.value=id;if(imported.grant)grants.value.dictionary=imported.grant;}else{customDictionaryId.value='';model.value=id;delete grants.value.dictionary;}}}
async function importNativeDictionary(){await operation(async()=>{const grant=await port.value!.pick!('dictionary');if(!grant)return;const asset=await port.value!.dictionaryFromGrant!(grant),id='custom-'+asset.asset_id;importedDictionaries.value.push({id,name:grant.label,asset,grant});customDictionaryId.value=id;grants.value.dictionary=grant;message.value='词典已选择，请检查并应用所选资源。';});}
async function importDictionary(event:Event){const input=event.target as HTMLInputElement,file=input.files?.[0];input.value='';if(!file||!port.value)return;await operation(async()=>{const asset=await port.value!.dictionary(file,controller.signal),id='custom-'+asset.asset_id;importedDictionaries.value.push({id,name:file.name,asset});customDictionaryId.value=id;dirty.value=true;});}
async function chooseCorpus(){await operation(async()=>{const grant=await port.value!.pick!('corpus');if(!grant)return;corpus.value=await port.value!.corpus!(grant,transcriptSource.value);loadedSource.value=transcriptSource.value;dirty.value=true;message.value=`已读取 ${corpus.value.length} 组音频与转写。`;});}
async function pick(purpose:string){await operation(async()=>{const value=await port.value!.pick!(purpose);if(value){grants.value[purpose]=value;message.value=['archive','manifest'].includes(purpose)?'离线组件已选择，请完成清单校验后在组件管理中导入。':'资源已选择，请在上方检查并应用所选资源。';}});}
function discardResources(){grants.value={};manifestHash.value='';customDictionaryId.value='';error.value='';message.value='已取消待检查选择，继续使用当前已应用的模型与词典。';}
async function check(action:'check'|'import'){await operation(async()=>{const values=Object.fromEntries(Object.entries(grants.value).map(([k,v])=>[k,v.id]));const result=await port.value!.component!({...values,action,runtime_id:runtime.value,model_id:model.value,trusted_manifest_sha256:manifestHash.value}) as {runtime_id?:string;model_id?:string};const next=await port.value!.catalog();const applied=next.models.find(m=>m.id===result.model_id&&m.validated_runtime===result.runtime_id);if(!applied||!next.runtimes.some(r=>r.id===result.runtime_id&&r.validated))throw Error('m11_component_not_registered');catalog.value=next;runtime.value=result.runtime_id!;model.value=result.model_id!;customDictionaryId.value='';grants.value={};manifestHash.value='';message.value=`已检查并应用：${applied.name} · ${applied.dictionary_name||'模型登记词典'}。`;const stored=projects.read(key,{...defaults(),runtime:'',model:''});if(!projects.write(key,{...stored,runtime:runtime.value,model:model.value}))message.value+=' 本机选择记忆保存失败，下次打开需重新选择。';});}
async function start(){await operation(async()=>{if(pendingResources.value)throw Error('新资源尚未应用，请先检查并应用，或取消待检查选择。');if(sourceChanged.value)throw Error('转写来源已更改，请重新读取目录或导入文件。');const settings=validate(config.value);selected.value=await port.value!.create({runtime_id:runtime.value,model_id:model.value,corpus:corpus.value,dictionary:dictionary.value??null,config:settings});events.value=[];await refresh();});}
async function cancel(){if(selected.value)await operation(async()=>{await port.value!.cancel(selected.value!.id);await refresh();});}
async function show(job:JobView){selected.value=job;events.value=[];log.value={text:'',truncated:false};await refresh();}
async function saveResults(){await operation(async()=>{if(!output.value)output.value=(await port.value!.chooseOutput!())??undefined;if(!output.value)return;const result=await port.value!.save!(selected.value!,output.value);message.value=`已保存 ${result.count} 个文件到 ${output.value.label}/${result.directory}。`;});}
</script>

<template>
<ModuleFrame unified fit class="mfa-page" label="MFA 自动标注工作区" :aria-busy="busy">
 <template #toolbar><ModuleToolbar>
  <button class="primary" v-if="port?.local" :disabled="busy" @click="chooseCorpus"><AppIcon name="folder"/>打开音频目录</button>
  <button class="primary" :disabled="busy||!port" @click="fileInput?.click()">导入音频与转写</button>
  <input ref="fileInput" hidden type="file" multiple accept=".wav,.lab,.txt,.TextGrid" @change="imported"/>

  <template #actions><button :disabled="busy" @click="save">保存参数草稿</button><button @click="help=true">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template>
 </ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-else-if="busy" kind="loading" message="正在处理。组件检查会执行真实 MFA 小任务，请稍候…"/><ModuleStatus v-if="!port" kind="empty" message="请从本地研究工作台或已登录网页项目打开 MFA。"/></template>
 <ModuleWorkbench unified :state-key="stateKey" left-label="组件与模型" right-label="任务与记录">
 <template #left>   <ModuleSection label="运行组件与模型" title="运行组件与模型">
    <div class="mfa-form"><label>运行环境<select v-model="runtime" :disabled="busy||running"><option value="">选择已检查组件</option><option v-for="r in catalog?.runtimes" :key="r.id" :value="r.id">MFA {{r.version}} · {{r.platform}} / {{r.arch}}</option></select></label>
     <label>声学模型<select :value="acousticSelection" :disabled="busy||running" @change="chooseAcoustic"><option value="">选择声学模型</option><option v-for="m in acousticModels" :key="m.id" :value="m.id">{{m.name}}</option><option v-if="grants.model" value="pending-model">{{grants.model.label}} · 待检查</option><option v-if="port?.local&&port.pick" value="import">导入其他声学模型…</option></select></label>
     <label>发音词典<select :value="dictionarySelection" :disabled="busy||!port||running" @change="chooseDictionary"><option v-if="!model" value="">选择发音词典</option><option v-for="m in registeredDictionaries" :key="m.id" :value="m.id">{{m.dictionary_name||'模型登记词典'}}</option><option v-for="d in importedDictionaries" :key="d.id" :value="d.id">{{d.name}} · {{grants.dictionary?.id===d.grant?.id&&d.grant?'待检查':'已导入'}}</option><option value="import">导入其他词典…</option></select></label>
     <input ref="dictInput" hidden type="file" accept=".dict,.txt" @change="importDictionary"/>
    </div>
    <p v-if="modelInfo" class="muted applied-resources">当前已应用：{{modelInfo.name}} · {{pendingResources?modelInfo.dictionary_name||'模型登记词典':selectedDictionaryName}}</p>
    <ModuleStatus v-if="pendingResources" kind="info" :message="grants.archive||grants.manifest?'离线组件尚未应用，请在组件管理中完成校验与导入。当前暂停提交对齐。':'新资源尚未应用。检查通过后切换模型与词典，当前暂停提交对齐。'"/>
    <div v-if="pendingResources" class="actions resource-actions"><button class="primary" :disabled="busy||running||!canCheck" @click="check('check')">检查并应用所选资源</button><button :disabled="busy||running" @click="discardResources">取消待检查选择</button></div>
    <details v-if="runtimeInfo" class="component-details"><summary>组件、资源与校验信息</summary><p>来源：{{runtimeInfo.source||'登记清单'}}</p><p>下载量：{{runtimeInfo.download_bytes?.toLocaleString()??'未测量 / 已有环境'}} B · 安装文件：{{runtimeInfo.installed_bytes?.toLocaleString()??'待测'}} B</p><p>模型：{{modelInfo?.model_bytes?.toLocaleString()??'待测'}} B · 登记词典：{{modelInfo?.dictionary_bytes?.toLocaleString()??'待测'}} B</p><p>任务临时空间监测预算 512,000,000 B，失败诊断保留后仍占本机空间。</p><dl><template v-for="(v,k) in runtimeInfo.versions" :key="k"><dt>{{k}}</dt><dd>{{v}}</dd></template></dl><p class="digest">运行时内容摘要：{{runtimeInfo.fingerprint}}</p><p class="digest">已应用声学模型 SHA-256：{{modelInfo?.model_sha256}}</p><p class="digest">已应用词典 SHA-256：{{pendingResources?modelInfo?.dictionary_sha256:dictionary?.sha256??modelInfo?.dictionary_sha256}}</p><p v-if="runtimeInfo.archive_sha256" class="digest">组件包 SHA-256：{{runtimeInfo.archive_sha256}}</p></details>
    <ModuleStatus v-if="catalog?.waiting_reason" kind="info" :message="m11Message(catalog.waiting_reason)"/>
    <button v-if="port?.local" class="component-toggle" :aria-expanded="componentOpen" @click="componentOpen=!componentOpen">{{componentOpen?'收起组件管理':'组件安装与环境检查'}}</button>
    <div v-if="componentOpen&&port?.local" class="component-manager">
     <ModuleStatus kind="info" :message="m11Message('m11_download_not_published')"/>
     <button disabled title="等待固定版本与可信摘要发布">下载并安装组件 · 待发布</button>
     <p class="muted">声学模型和词典在上方下拉框直接导入。未另选环境时，复用当前已检查的运行环境。</p>
     <div v-for="p in [{key:'runtime',label:'已有环境 / auto_alignment'},{key:'archive',label:'离线组件 ZIP'},{key:'manifest',label:'可信发布清单 JSON'}]" :key="p.key" class="resource-line"><span>{{p.label}}：{{grants[p.key]?.label||(p.key==='runtime'&&runtimeInfo?'使用当前已检查环境':'未选择')}}</span><button :disabled="busy||running" @click="pick(p.key)">选择</button></div>
     <label>独立可信来源提供的清单 SHA-256<input v-model="manifestHash" class="digest" maxlength="64" placeholder="从组件发布者的可信渠道核对" :disabled="busy"/></label>
     <div class="actions"><button :disabled="busy||running||!canImport" @click="check('import')">校验并导入离线组件</button></div>
     <p class="muted">检查包含版本、原生依赖与公开合成音频实际对齐。安装失败保留当前版本。组件、模型与主程序分别管理。</p>
    </div>
   </ModuleSection>
<ModuleSection label="对齐参数" title="对齐参数">
    <div class="parameter-row"><label>Beam<input type="number" min="1" max="10000" :value="config.beam" :disabled="busy" @change="beamInput"/></label><label>Retry beam<input v-model.number="config.retry_beam" type="number" min="1" max="40000" :disabled="busy"/></label></div>
    <p class="muted">本机使用 CPU。当前 MFA 3.3.8 的 GMM-HMM 对齐流程不支持 GPU。</p>

   </ModuleSection>
</template>
   <ModuleSection label="音频与既有转写" title="音频与既有转写">
    <label>转写来源<select v-model="transcriptSource" :disabled="busy"><option value="auto">自动 · 唯一同名转写</option><option value=".lab">LAB</option><option value=".txt">TXT</option><option value=".TextGrid">TextGrid</option></select></label>
    <p class="muted">选择语料目录会递归读取 WAV 与同名转写；导入音频与转写只读取手选文件。多种格式可共存，选择来源后只读取该格式。</p>
    <ModuleStatus v-if="sourceChanged" kind="info" message="转写来源已更改，请重新读取目录或导入文件。"/>
    <ModuleStatus v-if="!corpus.length" kind="empty" message="导入同名 WAV 与所选转写。已有 LAB 与 TextGrid 可同时保留。"/>
    <ul v-else class="corpus-list"><li v-for="c in corpus" :key="c.name"><span>{{c.name}}</span><span class="muted">{{c.transcript_format}}</span></li></ul>
    <p class="muted">已选择 {{corpus.length}} 组。当前每份音频接收上限 120 秒，任务最多 100 份且输入合计最多 64 MB。</p>
    <div v-if="port?.chooseOutput" class="resource-line"><span>输出目录：{{output?.label||'保存结果时选择'}}</span><button :disabled="busy" @click="operation(async()=>{output=(await port!.chooseOutput!())??undefined;})">选择输出目录</button></div>
   </ModuleSection>

 <template #right><ModuleSection label="对齐任务" title="运行对齐"><div class="actions"><button class="primary" :disabled="busy||!port||!corpus.length||sourceChanged||pendingResources||!runtime||!model||running" @click="start">开始对齐</button><button :disabled="busy||!running||selected?.state==='cancel_requested'" @click="cancel">取消任务</button></div></ModuleSection>
 <ModuleStatus v-if="message" kind="info" :message="message"/>
 <button :disabled="busy||!port" @click="refresh">刷新任务</button>

   <ModuleSection label="任务与日志" title="任务与日志">
    <label>任务记录<select :value="selected?.id??''" @change="show(jobs.find(j=>j.id===($event.target as HTMLSelectElement).value)!)"><option value="" disabled>选择任务</option><option v-for="j in jobs" :key="j.id" :value="j.id">{{new Date(j.created_at*1000).toLocaleString()}} · {{m11Message(j.state)}}</option></select></label>
    <ModuleStatus v-if="!selected" kind="empty" message="还未选择任务。运行后会显示实际阶段与结果。"/>
    <template v-else>
     <div class="job-state"><strong>{{m11Message(selected.state)}}</strong><span>{{Math.round(selected.progress*100)}}%</span></div><progress :value="selected.progress" max="1" aria-label="任务阶段进度"/>
     <ModuleStatus v-if="selected.waiting_reason&&selected.state==='queued'" kind="info" :message="m11Message(selected.waiting_reason)"/>
     <ModuleStatus v-if="selected.error_code" kind="error" :message="m11Message(selected.error_code)"/>
     <ol class="event-log" aria-label="脱敏任务日志"><li v-for="e in events" :key="e.sequence"><time>{{new Date(e.created_at*1000).toLocaleTimeString()}}</time><span>{{m11Message(e.code)}}</span></li></ol>
     <details v-if="log.text"><summary>查看脱敏 MFA 日志</summary><p v-if="log.truncated" class="muted">显示最后 256 KiB；完整原始日志保留在本机任务诊断目录。</p><pre class="native-log">{{log.text}}</pre></details>
     <p v-if="running" class="muted">进度表示任务阶段，不估算剩余时间。关闭页面后可从任务记录继续查看。</p>
    </template>
   </ModuleSection>
   <ModuleSection label="TextGrid 与溯源" title="TextGrid 与溯源">
    <ModuleStatus v-if="!resultFiles.length" kind="empty" message="完整结果发布后，可下载 TextGrid 和参数、输入、模型及运行时溯源。"/>
    <div v-for="f in resultFiles" :key="f.id" class="result-row"><span>{{f.name}}</span><span class="muted">{{f.size_bytes.toLocaleString()}} B</span><button :disabled="busy" @click="operation(()=>port!.download(selected!,f.id))">下载</button></div>
    <button class="primary" v-if="resultFiles.length&&port?.save" :disabled="busy" @click="saveResults">保存完整结果到输出目录</button>
    <p class="muted">自动边界需人工校对。保存使用独立结果子目录，保留原音频与原始 TextGrid。</p>
   </ModuleSection>
 </template>
 </ModuleWorkbench>
 <ModalDialog v-if="help" title="MFA 自动标注帮助" @close="help=false"><p>MFA 将音频与已有文本强制对齐。目录入口会递归读取，文件入口只读取手选文件。LAB、TXT 与 TextGrid 可共存，通过转写来源选择本次读取的格式。自动模式只接受唯一同名转写。</p><p>TextGrid 优先读取独立转写区间层。若仅有一个 words 层，则按顺序提取词标签为整段转写重新对齐，原边界不作为约束；phones 层不作为转写。原文件保留。声学模型与词典音素集必须匹配，汉字词典配汉字，拼音词典配其规定的拼音与声调格式。桌面可从上方模型与词典下拉框直接导入，检查通过后应用并记住组合。新资源待检查时暂停提交，取消选择恢复已应用组合。自检支持词典中的 a、啊 和 a1。网页使用管理员登记模型，导入词典只影响本次任务。未登录词会在特征提取前报错。</p><p>新草稿 Beam 默认 100、Retry beam 默认 400，已保存参数保留。提高 Beam 可放宽搜索，也可能增加耗时。调整 Beam 时将 Retry beam 至少提高到四倍；提交时 Retry beam 不大于 Beam 则恢复四倍。</p><p>本机使用 CPU。当前 MFA 3.3.8 的 GMM-HMM 对齐流程不支持 GPU。主程序不附带完整 MFA，已有 Windows 3.3.8 环境可通过实际检查后使用，装好组件与模型后本地对齐无需公网。</p><p>日志仅显示脱敏阶段。取消、超时或失败后，本机独立任务目录保留部分输出和诊断，不作为完整结果。网页数据遵循账号政策，未验证节点与服务器能力保持等待。</p></ModalDialog>
</ModuleFrame>
</template>

<style scoped>
.component-details{margin-top:10px;font-size:var(--support-size);overflow-wrap:anywhere}.component-details dl{display:grid;grid-template-columns:1fr 1fr;gap:4px}.component-details dd{margin:0}.native-log{white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;font-family:var(--font-mono);font-size:var(--support-size)}.mfa-columns{display:grid;grid-template-columns:var(--panel-left,400px) minmax(300px,1fr);gap:var(--module-gap);align-items:start}.mfa-controls,.mfa-results{display:grid;gap:var(--module-gap);min-width:0}.mfa-form,.component-manager{display:grid;gap:var(--control-gap)}label{display:grid;gap:6px;font-size:var(--control-size)}select,input{width:100%;min-width:0}.resource-line,.actions,.parameter-row,.result-row,.job-state{display:flex;align-items:center;gap:var(--control-gap);flex-wrap:wrap}.resource-line{justify-content:space-between}.resource-line span,.result-row>span:first-child{overflow-wrap:anywhere;flex:1;min-width:140px}.parameter-row label{flex:1;min-width:100px}.actions{margin-top:12px}.component-toggle{margin-top:12px}.component-manager{margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}.digest,.event-log{font-family:var(--font-mono)}.muted{color:var(--muted);font-size:var(--support-size);line-height:1.6}.corpus-list{list-style:none;padding:0;max-height:240px;overflow:auto}.corpus-list li{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.job-state{justify-content:space-between;margin-top:12px}progress{width:100%;height:8px;accent-color:var(--accent)}.event-log{padding:0;list-style:none;min-height:100px;max-height:300px;overflow:auto;background:var(--bg);border:1px solid var(--border);border-radius:6px}.event-log li{padding:7px 10px;display:flex;gap:12px;font-size:var(--support-size)}.event-log time{color:var(--muted);white-space:nowrap}.result-row{padding:8px 0;border-bottom:1px solid var(--border)}@container module (max-width:800px){.mfa-columns{grid-template-columns:1fr}}

.mfa-columns{flex:1;min-height:0;align-items:stretch}
.mfa-controls,.mfa-results{min-height:0;overflow:auto;align-content:start;overscroll-behavior:contain}
@container module (max-width:800px){.mfa-columns{flex:none}.mfa-controls,.mfa-results{overflow:visible}}
</style>
