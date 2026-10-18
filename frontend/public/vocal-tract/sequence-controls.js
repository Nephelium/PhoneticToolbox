import {articulationParameters} from './articulation.mjs';
/* Shared sequence editor. All phonetic parsing and constraints run in core. */
export function setupSequenceControls({metadata,initial={},initiallyOpen=true,getCatalog,post,append,onChange,isBusy}){
  const $=id=>document.getElementById(id),root=$('sequenceControls');
  root.innerHTML='<details class="ipa-sequence" open><summary>从国际音标生成</summary><label for="sequenceIpa">国际音标</label><textarea id="sequenceIpa" rows="2" placeholder="例如 a i u，或连续输入已存音标" spellcheck="false"></textarea><div class="ipa-sequence-actions"><label>初始时长 <select id="ipaTiming"><option value="auto">按音标类型</option><option value="uniform">统一设置</option></select></label><label id="ipaDurationField" hidden><input aria-label="统一时长（秒）" id="ipaDuration" type="number" min="0.05" max="3" step="0.01" value="0.2"> 秒</label><button id="generateIpa" class="primary">添加到序列</button></div><label class="ipa-stop-option"><input id="ipaStops" type="checkbox" checked> 塞音使用闭塞与快速释放时序</label><p id="ipaSequenceStatus" class="minor" role="status">默认：塞音 80 ms、擦音 160–180 ms、元音 200 ms，逐项可改。普通非末项为过渡，末项为停留；塞音时序可分开编辑闭塞、释放与浊音起始。</p></details><details class="global-articulation"><summary>全局构音调整 <span id="globalCount"></span></summary><p class="minor">合成时作用于每个姿态，达到边界后限幅。原构型保留。</p><div class="global-add"><select id="globalParameter" aria-label="选择全局调整参数"></select><button id="addGlobalControl">添加调整</button></div><div id="globalRows"></div><button id="resetGlobalControls" class="quiet">恢复全部原值</button></details>';
  const parameters=articulationParameters(metadata);
  root.querySelector('.ipa-sequence').open=initiallyOpen;
  let controls=structuredClone(initial),working=false,locked=false;
  const updateLock=()=>root.querySelectorAll('input,select,textarea,button').forEach(e=>e.disabled=locked||working);
  for(const p of parameters){const o=document.createElement('option');o.value=p.name;o.textContent=p.label;$('globalParameter').append(o);}
  function render(){
    $('globalRows').replaceChildren();$('globalCount').textContent=Object.keys(controls).length?`· ${Object.keys(controls).length} 项`:'';
    for(const [key,value] of Object.entries(controls)){
      const p=parameters.find(p=>p.name===key);if(!p)continue;
      const row=document.createElement('div');row.className='global-row';
      const label=document.createElement('label');label.textContent=p.label;label.htmlFor='global-value-'+key;
      const mode=document.createElement('select');mode.setAttribute('aria-label',p.label+'调整方式');
      for(const [id,name] of [['fixed','固定为'],['offset','相对偏移']]){const o=document.createElement('option');o.value=id;o.textContent=name;mode.append(o);}mode.value=value.mode;
      const input=document.createElement('input');Object.assign(input,{id:'global-value-'+key,type:'number',step:.01,value:value.value});
      const limits=()=>{input.min=mode.value==='fixed'?p.min:p.min-p.max;input.max=mode.value==='fixed'?p.max:p.max-p.min;};limits();
      const unit=document.createElement('span');unit.textContent=p.unit;unit.title=p.hint||'';input.title=p.hint||'';
      const remove=document.createElement('button');remove.textContent='×';remove.setAttribute('aria-label','恢复'+p.label+'原值');
      mode.onchange=()=>{value.mode=mode.value;limits();value.value=mode.value==='offset'?0:Math.max(p.min,Math.min(p.max,value.value));render();onChange();};
      input.onchange=()=>{if(!input.value.trim()||!input.checkValidity()){input.reportValidity();input.value=value.value;return;}value.value=Number(input.value);onChange();};
      remove.onclick=()=>{delete controls[key];render();onChange();};
      row.append(label,mode,input,unit,remove);$('globalRows').append(row);
    }
    updateLock();
  }
  $('addGlobalControl').onclick=()=>{const p=parameters.find(p=>p.name===$('globalParameter').value);if(!controls[p.name]){controls[p.name]={mode:'offset',value:0};render();onChange();}};
  $('resetGlobalControls').onclick=()=>{controls={};render();onChange();};
  $('ipaTiming').onchange=()=>{$('ipaDurationField').hidden=$('ipaTiming').value==='auto';};
  $('generateIpa').onclick=async()=>{
    if(working||isBusy())return;
    const duration=$('ipaDuration');if($('ipaTiming').value==='uniform'&&(!duration.checkValidity()||!duration.value.trim())){duration.reportValidity();return;}
    working=true;updateLock();
    try{
      const result=await(await post('sequence/ipa',{text:$('sequenceIpa').value,stop_timing:$('ipaStops').checked,duration:$('ipaTiming').value==='auto'?null:Number(duration.value),presets:getCatalog()})).json();
      if(result.missing.length){$('ipaSequenceStatus').textContent='缺少已存构型：'+result.missing.join('、')+'。请先在当前模型下保存这些音标的构型。';return;}
      await append(result.frames);$('ipaSequenceStatus').textContent=`已添加 ${result.frames.length} 项，可逐项调整时长、构型与声音。`;
      root.querySelector('.ipa-sequence').open=false;
    }catch(e){$('ipaSequenceStatus').textContent=e.message;}
    finally{working=false;locked=isBusy();updateLock();}
  };
  render();
  return {get:()=>structuredClone(controls),set:value=>{controls=structuredClone(value??{});render();},
    lock:value=>{locked=value;updateLock();}};
}
