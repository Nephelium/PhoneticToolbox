import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
import {articulationParameters} from './articulation.mjs';
/* Shared sequence editor. All phonetic parsing and constraints run in core. */
export function setupSequenceControls({metadata,initial={},initiallyOpen=true,getCatalog,post,append,onChange,isBusy}){
  const $=id=>document.getElementById(id),root=$('sequenceControls');
  setHTML(root,()=>"<details class=\"ipa-sequence\" open><summary><!--m10-i18n:modules.vocal-tract-frame.text.generate_from_ipa_547-->从国际音标生成</summary><label for=\"sequenceIpa\"><!--m10-i18n:modules.vocal-tract-frame.text.international_phonetic_alphabet_548-->国际音标</label><textarea data-m10-i18n='{\"placeholder\":\"modules.vocal-tract-frame.text.e_g_a_i_u_or_concatenate_saved_symbols_549\"}' id=\"sequenceIpa\" rows=\"2\" placeholder=\"例如 a i u，或连续输入已存音标\" spellcheck=\"false\"></textarea><div class=\"ipa-sequence-actions\"><label><!--m10-i18n:modules.vocal-tract-frame.text.initial_duration_550-->初始时长 <select id=\"ipaTiming\"><option value=\"auto\"><!--m10-i18n:modules.vocal-tract-frame.text.by_symbol_type_551-->按音标类型</option><option value=\"uniform\"><!--m10-i18n:modules.vocal-tract-frame.text.uniform_duration_552-->统一设置</option></select></label><label id=\"ipaDurationField\" hidden><input data-m10-i18n='{\"aria-label\":\"modules.vocal-tract-frame.text.uniform_duration_seconds_553\"}' aria-label=\"统一时长（秒）\" id=\"ipaDuration\" type=\"number\" min=\"0.05\" max=\"3\" step=\"0.01\" value=\"0.2\"><!--m10-i18n:modules.vocal-tract-frame.text.s_144--> 秒</label><button id=\"generateIpa\" class=\"primary\"><!--m10-i18n:modules.vocal-tract-frame.text.add_to_sequence_554-->添加到序列</button></div><label class=\"ipa-stop-option\"><input id=\"ipaStops\" type=\"checkbox\" checked><!--m10-i18n:modules.vocal-tract-frame.text.use_closure_and_rapid_release_timing_for_stops_555--> 塞音使用闭塞与快速释放时序</label><p id=\"ipaSequenceStatus\" class=\"minor\" role=\"status\"><!--m10-i18n:modules.vocal-tract-frame.text.defaults_stops_80_ms_fricatives_160_180_ms_vowels_200_ms_editable_individual_556-->默认：塞音 80 ms、擦音 160–180 ms、元音 200 ms，逐项可改。普通非末项为过渡，末项为停留；塞音时序可分开编辑闭塞、释放与浊音起始。</p></details><details class=\"global-articulation\"><summary><!--m10-i18n:modules.vocal-tract-frame.text.global_articulatory_adjustments_557-->全局构音调整 <span id=\"globalCount\"></span></summary><p class=\"minor\"><!--m10-i18n:modules.vocal-tract-frame.text.applied_to_every_posture_during_synthesis_clipped_to_bounds_original_configu_558-->合成时作用于每个姿态，达到边界后限幅。原构型保留。</p><div class=\"global-add\"><select data-m10-i18n='{\"aria-label\":\"modules.vocal-tract-frame.text.select_global_parameter_559\"}' id=\"globalParameter\" aria-label=\"选择全局调整参数\"></select><button id=\"addGlobalControl\"><!--m10-i18n:modules.vocal-tract-frame.text.add_adjustment_560-->添加调整</button></div><div id=\"globalRows\"></div><button id=\"resetGlobalControls\" class=\"quiet\"><!--m10-i18n:modules.vocal-tract-frame.text.restore_all_original_values_561-->恢复全部原值</button></details>");
  const parameters=articulationParameters(metadata);
  root.querySelector('.ipa-sequence').open=initiallyOpen;
  let controls=structuredClone(initial),working=false,locked=false;
  const updateLock=()=>root.querySelectorAll('input,select,textarea,button').forEach(e=>e.disabled=locked||working);
  for(const p of parameters){const o=document.createElement('option');o.value=p.name;bindText(o,()=>p.label);$('globalParameter').append(o);}
  function render(){
    $('globalRows').replaceChildren();bindText($('globalCount'),()=>Object.keys(controls).length?msg("modules.vocal-tract-frame.text.adjustments_value_294",{v0:()=>Object.keys(controls).length}):'');
    for(const [key,value] of Object.entries(controls)){
      const p=parameters.find(p=>p.name===key);if(!p)continue;
      const row=document.createElement('div');row.className='global-row';
      const label=document.createElement('label');bindText(label,()=>p.label);label.htmlFor='global-value-'+key;
      const mode=document.createElement('select');bindAttribute(mode,"aria-label",()=>join(()=>p.label,()=>msg("modules.vocal-tract-frame.text.adjustment_mode_295")));
      for(const [id,name] of [['fixed',msg("modules.vocal-tract-frame.text.fixed_at_114")],['offset',msg("modules.vocal-tract-frame.text.relative_offset_296")]]){const o=document.createElement('option');o.value=id;bindText(o,()=>name);mode.append(o);}mode.value=value.mode;
      const input=document.createElement('input');Object.assign(input,{id:'global-value-'+key,type:'number',step:.01,value:value.value});
      const limits=()=>{input.min=mode.value==='fixed'?p.min:p.min-p.max;input.max=mode.value==='fixed'?p.max:p.max-p.min;};limits();
      const unit=document.createElement('span');bindText(unit,()=>p.unit);bindAttribute(unit,'title',()=>p.hint||'');bindAttribute(input,'title',()=>p.hint||'');
      const remove=document.createElement('button');bindText(remove,()=>'×');bindAttribute(remove,"aria-label",()=>join(()=>join(()=>msg("modules.vocal-tract-frame.text.restore_297"),()=>p.label),()=>msg("modules.vocal-tract-frame.text.to_original_value_298")));
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
      if(result.missing.length){bindText($('ipaSequenceStatus'),()=>join(()=>join(()=>msg("modules.vocal-tract-frame.text.missing_saved_configurations_299"),()=>result.missing.join('、')),()=>msg("modules.vocal-tract-frame.text.save_configurations_for_these_symbols_under_the_current_model_first_300")));return;}
      await append(result.frames);bindText($('ipaSequenceStatus'),()=>msg("modules.vocal-tract-frame.text.added_value_adjust_individual_durations_configurations_and_sound_301",{v0:()=>result.frames.length}));
      root.querySelector('.ipa-sequence').open=false;
    }catch(e){bindText($('ipaSequenceStatus'),()=>e.message);}
    finally{working=false;locked=isBusy();updateLock();}
  };
  render();
  return {get:()=>structuredClone(controls),set:value=>{controls=structuredClone(value??{});render();},
    lock:value=>{locked=value;updateLock();}};
}
