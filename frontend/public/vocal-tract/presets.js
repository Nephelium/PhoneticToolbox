import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
import {setupPresetPack} from './preset-pack.js';
import {refreshBindings} from './i18n.mjs';

export function builtinPose(metadata,name){
  const pose={params:metadata.presets[name],lip_width:1,f0:metadata.default_f0??120,
    manual_root:false,larynx_height:0,tongue_shape:[0,0,0,0],lip_shape:[0,0,0,0],
    source:metadata.source_presets.voiced,...metadata.consonants?.[name],...metadata.default_poses?.[name],preset:name};
  return structuredClone(pose);
}

export function presetSymbol(name,symbols){
  const key=(name||'').trim().replace(/^[/\[]|[/\]]$/g,'');
  return symbols.has(key)?key:null;
}

export function presetInputName(value,symbols){
  const name=typeof value==='string'?value.trim():'';
  if(!name)throw uiError(msg("modules.vocal-tract-frame.text.enter_a_phonetic_symbol_218"));
  if([...name].length>80||/[\u0000-\u001f]/u.test(name))throw uiError(msg("modules.vocal-tract-frame.text.symbol_names_must_be_a_single_line_of_at_most_80_characters_219"));
  return presetSymbol(name,symbols)||name;
}

export function setupPresets({initial,initialLibrary,builtins=[],favorites:initialFavorites=[],chooseBuiltin,chart,getPose,apply,post,isReady,onSaved,onDeleted,notify}){
  const $=id=>document.getElementById(id),dialog=$('presetDialog');
  const entries=new Map(chart.entries.map(entry=>[entry.id,entry]));
  const symbols=new Set(chart.entries.map(entry=>entry.insertText));
  let presets=initial,favorites=[...initialFavorites],working=false,mode='load',selected='',common=false;
  let library=initialLibrary,pendingDelete='';
  const status=message=>bindText($('presetStatus'),()=>message);
  const key=name=>(presetSymbol(name,symbols)||(name||'').trim()).normalize('NFC');
  const customStored=symbol=>presets.findLast(p=>key(p.name)===symbol);
  const stored=symbol=>customStored(symbol)||builtins.find(p=>key(p.name)===symbol);
  const commonPanel=document.createElement('div');commonPanel.id='presetCommon';commonPanel.hidden=true;
  $('presetChart').before(commonPanel);
  const sets=document.createElement('div');sets.className='preset-libraries';
  setHTML(sets,()=>"<div class=\"preset-library-toolbar\">\n    <label><!--m10-i18n:modules.vocal-tract-frame.text.configuration_set_537-->构型套装<select data-m10-i18n='{\"aria-label\":\"modules.vocal-tract-frame.text.configuration_set_537\"}' id=\"presetSet\" aria-label=\"构型套装\"></select></label>\n    <label><!--m10-i18n:modules.vocal-tract-frame.text.set_name_538-->套装名称<input data-m10-i18n='{\"placeholder\":\"modules.vocal-tract-frame.text.enter_a_set_name_539\"}' id=\"presetSetName\" maxlength=\"120\" placeholder=\"输入套装名称\"></label>\n    <div class=\"preset-set-actions\"><button id=\"presetSetCreate\" type=\"button\"><!--m10-i18n:modules.vocal-tract-frame.text.new_empty_set_540-->新建空套装</button><button id=\"presetSetCopy\" type=\"button\"><!--m10-i18n:modules.vocal-tract-frame.text.copy_current_set_541-->复制当前套装</button><button id=\"presetSetRename\" type=\"button\"><!--m10-i18n:modules.vocal-tract-frame.text.rename_542-->重命名</button></div>\n    </div><p class=\"minor preset-library-hint\"><!--m10-i18n:modules.vocal-tract-frame.text.after_switching_sets_configuration_saving_and_ipa_synthesis_use_the_selected_543-->切换套装后，保存构型和 IPA 合成使用所选套装，已有关键帧保留。</p>\n    <details class=\"preset-manager\"><summary><span><!--m10-i18n:modules.vocal-tract-frame.text.manage_saved_configurations_544-->管理已存构型</span><span id=\"presetCount\"></span></summary><div id=\"presetNames\"></div></details>");
  $('presetChart').before(sets);
  function adopt(value){library=value;const current=library.sets.find(s=>s.id===library.active);presets=current.presets;favorites=[...current.favorites];selected='';pendingDelete='';render();}
  async function deletePreset(p){
    if(working||pendingDelete!==p.id)return;
    const next=structuredClone(library),current=next.sets.find(s=>s.id===next.active);
    // Favorites and presets can be newer than the last loaded library snapshot.
    current.presets=presets.filter(item=>item.id!==p.id);
    current.favorites=favorites.filter(id=>id!==p.id);
    working=true;status(()=>msg("modules.vocal-tract-frame.text.deleting_value_221",{v0:()=>p.name}));render();
    try{
      adopt(await(await post('library/save',next)).json());onDeleted?.(p.id);
      status(()=>msg("modules.vocal-tract-frame.text.deleted_value_222",{v0:()=>p.name}));
    }catch(error){status(()=>join(()=>msg("modules.vocal-tract-frame.text.delete_failed_223"),()=>error.message));}
    finally{working=false;render();(sets.querySelector('[data-confirm-delete]')||sets.querySelector('summary')).focus();}
  }
  async function editSet(action){
    if(working||!isReady())return;const edit={action,name:$('presetSetName').value,id:$('presetSet').value};working=true;render();
    try{adopt(await(await post('library/edit',edit)).json());status(()=>join(()=>msg("modules.vocal-tract-frame.text.current_set_224"),()=>library.sets.find(s=>s.id===library.active).name));$('presetSetName').value='';}
    catch(e){status(()=>e.message);}finally{working=false;render();}
  }
  $('presetSet').onchange=()=>editSet('select');
  for(const [id,action] of [['presetSetCreate','create'],['presetSetCopy','copy'],['presetSetRename','rename']])$(id).onclick=()=>editSet(action);
  function renderSets(){
    const select=$('presetSet');select.replaceChildren();
    for(const set of library.sets){const o=document.createElement('option');o.value=set.id;bindText(o,()=>join(()=>set.name+' · ',()=>msg("modules.vocal-tract-frame.count.configurations",{count:set.presets.length})));select.append(o);}select.value=library.active;
    for(const e of sets.querySelectorAll('input,select,button'))e.disabled=working;
    $('presetNames').replaceChildren();
    bindText($('presetCount'),()=>msg("modules.vocal-tract-frame.count.configurations",{count:presets.length}));
    if(!presets.length){const empty=document.createElement('p');empty.className='minor';bindText(empty,()=>msg("modules.vocal-tract-frame.text.no_configurations_saved_in_this_set_226"));$('presetNames').append(empty);}
    for(const p of presets){
      const row=document.createElement('div'),input=document.createElement('input'),button=document.createElement('button'),remove=document.createElement('button');row.className='preset-name-row';row.dataset.presetId=p.id;row.dataset.confirming=String(pendingDelete===p.id);input.value=p.name;input.maxLength=80;bindAttribute(input,"aria-label",()=>join(()=>msg("modules.vocal-tract-frame.text.configuration_name_227"),()=>p.name));button.type='button';bindText(button,()=>msg("modules.vocal-tract-frame.text.save_name_228"));button.disabled=input.disabled=working||pendingDelete===p.id;
      button.onclick=async()=>{if(working)return;const next=presets.map(item=>item.id===p.id?{...item,name:input.value}:item);working=true;render();try{presets=await(await post('presets/save',{presets:next})).json();adopt(await(await post('library/load')).json());status(()=>msg("modules.vocal-tract-frame.text.configuration_name_saved_229"));}catch(e){status(()=>e.message);}finally{working=false;render();}};
      remove.type='button';remove.className='preset-delete';bindText(remove,()=>msg("modules.vocal-tract-frame.text.delete_230"));bindAttribute(remove,"aria-label",()=>join(()=>msg("modules.vocal-tract-frame.text.delete_configuration_231"),()=>p.name));remove.disabled=working||pendingDelete===p.id;
      remove.onclick=()=>{if(working)return;pendingDelete=p.id;render();sets.querySelector('[data-confirm-delete]').focus();};
      row.append(input,button,remove);
      if(pendingDelete===p.id){
        const confirmation=document.createElement('div'),message=document.createElement('span'),confirm=document.createElement('button'),cancel=document.createElement('button');
        confirmation.className='preset-delete-confirmation';bindText(message,()=>msg("modules.vocal-tract-frame.text.delete_value_232",{v0:()=>p.name}));bindText(confirm,()=>msg("modules.vocal-tract-frame.text.confirm_deletion_233"));confirm.dataset.confirmDelete=p.id;bindText(cancel,()=>msg("modules.vocal-tract-frame.text.cancel_151"));
        confirm.type=cancel.type='button';confirm.disabled=cancel.disabled=working;
        confirm.onclick=()=>deletePreset(p);cancel.onclick=()=>{if(working)return;pendingDelete='';render();[...sets.querySelectorAll('.preset-name-row')].find(e=>e.dataset.presetId===p.id)?.querySelector('button:last-child')?.focus();};
        confirmation.append(message,confirm,cancel);row.append(confirmation);
      }
      $('presetNames').append(row);
    }
  }

  function symbolButton(id){
    const entry=entries.get(id),button=document.createElement('button');
    button.type='button';button.className='preset-ipa';bindText(button,()=>entry.display);
    button.dataset.ipa=entry.insertText;button.dataset.name=entry.nameZh;button.dataset.symbolId=id;
    button.onclick=()=>choose(entry.insertText);return button;
  }
  function section(title){
    const section=document.createElement('section'),heading=document.createElement('h3');
    bindText(heading,()=>title);section.append(heading);return section;
  }
  const consonants=section(msg("modules.vocal-tract-frame.text.consonants_234")),scroller=document.createElement('div'),table=document.createElement('table');
  scroller.className='preset-consonants';bindAttribute(table,"aria-label",()=>msg("modules.vocal-tract-frame.text.international_phonetic_alphabet_consonant_chart_235"));
  const head=table.createTHead().insertRow();
  for(let i=0;i<=chart.consonants.columns.length;i++){const index=i,th=document.createElement('th');th.scope='col';bindText(th,()=>index?chart.consonants.columns[index-1]:msg("modules.vocal-tract-frame.text.manner_of_articulation_236"));head.append(th);}
  const body=table.createTBody();
  for(const row of chart.consonants.rows){
    const tr=body.insertRow(),th=document.createElement('th');th.scope='row';bindText(th,()=>row.label);tr.append(th);
    for(const ids of row.cells){const td=tr.insertCell();for(const id of ids)td.append(symbolButton(id));}
  }
  scroller.append(table);consonants.append(scroller);$('presetChart').append(consonants);
  const lower=document.createElement('div');lower.className='preset-chart-lower';
  const vowels=section(msg("modules.vocal-tract-frame.text.vowels_237")),map=document.createElement('div');map.className='preset-vowels';
  setHTML(map,()=>"<svg viewBox=\"0 0 100 100\" preserveAspectRatio=\"none\" aria-hidden=\"true\"><path d=\"M20 9H83V91H42ZM27 37H83M34 65H83M50 9L60 65L63 91\"/></svg><span class=\"vowel-label front\"><!--m10-i18n:modules.vocal-tract-frame.vowels.front-->前</span><span class=\"vowel-label central\"><!--m10-i18n:modules.vocal-tract-frame.text.central_545-->央</span><span class=\"vowel-label back\"><!--m10-i18n:modules.vocal-tract-frame.vowels.back-->后</span>");
  for(const point of chart.vowels){const pair=document.createElement('div');pair.className='preset-vowel-pair';pair.style.left=point.x+'%';pair.style.top=point.y+'%';for(const id of point.ids)pair.append(symbolButton(id));map.append(pair);}
  vowels.append(map);lower.append(vowels);
  const extras=section(msg("modules.vocal-tract-frame.text.other_consonants_239")),flow=document.createElement('div');flow.className='preset-symbol-flow';
  for(const id of chart.extras)flow.append(symbolButton(id));extras.append(flow);
  const hint=document.createElement('p');hint.className='minor';bindText(hint,()=>msg("modules.vocal-tract-frame.text.hover_for_symbol_names_lit_symbols_have_configurations_marks_bottom_bar_favo_240"));extras.append(hint);
  const custom=section(msg("modules.vocal-tract-frame.text.custom_symbols_241"));custom.id='customPresetSection';
  const customList=document.createElement('div');customList.id='customPresetSymbols';customList.className='preset-symbol-flow';custom.append(customList);
  const empty=document.createElement('p');empty.id='customPresetEmpty';empty.className='minor';bindText(empty,()=>msg("modules.vocal-tract-frame.text.use_enter_symbol_at_the_top_right_to_add_symbols_or_combinations_absent_from_242"));custom.append(empty);
  extras.append(custom);lower.append(extras);$('presetChart').append(lower);
  function render(){
    renderSets();
    $('presetChart').hidden=common;commonPanel.hidden=!common;commonPanel.replaceChildren();
    if(common)for(const [title,items] of [[msg("modules.vocal-tract-frame.text.built_in_configurations_243"),builtins],[msg("modules.vocal-tract-frame.text.saved_configurations_244"),presets]]){
      if(!items.length)continue;const group=section(title),buttons=document.createElement('div');buttons.className='preset-symbol-flow';
      for(const p of items){const b=document.createElement('button');b.type='button';bindText(b,()=>(favorites.includes(p.id)?'★ ':'')+p.name);b.dataset.commonId=p.id;b.dataset.commonName=p.name;b.disabled=working;b.setAttribute('aria-pressed',favorites.includes(p.id));bindAttribute(b,"aria-label",()=>join(()=>join(()=>join(()=>join(()=>msg("modules.vocal-tract-frame.text.favorite_245"),()=>title),()=>' /'),()=>p.name),()=>'/'));b.onclick=()=>toggleFavorite(p);buttons.append(b);}
      group.append(buttons);commonPanel.append(group);
    }
    const customPresets=presets.filter(p=>!presetSymbol(p.name,symbols));
    $('customPresetSymbols').replaceChildren();
    for(const p of customPresets){
      const button=document.createElement('button');button.type='button';button.className='preset-ipa custom-preset-ipa';
      bindText(button,()=>p.name);button.dataset.ipa=key(p.name);button.dataset.customIpa='true';button.dataset.name=msg("modules.vocal-tract-frame.text.custom_symbols_241");
      button.onclick=()=>choose(key(p.name));$('customPresetSymbols').append(button);
    }
    $('customPresetEmpty').hidden=!!customPresets.length;
    bindText($('customPresetEmpty'),()=>mode==='save'?msg("modules.vocal-tract-frame.text.use_enter_symbol_at_the_top_right_to_add_symbols_or_combinations_absent_from_242"):msg("modules.vocal-tract-frame.text.no_custom_symbols_saved_yet_246"));
    $('presetInputButton').hidden=mode!=='save';$('presetInputButton').disabled=working;
    $('presetInputSave').disabled=working;$('presetInputCancel').disabled=working;$('presetInput').disabled=working;
    for(const button of $('presetChart').querySelectorAll('[data-ipa]')){
      const p=stored(button.dataset.ipa),label=common?msg("modules.vocal-tract-frame.text.favorite_247"):mode==='save'?msg("modules.vocal-tract-frame.text.save_145"):msg("modules.vocal-tract-frame.text.load_248");
      button.disabled=working||(mode==='load'&&!p);button.classList.toggle('stored',!!p);
      button.classList.toggle('selected',!!p&&p.id===selected);button.classList.toggle('favorite',!!p&&favorites.includes(p.id));
      if(common)button.setAttribute('aria-pressed',!!p&&favorites.includes(p.id));else button.removeAttribute('aria-pressed');
      const symbolName=()=>button.dataset.customIpa?msg("modules.vocal-tract-frame.text.custom_symbols_241"):entries.get(button.dataset.symbolId)?.nameZh??button.dataset.name;
      bindAttribute(button,'title',()=>msg("modules.vocal-tract-frame.text.value_valuevalue_47",{v0:()=>button.dataset.ipa,v1:symbolName,v2:()=>p?(p.builtin?msg("modules.vocal-tract-frame.text.built_in_201"):msg("modules.vocal-tract-frame.text.saved_249")):''}));
      bindAttribute(button,"aria-label",()=>`${label} /${button.dataset.ipa}/ ${symbolName()}`);
    }
    bindAttribute($('customPreset'),'title',()=>msg("modules.vocal-tract-frame.text.select_a_configuration_or_set_bottom_bar_favorites_250"));
    $('presetFavorites').hidden=mode!=='load';$('presetFavorites').disabled=working;$('presetFavorites').setAttribute('aria-pressed',common);bindText($('presetFavorites'),()=>common?msg("modules.vocal-tract-frame.text.finish_favorites_setup_251"):msg("modules.vocal-tract-frame.text.set_favorites_252"));renderShortcuts();
  }
  function open(nextMode){
    if(working)return;common=false;pendingDelete='';mode=nextMode;bindText($('presetTitle'),()=>mode==='save'?msg("modules.vocal-tract-frame.text.save_configuration_choose_symbol_253"):msg("modules.vocal-tract-frame.text.saved_configurations_choose_symbol_254"));
    $('presetInputRow').hidden=true;$('presetInput').value='';
    status(()=>mode==='save'?msg("modules.vocal-tract-frame.text.click_a_symbol_to_save_current_configuration_and_source_existing_symbols_wil_255"):msg("modules.vocal-tract-frame.text.click_a_lit_symbol_to_load_set_favorites_places_configurations_on_the_bottom_256"));
    render();dialog.showModal();
  }
  $('presetInputButton').onclick=()=>{$('presetInputRow').hidden=false;$('presetInput').focus();};
  $('presetInputCancel').onclick=()=>{$('presetInputRow').hidden=true;$('presetInputButton').focus();};
  $('presetInputRow').onsubmit=event=>{
    event.preventDefault();if(working||mode!=='save')return;
    try{void choose(presetInputName($('presetInput').value,symbols));}
    catch(error){status(()=>error.message);$('presetInput').focus();}
  };
  function renderShortcuts(){
    const all=[...builtins,...presets];$('presets').replaceChildren();
    for(const id of favorites){const p=all.find(p=>p.id===id);if(!p)continue;const b=document.createElement('button');bindText(b,()=>p.name);b.dataset.preset=p.id;bindAttribute(b,'title',()=>join(()=>join(()=>join(()=>join(()=>msg("modules.vocal-tract-frame.text.load_257"),()=>(p.builtin?msg("modules.vocal-tract-frame.text.built_in_258"):msg("modules.vocal-tract-frame.text.saved_259"))),()=>' /'),()=>p.name),()=>msg("modules.vocal-tract-frame.text.configuration_260")));b.classList.toggle('selected',id===selected);b.onclick=()=>load(p);$('presets').append(b);}
  }
  $('presetFavorites').onclick=()=>{common=!common;status(()=>common?msg("modules.vocal-tract-frame.text.click_configurations_to_add_remove_favorites_means_selected_261"):msg("modules.vocal-tract-frame.text.click_a_lit_symbol_to_load_a_configuration_262"));render();};
  function load(p){if(!isReady()){status(()=>msg("modules.vocal-tract-frame.text.configuration_updating_click_again_in_a_moment_263"));return;}if(p.builtin)chooseBuiltin(p.id);else apply(p);selected=p.id;renderShortcuts();dialog.close();}
  async function toggleFavorite(p){
    if(working)return;working=true;render();
    try{const ids=favorites.includes(p.id)?favorites.filter(id=>id!==p.id):[...favorites,p.id];favorites=await(await post('favorites/save',{ids})).json();status(()=>msg("modules.vocal-tract-frame.text.favorites_updated_264"));}
    catch(e){status(()=>join(()=>msg("modules.vocal-tract-frame.text.save_failed_125"),()=>e.message));}finally{working=false;render();}
  }
  async function choose(symbol){
    if(working)return;
    if(mode==='load'){const p=stored(symbol);if(p){if(common)await toggleFavorite(p);else load(p);}return;}
    if(!isReady()){status(()=>msg("modules.vocal-tract-frame.text.configuration_updating_wait_before_saving_a_symbol_265"));return;}
    const previous=customStored(symbol),p={...getPose(),name:symbol,id:previous?.id||crypto.randomUUID(),duration:.6};
    const next=previous?presets.map(item=>item.id===previous.id?p:item):[...presets,p];
    working=true;status(()=>msg("modules.vocal-tract-frame.text.saving_value_266",{v0:()=>symbol}));render();
    try{const saved=await(await post('presets/save',{presets:next})).json();presets=saved;library=await(await post('library/load')).json();selected=p.id;onSaved(p);dialog.close();notify(()=>msg("modules.vocal-tract-frame.text.saved_value_configuration_and_source_settings_267",{v0:()=>symbol}));}
    catch(error){status(()=>join(()=>msg("modules.vocal-tract-frame.text.save_failed_125"),()=>error.message));}
    finally{working=false;render();}
  }
  $('customPreset').onclick=()=>open('load');$('savePreset').onclick=()=>open('save');
  $('presetCancel').onclick=()=>{if(!working)dialog.close();};
  const pack=setupPresetPack({getLibrary:()=>structuredClone(library),setLibrary:adopt,getPresets:()=>structuredClone(presets),getBuiltins:()=>structuredClone(builtins),setPresets:value=>{presets=value;render();},post,isReady,notify,onClose:()=>{render();dialog.showModal();}});
  const share=document.createElement('button');share.id='presetShare';bindText(share,()=>msg("modules.vocal-tract-frame.text.import_share_268"));share.type='button';share.onclick=()=>{if(!working){dialog.close();pack.open();}};
  $('presetCancel').before(share);
  dialog.addEventListener('cancel',event=>{if(working)event.preventDefault();});
  document.addEventListener('m10-ipa-chart',event=>{
    const next=event.detail;if(!next?.entries)return;
    for(const entry of next.entries){const existing=entries.get(entry.id);if(existing)existing.nameZh=entry.nameZh;}
    for(let i=0;i<chart.consonants.columns.length;i++)chart.consonants.columns[i]=next.consonants.columns[i]??chart.consonants.columns[i];
    for(let i=0;i<chart.consonants.rows.length;i++)chart.consonants.rows[i].label=next.consonants.rows[i]?.label??chart.consonants.rows[i].label;
    refreshBindings();
  });
  render();return {getCustom:()=>structuredClone(presets),getName:id=>[...presets,...builtins].find(p=>p.id===id)?.name||'',select:id=>{selected=id;}};
}
