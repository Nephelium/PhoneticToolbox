import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
/* One portable configuration library for native and browser file adapters. */
export function setupPresetPack({getLibrary,setLibrary,getPresets,getBuiltins=()=>[],setPresets,post,isReady,notify,onClose}){
  const dialog=document.createElement('dialog');dialog.id='presetPackDialog';
  setHTML(dialog,()=>"<div class=\"preset-dialog-heading\"><strong><!--m10-i18n:modules.vocal-tract-frame.text.import_and_share_configurations_520-->导入与分享构型</strong><button id=\"packClose\" class=\"quiet\"><!--m10-i18n:modules.vocal-tract-frame.text.return_to_library_521-->返回构型库</button></div><div class=\"pack-toolbar\"><button id=\"packOpen\"><!--m10-i18n:modules.vocal-tract-frame.text.open_configuration_package_522-->打开构型包</button><button id=\"packOwn\"><!--m10-i18n:modules.vocal-tract-frame.text.select_own_configurations_523-->选择自己的构型</button></div><label class=\"pack-title\"><!--m10-i18n:modules.vocal-tract-frame.text.package_name_524-->构型包名称 <input id=\"packTitle\" maxlength=\"120\" value=\"声道构型包\"></label><p id=\"packInfo\" class=\"minor\"></p><div class=\"pack-select\"><button id=\"packAll\"><!--m10-i18n:modules.vocal-tract-frame.text.select_all_526-->全选</button><button id=\"packNone\"><!--m10-i18n:modules.vocal-tract-frame.text.select_none_527-->全不选</button><span id=\"packCount\"></span></div><div id=\"packItems\" class=\"pack-items\"></div><label id=\"packTargetRow\" hidden><!--m10-i18n:modules.vocal-tract-frame.text.import_into_528-->导入到 <select id=\"packTarget\"><option value=\"new\"><!--m10-i18n:modules.vocal-tract-frame.text.new_set_529-->新套装</option><option value=\"current\"><!--m10-i18n:modules.vocal-tract-frame.text.current_set_530-->当前套装</option></select></label><label id=\"packRecomputeRow\" hidden><input id=\"packRecompute\" type=\"checkbox\"><!--m10-i18n:modules.vocal-tract-frame.text.recompute_with_current_geometry_before_importing_531-->按当前几何重算后导入</label><label id=\"packConflictRow\" hidden><!--m10-i18n:modules.vocal-tract-frame.text.same_named_symbols_532-->同名音标 <select id=\"packConflict\"><option value=\"keep\"><!--m10-i18n:modules.vocal-tract-frame.text.keep_existing_configurations_533-->保留已有构型</option><option value=\"replace\"><!--m10-i18n:modules.vocal-tract-frame.text.replace_with_imported_configurations_534-->使用导入构型替换</option></select></label><p id=\"packStatus\" role=\"status\"></p><div class=\"dialog-actions\"><button id=\"packImport\" class=\"primary\" hidden><!--m10-i18n:modules.vocal-tract-frame.text.import_selected_configurations_535-->导入所选构型</button><button id=\"packExport\" class=\"primary\"><!--m10-i18n:modules.vocal-tract-frame.text.export_selected_configurations_536-->导出所选构型</button></div>");
  document.body.append(dialog);const $=id=>dialog.querySelector('#'+id);
  const symbol=p=>(p.name||'').trim().replace(/^[/\[]|[/\]]$/g,'').normalize('NFC');
  let items=[],selection=new Set(),importing=false,working=false,ownKeys=new Set(),incoming=null;
  function render(){
    $('packItems').replaceChildren();const existing=new Set(getPresets().map(symbol));
    for(const [i,p] of items.entries()){
      const label=document.createElement('label'),input=document.createElement('input'),name=document.createElement('span');input.type='checkbox';input.checked=selection.has(i);input.disabled=working;
      input.onchange=()=>{input.checked?selection.add(i):selection.delete(i);bindText($('packStatus'),()=>msg("modules.vocal-tract-frame.text.selected_configurations_value_198",{v0:()=>selection.size}));update();};bindText(name,()=>join(()=>'/'+symbol(p)+'/',()=>(importing?(existing.has(symbol(p))?msg("modules.vocal-tract-frame.text.same_named_configuration_exists_199"):''):(ownKeys.has(symbol(p))?msg("modules.vocal-tract-frame.text.user_saved_200"):msg("modules.vocal-tract-frame.text.built_in_201")))));label.append(input,name);$('packItems').append(label);
    }
    $('packImport').hidden=!importing;$('packExport').hidden=importing;$('packTargetRow').hidden=!importing;$('packConflictRow').hidden=!importing||$('packTarget').value!=='current';$('packRecomputeRow').hidden=!importing||!incoming?.geometry_notice;
    update();
  }
  function update(){
    bindText($('packCount'),()=>msg("modules.vocal-tract-frame.text.selected_value_value_202",{v0:()=>selection.size,v1:()=>items.length}));
    for(const e of dialog.querySelectorAll('input,select,button'))e.disabled=working;
    $('packImport').disabled=$('packExport').disabled=working||!selection.size;
    if(importing&&incoming?.geometry_notice&&!$('packRecompute').checked)$('packImport').disabled=true;
    $('packAll').disabled=working||!items.length||selection.size===items.length;
    $('packNone').disabled=working||!selection.size;
  }
  function own(){
    importing=false;const saved=getPresets();ownKeys=new Set(saved.map(symbol));
    // The chart includes bundled author defaults too. Show the same catalogue,
    // with a saved override taking precedence over its bundled namesake.
    items=[...saved,...getBuiltins().filter(p=>!ownKeys.has(symbol(p)))];
    selection=new Set(items.flatMap((p,i)=>ownKeys.has(symbol(p))?[i]:[]));
    $('packTitle').value=getLibrary().sets.find(s=>s.id===getLibrary().active).name;
    bindText($('packInfo'),()=>msg("modules.vocal-tract-frame.pack.summary",{speaker:result.document.speaker==='W02'?msg("modules.vocal-tract-frame.text.female_209"):msg("modules.vocal-tract-frame.text.male_210"),count:items.length}));bindText($('packStatus'),()=>msg("modules.vocal-tract-frame.text.all_configurations_selected_value_207",{v0:()=>items.length}));render();};
  $('packNone').onclick=()=>{selection.clear();bindText($('packStatus'),()=>msg("modules.vocal-tract-frame.text.all_selections_cleared_208"));render();};
  $('packClose').onclick=()=>{if(!working){dialog.close();onClose();}};
  dialog.addEventListener('cancel',e=>{e.preventDefault();if(!working){dialog.close();onClose();}});
  $('packOpen').onclick=async()=>{
    working=true;update();
    try{
      const result=await(await post('preset-pack/open')).json();if(result.cancelled)return;
      incoming=result.document;$('packTarget').value='new';$('packRecompute').checked=false;items=result.document.presets;selection=new Set(items.map((_,i)=>i));importing=true;const titles=new Set(getLibrary().sets.map(s=>s.name)),base=result.document.title.slice(0,70);let title=base,n=2;while(titles.has(title))title=base+'（'+n+++'）';$('packTitle').value=title;
      bindText($('packInfo'),()=>join(()=>join(()=>join(()=>(result.document.speaker==='W02'?msg("modules.vocal-tract-frame.text.female_209"):msg("modules.vocal-tract-frame.text.male_210")),()=>msg("modules.vocal-tract-frame.text.vocal_tract_model_211")),()=>items.length),()=>msg("modules.vocal-tract-frame.text.symbols_confirm_import_to_use_them_for_ipa_synthesis_212")));bindText($('packStatus'),()=>join(()=>(incoming.geometry_notice?incoming.geometry_notice+' ':''),()=>msg("modules.vocal-tract-frame.text.package_loaded_not_yet_written_to_the_library_the_new_set_uses_the_name_abov_213")));render();
    }catch(e){bindText($('packStatus'),()=>e.message);}finally{working=false;update();}
  };
  $('packExport').onclick=async()=>{
    working=true;update();
    try{const value=await(await post('preset-pack/save',{title:$('packTitle').value,presets:items.filter((_,i)=>selection.has(i))})).json();if(value.saved)bindText($('packStatus'),()=>join(()=>join(()=>msg("modules.vocal-tract-frame.text.exported_166"),()=>value.name),()=>msg("modules.vocal-tract-frame.text.share_this_file_with_others_using_the_same_vocal_tract_model_214")));}
    catch(e){bindText($('packStatus'),()=>e.message);}finally{working=false;update();}
  };
  $('packImport').onclick=async()=>{
    if(!isReady()){bindText($('packStatus'),()=>msg("modules.vocal-tract-frame.text.finish_current_editing_or_playback_before_importing_215"));return;}
    working=true;update();
    try{
      const result=await(await post('library/import',{document:{...incoming,presets:items.filter((_,i)=>selection.has(i))},target:$('packTarget').value,name:$('packTitle').value,conflict:$('packConflict').value,recompute:$('packRecompute').checked})).json();
      setLibrary(result.library);const {added,replaced,kept}=result;
      own();bindText($('packStatus'),()=>msg("modules.vocal-tract-frame.text.added_value_replaced_value_retained_same_named_value_these_symbols_are_now_a_216",{v0:()=>added,v1:()=>replaced,v2:()=>kept}));notify(()=>msg("modules.vocal-tract-frame.text.configuration_library_updated_217"));
    }catch(e){bindText($('packStatus'),()=>e.message);}finally{working=false;update();}
  };
  return {open(){own();dialog.showModal();void refreshOwn();}};
}
