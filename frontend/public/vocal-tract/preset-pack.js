/* One portable configuration library for native and browser file adapters. */
export function setupPresetPack({getLibrary,setLibrary,getPresets,getBuiltins=()=>[],setPresets,post,isReady,notify,onClose}){
  const dialog=document.createElement('dialog');dialog.id='presetPackDialog';
  dialog.innerHTML='<div class="preset-dialog-heading"><strong>导入与分享构型</strong><button id="packClose" class="quiet">返回构型库</button></div><div class="pack-toolbar"><button id="packOpen">打开构型包</button><button id="packOwn">选择自己的构型</button></div><label class="pack-title">构型包名称 <input id="packTitle" maxlength="120" value="声道构型包"></label><p id="packInfo" class="minor"></p><div class="pack-select"><button id="packAll">全选</button><button id="packNone">全不选</button><span id="packCount"></span></div><div id="packItems" class="pack-items"></div><label id="packTargetRow" hidden>导入到 <select id="packTarget"><option value="new">新套装</option><option value="current">当前套装</option></select></label><label id="packRecomputeRow" hidden><input id="packRecompute" type="checkbox">按当前几何重算后导入</label><label id="packConflictRow" hidden>同名音标 <select id="packConflict"><option value="keep">保留已有构型</option><option value="replace">使用导入构型替换</option></select></label><p id="packStatus" role="status"></p><div class="dialog-actions"><button id="packImport" class="primary" hidden>导入所选构型</button><button id="packExport" class="primary">导出所选构型</button></div>';
  document.body.append(dialog);const $=id=>dialog.querySelector('#'+id);
  const symbol=p=>(p.name||'').trim().replace(/^[/\[]|[/\]]$/g,'').normalize('NFC');
  let items=[],selection=new Set(),importing=false,working=false,ownKeys=new Set(),incoming=null;
  function render(){
    $('packItems').replaceChildren();const existing=new Set(getPresets().map(symbol));
    for(const [i,p] of items.entries()){
      const label=document.createElement('label'),input=document.createElement('input'),name=document.createElement('span');input.type='checkbox';input.checked=selection.has(i);input.disabled=working;
      input.onchange=()=>{input.checked?selection.add(i):selection.delete(i);$('packStatus').textContent=`已选择 ${selection.size} 个构型。`;update();};name.textContent='/'+symbol(p)+'/'+(importing?(existing.has(symbol(p))?' · 已有同名构型':''):(ownKeys.has(symbol(p))?' · 自存':' · 内置'));label.append(input,name);$('packItems').append(label);
    }
    $('packImport').hidden=!importing;$('packExport').hidden=importing;$('packTargetRow').hidden=!importing;$('packConflictRow').hidden=!importing||$('packTarget').value!=='current';$('packRecomputeRow').hidden=!importing||!incoming?.geometry_notice;
    update();
  }
  function update(){
    $('packCount').textContent=`已选 ${selection.size} / ${items.length}`;
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
    $('packInfo').textContent=`当前模型：${saved.length} 个自存构型，${items.length-saved.length} 个内置构型。勾选后导出，包含器官参数与声源设置。`;
    $('packStatus').textContent=saved.length?`已选择全部 ${saved.length} 个自存构型。`:'当前模型尚无自存构型。可以勾选内置构型，或返回构型库保存当前姿势。';render();
  }
  async function refreshOwn(){
    if(working)return;working=true;update();
    try{setLibrary(await(await post('library/load')).json());own();}
    catch(e){$('packStatus').textContent='读取构型库失败：'+e.message;}
    finally{working=false;render();}
  }
  $('packTarget').onchange=render;$('packRecompute').onchange=update;
  $('packOwn').onclick=refreshOwn;
  $('packAll').onclick=()=>{selection=new Set(items.map((_,i)=>i));$('packStatus').textContent=`已选择全部 ${items.length} 个构型。`;render();};
  $('packNone').onclick=()=>{selection.clear();$('packStatus').textContent='已取消全部选择。';render();};
  $('packClose').onclick=()=>{if(!working){dialog.close();onClose();}};
  dialog.addEventListener('cancel',e=>{e.preventDefault();if(!working){dialog.close();onClose();}});
  $('packOpen').onclick=async()=>{
    working=true;update();
    try{
      const result=await(await post('preset-pack/open')).json();if(result.cancelled)return;
      incoming=result.document;$('packTarget').value='new';$('packRecompute').checked=false;items=result.document.presets;selection=new Set(items.map((_,i)=>i));importing=true;const titles=new Set(getLibrary().sets.map(s=>s.name)),base=result.document.title.slice(0,70);let title=base,n=2;while(titles.has(title))title=base+'（'+n+++'）';$('packTitle').value=title;
      $('packInfo').textContent=(result.document.speaker==='W02'?'女性':'男性')+'声道模型 · '+items.length+' 个音标。确认后导入，随即用于 IPA 合成。';$('packStatus').textContent=(incoming.geometry_notice?incoming.geometry_notice+' ':'')+'已读取构型包，尚未写入构型库。新套装使用上方名称。';render();
    }catch(e){$('packStatus').textContent=e.message;}finally{working=false;update();}
  };
  $('packExport').onclick=async()=>{
    working=true;update();
    try{const value=await(await post('preset-pack/save',{title:$('packTitle').value,presets:items.filter((_,i)=>selection.has(i))})).json();if(value.saved)$('packStatus').textContent='已导出 '+value.name+'，可把文件分享给使用相同声道模型的其他人。';}
    catch(e){$('packStatus').textContent=e.message;}finally{working=false;update();}
  };
  $('packImport').onclick=async()=>{
    if(!isReady()){$('packStatus').textContent='请等当前编辑或播放结束后再导入。';return;}
    working=true;update();
    try{
      const result=await(await post('library/import',{document:{...incoming,presets:items.filter((_,i)=>selection.has(i))},target:$('packTarget').value,name:$('packTitle').value,conflict:$('packConflict').value,recompute:$('packRecompute').checked})).json();
      setLibrary(result.library);const {added,replaced,kept}=result;
      own();$('packStatus').textContent=`新增 ${added} 个，替换 ${replaced} 个，保留 ${kept} 个同名构型。现在可直接输入这些音标生成序列。`;notify('构型库已更新');
    }catch(e){$('packStatus').textContent=e.message;}finally{working=false;update();}
  };
  return {open(){own();dialog.showModal();void refreshOwn();}};
}
