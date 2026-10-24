import {setupPresetPack} from './preset-pack.js';

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
  if(!name)throw Error('请输入音标。');
  if([...name].length>80||/[\u0000-\u001f]/u.test(name))throw Error('音标须为不超过 80 字的单行文本。');
  return presetSymbol(name,symbols)||name;
}

export function setupPresets({initial,initialLibrary,builtins=[],favorites:initialFavorites=[],chooseBuiltin,chart,getPose,apply,post,isReady,onSaved,onDeleted,notify}){
  const $=id=>document.getElementById(id),dialog=$('presetDialog');
  const entries=new Map(chart.entries.map(entry=>[entry.id,entry]));
  const symbols=new Set(chart.entries.map(entry=>entry.insertText));
  let presets=initial,favorites=[...initialFavorites],working=false,mode='load',selected='',common=false;
  let library=initialLibrary,pendingDelete='';
  const status=message=>$('presetStatus').textContent=message;
  const key=name=>(presetSymbol(name,symbols)||(name||'').trim()).normalize('NFC');
  const customStored=symbol=>presets.findLast(p=>key(p.name)===symbol);
  const stored=symbol=>customStored(symbol)||builtins.find(p=>key(p.name)===symbol);
  const commonPanel=document.createElement('div');commonPanel.id='presetCommon';commonPanel.hidden=true;
  $('presetChart').before(commonPanel);
  const sets=document.createElement('div');sets.className='preset-libraries';
  sets.innerHTML=`<div class="preset-library-toolbar">
    <label>构型套装<select id="presetSet" aria-label="构型套装"></select></label>
    <label>套装名称<input id="presetSetName" maxlength="120" placeholder="输入套装名称"></label>
    <div class="preset-set-actions"><button id="presetSetCreate" type="button">新建空套装</button><button id="presetSetCopy" type="button">复制当前套装</button><button id="presetSetRename" type="button">重命名</button></div>
    </div><p class="minor preset-library-hint">切换套装后，保存构型和 IPA 合成使用所选套装，已有关键帧保留。</p>
    <details class="preset-manager"><summary><span>管理已存构型</span><span id="presetCount"></span></summary><div id="presetNames"></div></details>`;
  $('presetChart').before(sets);
  function adopt(value){library=value;const current=library.sets.find(s=>s.id===library.active);presets=current.presets;favorites=[...current.favorites];selected='';pendingDelete='';render();}
  async function deletePreset(p){
    if(working||pendingDelete!==p.id)return;
    const next=structuredClone(library),current=next.sets.find(s=>s.id===next.active);
    // Favorites and presets can be newer than the last loaded library snapshot.
    current.presets=presets.filter(item=>item.id!==p.id);
    current.favorites=favorites.filter(id=>id!==p.id);
    working=true;status(`正在删除 /${p.name}/…`);render();
    try{
      adopt(await(await post('library/save',next)).json());onDeleted?.(p.id);
      status(`已删除 /${p.name}/。`);
    }catch(error){status('删除失败：'+error.message);}
    finally{working=false;render();(sets.querySelector('[data-confirm-delete]')||sets.querySelector('summary')).focus();}
  }
  async function editSet(action){
    if(working||!isReady())return;const edit={action,name:$('presetSetName').value,id:$('presetSet').value};working=true;render();
    try{adopt(await(await post('library/edit',edit)).json());status('当前套装：'+library.sets.find(s=>s.id===library.active).name);$('presetSetName').value='';}
    catch(e){status(e.message);}finally{working=false;render();}
  }
  $('presetSet').onchange=()=>editSet('select');
  for(const [id,action] of [['presetSetCreate','create'],['presetSetCopy','copy'],['presetSetRename','rename']])$(id).onclick=()=>editSet(action);
  function renderSets(){
    const select=$('presetSet');select.replaceChildren();
    for(const set of library.sets){const o=document.createElement('option');o.value=set.id;o.textContent=set.name+' · '+set.presets.length+' 个';select.append(o);}select.value=library.active;
    for(const e of sets.querySelectorAll('input,select,button'))e.disabled=working;
    $('presetNames').replaceChildren();
    $('presetCount').textContent=presets.length+' 个';
    if(!presets.length){const empty=document.createElement('p');empty.className='minor';empty.textContent='当前套装尚未保存构型。';$('presetNames').append(empty);}
    for(const p of presets){
      const row=document.createElement('div'),input=document.createElement('input'),button=document.createElement('button'),remove=document.createElement('button');row.className='preset-name-row';row.dataset.presetId=p.id;row.dataset.confirming=String(pendingDelete===p.id);input.value=p.name;input.maxLength=80;input.setAttribute('aria-label','构型名称 '+p.name);button.type='button';button.textContent='保存名称';button.disabled=input.disabled=working||pendingDelete===p.id;
      button.onclick=async()=>{if(working)return;const next=presets.map(item=>item.id===p.id?{...item,name:input.value}:item);working=true;render();try{presets=await(await post('presets/save',{presets:next})).json();adopt(await(await post('library/load')).json());status('构型名称已保存。');}catch(e){status(e.message);}finally{working=false;render();}};
      remove.type='button';remove.className='preset-delete';remove.textContent='删除';remove.setAttribute('aria-label','删除构型 '+p.name);remove.disabled=working||pendingDelete===p.id;
      remove.onclick=()=>{if(working)return;pendingDelete=p.id;render();sets.querySelector('[data-confirm-delete]').focus();};
      row.append(input,button,remove);
      if(pendingDelete===p.id){
        const confirmation=document.createElement('div'),message=document.createElement('span'),confirm=document.createElement('button'),cancel=document.createElement('button');
        confirmation.className='preset-delete-confirmation';message.textContent=`删除 /${p.name}/？`;confirm.textContent='确认删除';confirm.dataset.confirmDelete=p.id;cancel.textContent='取消';
        confirm.type=cancel.type='button';confirm.disabled=cancel.disabled=working;
        confirm.onclick=()=>deletePreset(p);cancel.onclick=()=>{if(working)return;pendingDelete='';render();[...sets.querySelectorAll('.preset-name-row')].find(e=>e.dataset.presetId===p.id)?.querySelector('button:last-child')?.focus();};
        confirmation.append(message,confirm,cancel);row.append(confirmation);
      }
      $('presetNames').append(row);
    }
  }

  function symbolButton(id){
    const entry=entries.get(id),button=document.createElement('button');
    button.type='button';button.className='preset-ipa';button.textContent=entry.display;
    button.dataset.ipa=entry.insertText;button.dataset.name=entry.nameZh;
    button.onclick=()=>choose(entry.insertText);return button;
  }
  function section(title){
    const section=document.createElement('section'),heading=document.createElement('h3');
    heading.textContent=title;section.append(heading);return section;
  }
  const consonants=section('辅音'),scroller=document.createElement('div'),table=document.createElement('table');
  scroller.className='preset-consonants';table.setAttribute('aria-label','国际音标辅音表');
  const head=table.createTHead().insertRow();
  for(const label of ['构音方式',...chart.consonants.columns]){const th=document.createElement('th');th.scope='col';th.textContent=label;head.append(th);}
  const body=table.createTBody();
  for(const row of chart.consonants.rows){
    const tr=body.insertRow(),th=document.createElement('th');th.scope='row';th.textContent=row.label;tr.append(th);
    for(const ids of row.cells){const td=tr.insertCell();for(const id of ids)td.append(symbolButton(id));}
  }
  scroller.append(table);consonants.append(scroller);$('presetChart').append(consonants);
  const lower=document.createElement('div');lower.className='preset-chart-lower';
  const vowels=section('元音'),map=document.createElement('div');map.className='preset-vowels';
  map.innerHTML='<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M20 9H83V91H42ZM27 37H83M34 65H83M50 9L60 65L63 91"/></svg><span class="vowel-label front">前</span><span class="vowel-label central">央</span><span class="vowel-label back">后</span>';
  for(const point of chart.vowels){const pair=document.createElement('div');pair.className='preset-vowel-pair';pair.style.left=point.x+'%';pair.style.top=point.y+'%';for(const id of point.ids)pair.append(symbolButton(id));map.append(pair);}
  vowels.append(map);lower.append(vowels);
  const extras=section('其他辅音'),flow=document.createElement('div');flow.className='preset-symbol-flow';
  for(const id of chart.extras)flow.append(symbolButton(id));extras.append(flow);
  const hint=document.createElement('p');hint.className='minor';hint.textContent='悬停查看音标名称；点亮表示可用构形，★ 表示已放到底部常用栏。';extras.append(hint);
  const custom=section('自定义音标');custom.id='customPresetSection';
  const customList=document.createElement('div');customList.id='customPresetSymbols';customList.className='preset-symbol-flow';custom.append(customList);
  const empty=document.createElement('p');empty.id='customPresetEmpty';empty.className='minor';empty.textContent='点击右上角输入音标，添加表中没有的音标或组合。';custom.append(empty);
  extras.append(custom);lower.append(extras);$('presetChart').append(lower);
  function render(){
    renderSets();
    $('presetChart').hidden=common;commonPanel.hidden=!common;commonPanel.replaceChildren();
    if(common)for(const [title,items] of [['内置构形',builtins],['已存构型',presets]]){
      if(!items.length)continue;const group=section(title),buttons=document.createElement('div');buttons.className='preset-symbol-flow';
      for(const p of items){const b=document.createElement('button');b.type='button';b.textContent=(favorites.includes(p.id)?'★ ':'')+p.name;b.dataset.commonId=p.id;b.dataset.commonName=p.name;b.disabled=working;b.setAttribute('aria-pressed',favorites.includes(p.id));b.setAttribute('aria-label','常用 '+title+' /'+p.name+'/');b.onclick=()=>toggleFavorite(p);buttons.append(b);}
      group.append(buttons);commonPanel.append(group);
    }
    const customPresets=presets.filter(p=>!presetSymbol(p.name,symbols));
    $('customPresetSymbols').replaceChildren();
    for(const p of customPresets){
      const button=document.createElement('button');button.type='button';button.className='preset-ipa custom-preset-ipa';
      button.textContent=p.name;button.dataset.ipa=key(p.name);button.dataset.customIpa='true';button.dataset.name='自定义音标';
      button.onclick=()=>choose(key(p.name));$('customPresetSymbols').append(button);
    }
    $('customPresetEmpty').hidden=!!customPresets.length;
    $('customPresetEmpty').textContent=mode==='save'?'点击右上角输入音标，添加表中没有的音标或组合。':'尚未保存自定义音标。';
    $('presetInputButton').hidden=mode!=='save';$('presetInputButton').disabled=working;
    $('presetInputSave').disabled=working;$('presetInputCancel').disabled=working;$('presetInput').disabled=working;
    for(const button of $('presetChart').querySelectorAll('[data-ipa]')){
      const p=stored(button.dataset.ipa),label=common?'常用':mode==='save'?'保存':'加载';
      button.disabled=working||(mode==='load'&&!p);button.classList.toggle('stored',!!p);
      button.classList.toggle('selected',!!p&&p.id===selected);button.classList.toggle('favorite',!!p&&favorites.includes(p.id));
      if(common)button.setAttribute('aria-pressed',!!p&&favorites.includes(p.id));else button.removeAttribute('aria-pressed');
      button.title=`${button.dataset.ipa} · ${button.dataset.name}${p?(p.builtin?' · 内置':' · 已保存'):''}`;
      button.setAttribute('aria-label',`${label} /${button.dataset.ipa}/ ${button.dataset.name}`);
    }
    $('customPreset').title='选择构形或设置底部常用栏';
    $('presetFavorites').hidden=mode!=='load';$('presetFavorites').disabled=working;$('presetFavorites').setAttribute('aria-pressed',common);$('presetFavorites').textContent=common?'完成常用设置':'设置常用 ★';renderShortcuts();
  }
  function open(nextMode){
    if(working)return;common=false;pendingDelete='';mode=nextMode;$('presetTitle').textContent=mode==='save'?'保存构形 · 选择音标':'已存构型 · 选择音标';
    $('presetInputRow').hidden=true;$('presetInput').value='';
    status(mode==='save'?'点击音标保存当前构形与声源设置；已有音标将更新。':'点击点亮的音标载入；设置常用可把构形放到底部。');
    render();dialog.showModal();
  }
  $('presetInputButton').onclick=()=>{$('presetInputRow').hidden=false;$('presetInput').focus();};
  $('presetInputCancel').onclick=()=>{$('presetInputRow').hidden=true;$('presetInputButton').focus();};
  $('presetInputRow').onsubmit=event=>{
    event.preventDefault();if(working||mode!=='save')return;
    try{void choose(presetInputName($('presetInput').value,symbols));}
    catch(error){status(error.message);$('presetInput').focus();}
  };
  function renderShortcuts(){
    const all=[...builtins,...presets];$('presets').replaceChildren();
    for(const id of favorites){const p=all.find(p=>p.id===id);if(!p)continue;const b=document.createElement('button');b.textContent=p.name;b.dataset.preset=p.id;b.title='载入'+(p.builtin?'内置':'已存')+' /'+p.name+'/ 构形';b.classList.toggle('selected',id===selected);b.onclick=()=>load(p);$('presets').append(b);}
  }
  $('presetFavorites').onclick=()=>{common=!common;status(common?'点击构形加入或移出常用栏，★ 表示已选。':'点击点亮的音标载入构形。');render();};
  function load(p){if(!isReady()){status('正在更新构形，请稍后再次点击。');return;}if(p.builtin)chooseBuiltin(p.id);else apply(p);selected=p.id;renderShortcuts();dialog.close();}
  async function toggleFavorite(p){
    if(working)return;working=true;render();
    try{const ids=favorites.includes(p.id)?favorites.filter(id=>id!==p.id):[...favorites,p.id];favorites=await(await post('favorites/save',{ids})).json();status('常用栏已更新。');}
    catch(e){status('保存失败：'+e.message);}finally{working=false;render();}
  }
  async function choose(symbol){
    if(working)return;
    if(mode==='load'){const p=stored(symbol);if(p){if(common)await toggleFavorite(p);else load(p);}return;}
    if(!isReady()){status('正在更新构形，请稍后再次点击音标保存。');return;}
    const previous=customStored(symbol),p={...getPose(),name:symbol,id:previous?.id||crypto.randomUUID(),duration:.6};
    const next=previous?presets.map(item=>item.id===previous.id?p:item):[...presets,p];
    working=true;status(`正在保存 /${symbol}/…`);render();
    try{const saved=await(await post('presets/save',{presets:next})).json();presets=saved;library=await(await post('library/load')).json();selected=p.id;onSaved(p);dialog.close();notify(`已保存 /${symbol}/ 构形与声源设置`);}
    catch(error){status('保存失败：'+error.message);}
    finally{working=false;render();}
  }
  $('customPreset').onclick=()=>open('load');$('savePreset').onclick=()=>open('save');
  $('presetCancel').onclick=()=>{if(!working)dialog.close();};
  const pack=setupPresetPack({getLibrary:()=>structuredClone(library),setLibrary:adopt,getPresets:()=>structuredClone(presets),getBuiltins:()=>structuredClone(builtins),setPresets:value=>{presets=value;render();},post,isReady,notify,onClose:()=>{render();dialog.showModal();}});
  const share=document.createElement('button');share.id='presetShare';share.textContent='导入 / 分享';share.type='button';share.onclick=()=>{if(!working){dialog.close();pack.open();}};
  $('presetCancel').before(share);
  dialog.addEventListener('cancel',event=>{if(working)event.preventDefault();});
  render();return {getCustom:()=>structuredClone(presets),getName:id=>[...presets,...builtins].find(p=>p.id===id)?.name||'',select:id=>{selected=id;}};
}
