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

export function setupPresets({initial,builtins=[],favorites:initialFavorites=[],chooseBuiltin,chart,getPose,apply,post,isReady,onSaved,notify}){
  const $=id=>document.getElementById(id),dialog=$('presetDialog');
  const entries=new Map(chart.entries.map(entry=>[entry.id,entry]));
  const symbols=new Set(chart.entries.map(entry=>entry.insertText));
  let presets=initial,favorites=[...initialFavorites],working=false,mode='load',selected='',common=false;
  const status=message=>$('presetStatus').textContent=message;
  const key=name=>presetSymbol(name,symbols)||(name||'').trim();
  const customStored=symbol=>presets.findLast(p=>key(p.name)===symbol);
  const stored=symbol=>customStored(symbol)||builtins.find(p=>key(p.name)===symbol);
  const commonPanel=document.createElement('div');commonPanel.id='presetCommon';commonPanel.hidden=true;
  $('presetChart').before(commonPanel);
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
    if(working)return;common=false;mode=nextMode;$('presetTitle').textContent=mode==='save'?'保存构形 · 选择音标':'已存构型 · 选择音标';
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
    try{const saved=await(await post('presets/save',{presets:next})).json();presets=saved;selected=p.id;onSaved(p);dialog.close();notify(`已保存 /${symbol}/ 构形与声源设置`);}
    catch(error){status('保存失败：'+error.message);}
    finally{working=false;render();}
  }
  $('customPreset').onclick=()=>open('load');$('savePreset').onclick=()=>open('save');
  $('presetCancel').onclick=()=>{if(!working)dialog.close();};
  dialog.addEventListener('cancel',event=>{if(working)event.preventDefault();});
  render();return {getName:id=>[...presets,...builtins].find(p=>p.id===id)?.name||'',select:id=>{selected=id;}};
}
