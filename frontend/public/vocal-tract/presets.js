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

export function setupPresets({initial,chart,getPose,apply,post,isReady,onSaved,notify}){
  const $=id=>document.getElementById(id),dialog=$('presetDialog');
  const entries=new Map(chart.entries.map(entry=>[entry.id,entry]));
  const symbols=new Set(chart.entries.map(entry=>entry.insertText));
  let presets=initial,working=false,mode='load',selected='';
  const status=message=>$('presetStatus').textContent=message;
  const key=name=>presetSymbol(name,symbols)||(name||'').trim();
  const stored=symbol=>presets.findLast(p=>key(p.name)===symbol);
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
  const hint=document.createElement('p');hint.className='minor';hint.textContent='悬停查看音标名称；标记表示已有自存构形。';extras.append(hint);
  const custom=section('自定义音标');custom.id='customPresetSection';
  const customList=document.createElement('div');customList.id='customPresetSymbols';customList.className='preset-symbol-flow';custom.append(customList);
  const empty=document.createElement('p');empty.id='customPresetEmpty';empty.className='minor';empty.textContent='点击右上角输入音标，添加表中没有的音标或组合。';custom.append(empty);
  extras.append(custom);lower.append(extras);$('presetChart').append(lower);
  function render(){
    const customPresets=presets.filter(p=>!presetSymbol(p.name,symbols));
    $('customPresetSymbols').replaceChildren();
    for(const p of customPresets){
      const button=document.createElement('button');button.type='button';button.className='preset-ipa custom-preset-ipa';
      button.textContent=p.name;button.dataset.ipa=key(p.name);button.dataset.customIpa='true';button.dataset.name='自定义音标';
      button.onclick=()=>mode==='load'?load(p):choose(key(p.name));$('customPresetSymbols').append(button);
    }
    $('customPresetEmpty').hidden=!!customPresets.length;
    $('customPresetEmpty').textContent=mode==='save'?'点击右上角输入音标，添加表中没有的音标或组合。':'尚未保存自定义音标。';
    $('presetInputButton').hidden=mode!=='save';$('presetInputButton').disabled=working;
    $('presetInputSave').disabled=working;$('presetInputCancel').disabled=working;$('presetInput').disabled=working;
    for(const button of $('presetChart').querySelectorAll('[data-ipa]')){
      const p=stored(button.dataset.ipa),label=mode==='save'?'保存':'加载';
      button.disabled=working||(mode==='load'&&!p);button.classList.toggle('stored',!!p);
      button.classList.toggle('selected',!!p&&p.id===selected);
      button.title=`${button.dataset.ipa} · ${button.dataset.name}${p?' · 已保存':''}`;
      button.setAttribute('aria-label',`${label} /${button.dataset.ipa}/ ${button.dataset.name}`);
    }
    $('customPreset').title=presets.length?`加载本机保存的 ${presets.length} 个构形`:'尚未保存构形';
  }
  function open(nextMode){
    if(working)return;mode=nextMode;$('presetTitle').textContent=mode==='save'?'保存构形 · 选择音标':'自存构形 · 选择音标';
    $('presetInputRow').hidden=true;$('presetInput').value='';
    status(mode==='save'?'点击音标保存当前构形与声源设置；已有音标将更新。':presets.length?'点击已保存的音标加载构形与声源设置。':'尚无自存构形。先调整构形，再点击保存构形。');
    render();dialog.showModal();
  }
  $('presetInputButton').onclick=()=>{$('presetInputRow').hidden=false;$('presetInput').focus();};
  $('presetInputCancel').onclick=()=>{$('presetInputRow').hidden=true;$('presetInputButton').focus();};
  $('presetInputRow').onsubmit=event=>{
    event.preventDefault();if(working||mode!=='save')return;
    try{void choose(presetInputName($('presetInput').value,symbols));}
    catch(error){status(error.message);$('presetInput').focus();}
  };
  function load(p){if(!isReady()){status('正在更新构形，请稍后再次点击。');return;}apply(p);selected=p.id;dialog.close();}
  async function choose(symbol){
    if(working)return;
    if(mode==='load'){const p=stored(symbol);if(p)load(p);return;}
    if(!isReady()){status('正在更新构形，请稍后再次点击音标保存。');return;}
    const previous=stored(symbol),p={...getPose(),name:symbol,id:previous?.id||crypto.randomUUID(),duration:.6};
    const next=previous?presets.map(item=>item.id===previous.id?p:item):[...presets,p];
    working=true;status(`正在保存 /${symbol}/…`);render();
    try{const saved=await(await post('presets/save',{presets:next})).json();presets=saved;selected=p.id;onSaved(p);dialog.close();notify(`已保存 /${symbol}/ 构形与声源设置`);}
    catch(error){status('保存失败：'+error.message);}
    finally{working=false;render();}
  }
  $('customPreset').onclick=()=>open('load');$('savePreset').onclick=()=>open('save');
  $('presetCancel').onclick=()=>{if(!working)dialog.close();};
  dialog.addEventListener('cancel',event=>{if(working)event.preventDefault();});
  render();return {getName:id=>presets.find(p=>p.id===id)?.name||'',select:id=>{selected=id;}};
}
