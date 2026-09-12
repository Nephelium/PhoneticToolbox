export function setupPresets({initial,getPose,apply,post,isReady,onSaved}){
  const $=id=>document.getElementById(id);let presets=initial,working=false;
  const status=s=>$('presetStatus').textContent=s;
  function render(){
    const value=$('customPreset').value;$('customPreset').replaceChildren(new Option('自存构形…',''));
    for(const p of presets)$('customPreset').append(new Option(p.name,p.id));
    $('customPreset').value=presets.some(p=>p.id===value)?value:'';
    $('renamePreset').disabled=!$('customPreset').value||working;
  }
  $('customPreset').onchange=()=>{const p=presets.find(p=>p.id===$('customPreset').value);if(p)apply(p);render();};
  const dialog=$('presetDialog');let renamed=null;
  $('savePreset').onclick=()=>{if(!isReady())return;renamed=null;$('presetName').value='';status('保存当前器官构形，声源继续使用声音栏目中的设置。');dialog.showModal();$('presetName').focus();};
  $('renamePreset').onclick=()=>{if(working||!isReady())return;renamed=presets.find(p=>p.id===$('customPreset').value);if(!renamed)return;$('presetName').value=renamed.name;status('修改构形名称');dialog.showModal();$('presetName').focus();};
  $('presetCancel').onclick=()=>dialog.close();
  $('presetConfirm').onclick=async()=>{
    const name=$('presetName').value.trim();if(!name){status('请填写构形名称。');return;}
    const p=renamed?{...renamed,name}:{...getPose(),name,id:crypto.randomUUID(),duration:.6};
    const next=renamed?presets.map(v=>v.id===p.id?p:v):[...presets,p];working=true;$('presetConfirm').disabled=true;
    try{presets=await(await post('presets/save',{presets:next})).json();render();$('customPreset').value=p.id;render();onSaved(p);dialog.close();}
    catch(e){status('保存失败：'+e.message);}finally{working=false;$('presetConfirm').disabled=false;render();}
  };
  $('presetName').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('presetConfirm').click();}};
  render();return {getName:id=>presets.find(p=>p.id===id)?.name||'',select:id=>{$('customPreset').value=presets.some(p=>p.id===id)?id:'';render();}};
}
