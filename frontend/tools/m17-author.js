const $=id=>document.getElementById(id),form=$('form'),fields=['nameZh','nameEn','descriptionZh','usageZh','contrastZh','notesZh'];
const session=location.hash.slice(1);history.replaceState(null,'',location.pathname);
let data,entry,dirty=false,saving=false;
async function api(method,body,path='/__m17_author/content'){
 const response=await fetch(path,{method,headers:{'X-M17-Session':session,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
 const result=await response.json();if(!response.ok)throw Error(result.message);return result;
}
const field=name=>form.elements.namedItem(name);
function status(value){$('status').textContent=value;}
function content(){
 const value=Object.fromEntries(fields.map(key=>[key,field(key).value]));
 const media={};for(const key of ['audio','video'])if(field(key).value.trim())media[key]=field(key).value.trim();
 if(field('animation').value.trim())media.animation=JSON.parse(field('animation').value);
 if(Object.keys(media).length)value.media=media;return value;
}
function preview(){const fragment=document.createDocumentFragment();for(const key of fields){const p=document.createElement('p');p.textContent=field(key).value;fragment.append(p);}$('preview').replaceChildren(fragment);}
function load(id){
 entry=data.catalog.find(e=>e.id===id);const value={...entry,...data.entries[id]};
 $('title').textContent=entry.display+' · '+value.nameZh;$('id').textContent=id;
 for(const key of fields)field(key).value=value[key]??'';
 for(const key of ['audio','video'])field(key).value=value.media?.[key]??'';
 field('animation').value=value.media?.animation?JSON.stringify(value.media.animation,null,2):'';
 dirty=false;status(data.entries[id]?'已载入保存内容。':'已载入原介绍。');preview();
}
function list(){
 const q=$('search').value.trim().toLowerCase(),options=data.catalog.filter(e=>[e.display,e.nameZh,e.nameEn,e.id].join(' ').toLowerCase().includes(q)).map(e=>{const o=document.createElement('option');o.value=e.id;o.textContent=e.display+' · '+e.nameZh;return o;});
 $('entries').replaceChildren(...options);if(entry)$('entries').value=entry.id;
}
$('search').addEventListener('input',list);
$('entries').addEventListener('change',()=>{if(saving){$('entries').value=entry.id;return;}if(dirty&&!confirm('当前内容未保存。放弃这些编辑并切换音标？')){$('entries').value=entry.id;return;}load($('entries').value);});
form.addEventListener('input',()=>{dirty=true;status('当前音标尚未保存。');preview();});
form.addEventListener('submit',async event=>{
 event.preventDefault();if(saving)return;saving=true;const buttons=[...form.querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);$('entries').disabled=true;
 try{const value=content(),before=JSON.stringify(value),id=entry.id,result=await api('PUT',{id,content:value,revision:data.revision});data.revision=result.revision;data.entries[id]=value;dirty=JSON.stringify(content())!==before;status(dirty?'已保存刚才的内容，后续编辑尚未保存。':result.message);}
 catch(error){status(error.message);}finally{saving=false;buttons.forEach(b=>b.disabled=false);$('entries').disabled=false;}
});
$('reset').addEventListener('click',()=>{for(const key of fields)field(key).value=entry[key]??'';for(const key of ['audio','video','animation'])field(key).value='';dirty=true;status('已恢复原介绍，点击保存后生效。');preview();});
$('stop').addEventListener('click',async()=>{if(saving)return;if(dirty&&!confirm('当前内容未保存。关闭维护工具？'))return;try{const result=await api('POST',{},'/__m17_author/stop');dirty=false;status(result.message);document.querySelectorAll('input,textarea,select,button').forEach(e=>e.disabled=true);}catch(error){status(error.message);}});
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
try{data=await api('GET');list();load(data.catalog[0].id);}catch(error){status(error.message);form.querySelectorAll('input,textarea,button').forEach(e=>e.disabled=true);}
