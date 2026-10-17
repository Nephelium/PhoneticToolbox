// Synthetic UI/screenshot transport only. Real persistence is tested in PG.
const fields=require('../../contracts/module-preference-fields.json');
const merge=(a,b)=>{const out={...a};for(const [k,v] of Object.entries(b))out[k]=v&&typeof v==='object'&&!Array.isArray(v)&&out[k]&&typeof out[k]==='object'?merge(out[k],v):v;return out;};
exports.createPreferencesFixture=()=>{
 const records=new Map();
 return {seed(owner,module,values){records.set(owner+':'+module,{module,revision:1,values,updated_at:'2026-10-17T00:00:00Z'});},reply(request,owner){
  const path=new URL(request.url()).pathname;if(path!=='/api/v1/preferences'&&!path.startsWith('/api/v1/preferences/'))return null;
  if(path==='/api/v1/preferences')return {status:200,body:{allowed_fields:fields,modules:[...records.entries()].filter(([key])=>key.startsWith(owner+':')).map(([,value])=>value)}};
  const module=path.split('/').at(-1),key=owner+':'+module,old=records.get(key),patch=request.postDataJSON();
  if((old?.revision??0)!==patch.expected_revision)return {status:409,body:{detail:'settings_revision_conflict'}};
  const value={module,revision:(old?.revision??0)+1,values:merge(old?.values??{},patch.changes),updated_at:'2026-10-17T00:00:00Z'};records.set(key,value);return {status:200,body:value};
 }};
};
