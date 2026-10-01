export interface FontPreferences {version:1;zh:string;latin:string;mono:string;ipa:'Doulos SIL';figure:{follow:boolean;zh:string;latin:string;size:number}}
export const candidates={zh:['Microsoft YaHei','SimSun','KaiTi','Source Han Serif SC','Noto Serif CJK SC','PingFang SC','Noto Sans CJK SC','Noto Sans SC','Source Han Sans SC'],latin:['Segoe UI','Times New Roman','Arial','Georgia','DejaVu Sans','Liberation Sans','Noto Sans'],mono:['Consolas','Cascadia Code','JetBrains Mono','Courier New','DejaVu Sans Mono','Liberation Mono','Menlo']};
export const defaults=():FontPreferences=>({version:1,zh:'',latin:'',mono:'',ipa:'Doulos SIL',figure:{follow:true,zh:'',latin:'',size:12}});
export const validFamily=(value:unknown):value is string=>typeof value==='string'&&value.length<=100&&!/[\x00-\x1f"'\\/;{}<>]/.test(value);
export const quoteFamily=(value:string)=>{if(!validFamily(value))throw Error('无效的字体名称。');return '"'+value+'"';};
export function normalizeFonts(value:unknown):FontPreferences{
 const d=defaults();if(!value||typeof value!=='object'||Array.isArray(value))return d;
 const p=value as Record<string,any>;if(p.version!==1)return d;
 for(const key of ['zh','latin','mono'] as const)if(validFamily(p[key]))d[key]=p[key].trim();
 if(p.figure&&typeof p.figure==='object'){
  d.figure.follow=typeof p.figure.follow==='boolean'?p.figure.follow:true;
  for(const key of ['zh','latin'] as const)if(validFamily(p.figure[key]))d.figure[key]=p.figure[key].trim();
  if(typeof p.figure.size==='number'&&Number.isFinite(p.figure.size)&&p.figure.size>=10&&p.figure.size<=24)d.figure.size=p.figure.size;
 }return d;
}
export const fontKey=(owner?:string)=>'fonts.v1.'+(owner?'owner.'+encodeURIComponent(owner):'desktop');
export const figureFonts=(p:FontPreferences)=>({zh:p.figure.follow?p.zh:p.figure.zh,latin:p.figure.follow?p.latin:p.figure.latin,ipa:'Doulos SIL' as const,size:p.figure.size});
