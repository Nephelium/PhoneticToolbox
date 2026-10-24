import type {BroadGroup,FeatureDetail} from './groups.ts';
export type Volume='P'|'L'|'S';
export type Mode='features'|'classification'|'paper'|'custom';
export interface GroupException {code:string;reason:string}
export interface AtlasFeature {id:string;volume:Volume;number:number;page:number;title:string;titleReviewed:boolean;categories:string[];labels:Record<string,string>;legend:string;categoryNote?:string;categoryReview?:{verified:number;total:number};groups?:BroadGroup[];groupReview?:{verifiedCodes:number;totalCodes:number;verifiedGroups:number;totalGroups:number;complete:boolean};groupExceptions?:Record<string,GroupException|GroupException[]>;groupNote?:string}
export interface Catalog {version:string;features:AtlasFeature[];survey:string;sources:{id:string;title:string;url?:string;license:string}[]}
export interface Site {id:string;name:string;lon:number;lat:number;admin:string|null;adminCandidates?:string[]}
export interface Observation {values:(string|null)[];raw:Record<string,{raw:string;rule:string}>}
export interface Admin {id:string;code:string;name:string;level:'province'|'city'|'district';parent:string;province:string;center:[number,number];extent:[number,number,number,number];source:string}
export interface ClassificationRow {id:string;sheet:string;row:number;values:string[];original:unknown[];admin:string|null;match:string|null}
export interface Classifications {author:string;book:string;columns:string[];sheets:{id:string;name:string;rows:ClassificationRow[]}[]}
export interface PaperPoint {mds:[number,number,number];sharp:string[];heat:string[]}
export interface PaperData {points:PaperPoint[];matrices:string[];colors:Record<string,Record<string,Record<string,string>>>}
export interface Resources {catalog:Catalog;sites:Site[];observations:Record<string,Observation>;classification:Classifications;admin:Admin[];paper:PaperData}
export interface MapPoint {id:string;name:string;lon:number;lat:number;color:string;label:string;multi?:boolean;selected?:boolean;unclassified?:boolean;incomplete?:boolean}
export interface ViewState {center:[number,number];zoom:number}
/** The retired world button saved this fixed view; ordinary pans remain untouched. */
export function isLegacyWorldView(view:ViewState){return Math.abs(view.center[0]-10)<1e-6&&Math.abs(view.center[1]-20)<1e-6&&Math.abs(view.zoom-2)<1e-6;}
export interface ProjectSettings {volume:Volume;query:string;locationQuery:string;region:string;category:string;sheet:string;classFilters:string[];classDepth:number;classSearch:string;level:'auto'|'province'|'city'|'district';labels:boolean;mapCount:number;extraFeatures:string[];paperMode:string;clusters:number;matrixName:string;onlySelected:boolean;onlyViewport:boolean;tableOpen:boolean;selectedSite:string;selectedClass:string}
export function defaultSettings():ProjectSettings{return {volume:'P',query:'',locationQuery:'',region:'',category:'',sheet:'combined',classFilters:['','','',''],classDepth:3,classSearch:'',level:'auto',labels:false,mapCount:1,extraFeatures:['F117_P','F68_L','F191_L'],paperMode:'rgb',clusters:5,matrixName:'main',onlySelected:false,onlyViewport:false,tableOpen:true,selectedSite:'',selectedClass:''};}
export interface Project {schema:'ptb-geolinguistics/1';dataset:string;name:string;edits:Record<string,string>;classEdits:Record<string,string[]>;labels:Record<string,Record<string,string>>;customSites:(Site&{value:string;note:string})[];drawings:unknown;selected:string[];view:ViewState;feature:string;mode:Mode;settings?:ProjectSettings;atlasView?:'original'|'split'|'interactive';bookmarks?:string[];featureScope?:'reviewed'|'all';featureDetail?:FeatureDetail}
export function emptyProject():Project{return {schema:'ptb-geolinguistics/1',dataset:'m19-local-20261022',name:'语言地图研究',edits:{},classEdits:{},labels:{},customSites:[],drawings:{type:'FeatureCollection',features:[]},selected:[],view:{center:[105,33],zoom:4},feature:'F1_P',mode:'classification',bookmarks:[],featureDetail:'broad'};}
export const volumeNames:Record<Volume,string>={P:'语音',L:'词汇',S:'语法'};
const palette=['#17678b','#c44c35','#69902c','#91639c','#d19617','#248e85','#cc6f9a','#6174b4','#9a713b','#607d81','#be523d','#566f29'];
export function categoryColor(value:string){let hash=2166136261;for(const ch of value)hash=Math.imul(hash^ch.charCodeAt(0),16777619);return palette[(hash>>>0)%palette.length];}
export function featureColor(value:string|null,feature:AtlasFeature){if(!value)return '#9099a2';const i=feature.categories.indexOf(value.split('#')[0]);return palette[(Math.max(0,i))%palette.length];}
export function observation(data:Resources,project:Project,feature:string,index:number){const key=data.sites[index].id+':'+feature;const edited=project.edits[key];return edited!==undefined?edited:data.observations[feature]?.values[index]??null;}
export function isObservationEdited(data:Resources,project:Project,feature:string,index:number){const key=data.sites[index].id+':'+feature;return Object.hasOwn(project.edits,key)&&project.edits[key]!==data.observations[feature]?.values[index];}
export function classValues(row:ClassificationRow,project:Project){return project.classEdits[row.id]??row.values;}
export function classPath(values:string[],level=4){return values.slice(6,6+level).map(v=>v||'未分').join(' / ');}
/** Empty string remains the historical wildcard; this value selects an empty classification layer. */
export const unclassifiedClassFilter='@empty-classification-layer';
export function classFilterMatches(filter:string,value:string){return filter===unclassifiedClassFilter?!value:!filter||value===filter;}
export function inRegion(adminId:string|null,region:string,index:Map<string,Admin>){if(!region)return true;if(!adminId)return false;let cursor=index.get(adminId);const seen=new Set<string>();while(cursor&&!seen.has(cursor.id)){if(cursor.id===region)return true;seen.add(cursor.id);cursor=index.get(cursor.parent);}return false;}
/** Keep a saved historical selection intact while displaying an available scope. */
export function regionSelection(region:string,index:ReadonlyMap<string,Admin>){const missing=!!region&&!index.has(region);return {effective:missing?'':region,missing};}
/** Follow actual parents: a province may directly contain source-city or district nodes. */
export function regionPath(region:string,index:ReadonlyMap<string,Admin>){const path:Admin[]=[],seen=new Set<string>();let current=index.get(region);while(current&&!seen.has(current.id)){path.unshift(current);if(current.level==='province')return path;seen.add(current.id);current=index.get(current.parent);}return [];}
export function diversity(records:ClassificationRow[],project:Project,level:number){const paths=new Map<string,ClassificationRow[]>();for(const row of records){const v=classValues(row,project);if(!v[5+level])continue;const key=classPath(v,level);paths.set(key,[...(paths.get(key)??[]),row]);}return [...paths].map(([path,rows])=>({path,rows})).sort((a,b)=>a.path.localeCompare(b.path,'zh'));}
export function categoryCounts(values:(string|null)[]){const counts=new Map<string,number>();for(const value of values)for(const code of new Set(value?.split('#')??['待核']))counts.set(code,(counts.get(code)??0)+1);return [...counts].map(([code,count])=>({code,count}));}
/** Codes are feature-local. Personal corrections take priority without changing the source dictionary. */
export function categoryMeaning(code:string,feature:AtlasFeature,project:Project){
 const personal=project.labels[feature.id]?.[code];
 return personal?.trim()?personal:feature.labels[code]??'';
}
export function categoryText(code:string,feature:AtlasFeature,project:Project,includeCode=false){
 if(code==='待核')return '原值待核';
 const label=categoryMeaning(code,feature,project);
 return label?(includeCode?`${label}（${code}）`:label):`${code} · 释义待核`;
}
/** Preserve the author's ordered # variants; slashes inside IPA labels are not variant separators. */
export function observationText(value:string|null,feature:AtlasFeature,project:Project,includeCode=false){
 if(value===null)return '原值待核';
 const parts=value.split('#').map(c=>categoryText(c,feature,project,includeCode));
 return parts.length===1?parts[0]:parts.map((v,i)=>`${i+1}. ${v}`).join('；');
}
/** Only the reviewed source dictionary controls the default phenomenon collection. */
export function hasCompleteMeanings(feature:AtlasFeature){return feature.categories.length>0&&feature.categories.every(code=>!!feature.labels[code]?.trim());}
export type ComparisonStatus='same'|'different'|'missing';
export interface FeatureComparison {feature:AtlasFeature;values:(string|null)[];status:ComparisonStatus}
/** Compare literal recorded categories, retaining ordered variants and missingness. This is not the paper distance. */
export function compareFeatures(data:Resources,project:Project,indices:number[]):FeatureComparison[]{
 if(indices.length<2)return [];
 return data.catalog.features.map(feature=>{const values=indices.map(i=>observation(data,project,feature.id,i));const status=values.some(v=>v===null||v.trim()==='')?'missing':new Set(values).size===1?'same':'different';return {feature,values,status};});
}
export function validateProject(input:unknown):Project{
 if(!input||typeof input!=='object')throw Error('工程文件不是有效对象。');const p=input as Project;
 if(p.schema!=='ptb-geolinguistics/1'||p.dataset!=='m19-local-20261022')throw Error('工程版本或底层数据版本不匹配。');
 if(p.featureScope!==undefined&&!['reviewed','all'].includes(p.featureScope))throw Error('语言现象范围无效。');
 if(p.featureDetail!==undefined&&!['broad','fine'].includes(p.featureDetail))throw Error('分类层级无效。');
 if(p.atlasView!==undefined&&!['original','split','interactive'].includes(p.atlasView))throw Error('原书显示方式无效。');
 if(p.bookmarks!==undefined&&(!Array.isArray(p.bookmarks)||p.bookmarks.length>510||p.bookmarks.some(id=>typeof id!=='string'||!/^F\d+_[PLS]$/.test(id))||new Set(p.bookmarks).size!==p.bookmarks.length))throw Error('地图收藏记录无效。');
 if(typeof p.name!=='string'||!Array.isArray(p.selected)||p.selected.some(v=>typeof v!=='string')||!Array.isArray(p.customSites)||!['features','classification','paper','custom'].includes(p.mode)||typeof p.feature!=='string')throw Error('工程内容不完整。');
 if(!p.view||!Array.isArray(p.view.center)||p.view.center.length!==2||p.view.center.some(v=>!Number.isFinite(v))||!Number.isFinite(p.view.zoom)||p.view.zoom<1||p.view.zoom>18)throw Error('地图视野无效。');
 for(const field of ['edits','classEdits','labels'] as const)if(!p[field]||typeof p[field]!=='object'||Array.isArray(p[field]))throw Error('工程编辑记录无效。');
 for(const [key,value] of Object.entries(p.edits))if(typeof value!=='string'||value.length>2000||!/^S\d+:F\d+_[PLS]$/.test(key))throw Error('特征编辑记录无效。');
 for(const value of Object.values(p.classEdits))if(!Array.isArray(value)||value.length!==10||value.some(v=>typeof v!=='string'))throw Error('分类编辑记录无效。');
 for(const [id,labels] of Object.entries(p.labels))if(!/^F\d+_[PLS]$/.test(id)||!labels||Array.isArray(labels)||typeof labels!=='object'||Object.values(labels).some(v=>typeof v!=='string'||v.length>2000))throw Error('类别释义无效。');
 for(const point of p.customSites)if(typeof point.id!=='string'||typeof point.name!=='string'||typeof point.value!=='string'||typeof point.note!=='string'||!Number.isFinite(point.lon)||!Number.isFinite(point.lat)||Math.abs(point.lon)>180||Math.abs(point.lat)>85)throw Error('自建调查点的坐标或字段无效。');
 const drawings=p.drawings as {type?:string;features?:unknown[]};if(drawings?.type!=='FeatureCollection'||!Array.isArray(drawings.features)||drawings.features.length>5000)throw Error('绘图内容无效。');
 let vertices=0;const line=(v:unknown):boolean=>Array.isArray(v)&&v.length>=2&&v.every(c=>Array.isArray(c)&&c.length===2&&c.every(Number.isFinite)&&Math.abs(c[0])<=180&&Math.abs(c[1])<=85&&++vertices<=100000);
 for(const item of drawings.features){const f=item as {type?:string;geometry?:{type?:string;coordinates?:unknown}};const g=f?.geometry;if(f?.type!=='Feature'||!g||(g.type==='LineString'?!line(g.coordinates):g.type==='Polygon'?!Array.isArray(g.coordinates)||!g.coordinates.length||!g.coordinates.every(r=>line(r)&&r.length>=4&&r[0][0]===r.at(-1)[0]&&r[0][1]===r.at(-1)[1]):true))throw Error('绘图坐标或形状无效。');}
 if(p.settings){const s=p.settings,defaults=defaultSettings();for(const key of Object.keys(defaults) as (keyof ProjectSettings)[])if(Array.isArray(defaults[key])?!Array.isArray(s[key])||((s[key] as unknown[]).some(v=>typeof v!=='string')):typeof s[key]!==typeof defaults[key])throw Error('工程筛选设置无效。');
  if(!['P','L','S'].includes(s.volume)||!['combined','areas','islands'].includes(s.sheet)||!['auto','province','city','district'].includes(s.level)||![1,2,4].includes(s.mapCount)||![1,2,3,4].includes(s.classDepth)||!Number.isInteger(s.clusters)||s.clusters<2||s.clusters>10||s.classFilters.length!==4||s.extraFeatures.length!==3||!['rgb','d1','d2','d3','sharp','heat'].includes(s.paperMode)||!['main','type-0-0','type-0-1','type-1-0','type-1-1'].includes(s.matrixName))throw Error('工程筛选设置超出范围。');}
 return {...structuredClone(p),featureDetail:p.featureDetail??'fine'};
}
export function csvText(rows:unknown[][]){return '\uFEFF'+rows.map(row=>row.map(v=>'"'+String(v??'').replace(/^[=+@\-\t\r]/,"'$&").replaceAll('"','""')+'"').join(',')).join('\r\n');}
export function nearest(matrix:Float64Array,index:number,count:number){const n=Math.sqrt(matrix.length);if(!Number.isInteger(n)||index<0||index>=n)throw Error('距离矩阵索引无效。');return Array.from({length:n},(_,i)=>({index:i,distance:matrix[index*n+i]})).filter(v=>v.index!==index).sort((a,b)=>a.distance-b.distance).slice(0,count);}
