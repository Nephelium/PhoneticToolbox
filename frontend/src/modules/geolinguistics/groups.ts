import {categoryText, featureColor, observationText, type AtlasFeature, type Project} from './model.ts';

export type FeatureDetail='broad'|'fine';
export interface BroadGroup {id:string;label:string;codes:string[];labelKind:'source'|'summary';page:number}
export const unresolvedMapping='@source-conflict';
export const unmappedBroad='@unmapped-broad';
export const pendingBroadSites='@pending-sites';
const groupKey=(id:string)=>'@group:'+id;
export const isBroadCategory=(code:string)=>code.startsWith('@group:');
export function groupFor(code:string,feature:AtlasFeature){return feature.groups?.find(g=>g.codes.includes(code));}
export function hasBroadMeanings(feature:AtlasFeature){return !!feature.groups?.length;}
export function completeBroadMeanings(feature:AtlasFeature){return feature.categories.length>0&&feature.categories.every(c=>!!groupFor(c,feature));}
/** Only explicit reviewed memberships are used. Prefixes and palette order never establish membership. */
export function displayCodes(value:string|null,feature:AtlasFeature,detail:FeatureDetail,siteId?:string):string[]{
 if(value===null||!value.trim())return ['待核'];
 if(detail==='fine')return value.split('#');
 const exception=siteId?feature.groupExceptions?.[siteId]:undefined;
 const conflicts=Array.isArray(exception)?exception:exception?[exception]:[];
 return [...new Set(value.split('#').map(code=>{if(conflicts.some(item=>item.code===code))return unresolvedMapping;const group=groupFor(code,feature);return group?groupKey(group.id):unmappedBroad;}))];
}
export function displayText(code:string,feature:AtlasFeature,project:Project,detail:FeatureDetail,includeCode=false){
 if(code===unresolvedMapping)return '原图对应待核';
 if(code===unmappedBroad)return '此记录的大类尚待对应';
 if(code==='待核')return '原值待核';
 if(detail==='fine')return categoryText(code,feature,project,includeCode);
 const group=feature.groups?.find(g=>groupKey(g.id)===code);
 return group?group.label:'此记录的大类尚待对应';
}
export function displayValue(value:string|null,feature:AtlasFeature,project:Project,detail:FeatureDetail,siteId?:string){
 if(detail==='fine')return observationText(value,feature,project);
 return displayCodes(value,feature,detail,siteId).map(code=>displayText(code,feature,project,detail)).join('；');
}
export function displayColor(code:string,feature:AtlasFeature,detail:FeatureDetail){
 if(detail==='fine')return featureColor(code==='待核'?null:code,feature);
 const group=feature.groups?.find(g=>groupKey(g.id)===code);
 return group?featureColor(group.id,{...feature,categories:feature.groups!.map(g=>g.id)}):'#9099a2';
}
export function displayCounts(records:{value:string|null;siteId?:string}[],feature:AtlasFeature,detail:FeatureDetail){
 const counts=new Map<string,number>();
 for(const record of records)for(const code of new Set(displayCodes(record.value,feature,detail,record.siteId)))counts.set(code,(counts.get(code)??0)+1);
 const order=detail==='broad'?[...(feature.groups??[]).map(g=>groupKey(g.id)),...feature.categories]:feature.categories;
 return [...counts].map(([code,count])=>({code,count})).sort((a,b)=>(order.indexOf(a.code)<0?Infinity:order.indexOf(a.code))-(order.indexOf(b.code)<0?Infinity:order.indexOf(b.code)));
}
export function groupCodes(code:string,feature:AtlasFeature){return feature.groups?.find(g=>groupKey(g.id)===code)?.codes.join('、')??code;}

/** Missing correspondence is a property of a record, never a linguistic category. */
export function broadPendingCount(records:{value:string|null;siteId?:string}[],feature:AtlasFeature){
 return records.filter(record=>displayCodes(record.value,feature,'broad',record.siteId).some(code=>!isBroadCategory(code))).length;
}
export function pointAppearance(value:string|null,feature:AtlasFeature,detail:FeatureDetail,siteId?:string){
 const codes=displayCodes(value,feature,detail,siteId);
 const categories=detail==='broad'?codes.filter(isBroadCategory):codes;
 const incomplete=detail==='broad'&&codes.some(code=>!isBroadCategory(code));
 return {color:displayColor(categories[0]??codes[0],feature,detail),multi:categories.length>1,unclassified:incomplete&&!categories.length,incomplete};
}
