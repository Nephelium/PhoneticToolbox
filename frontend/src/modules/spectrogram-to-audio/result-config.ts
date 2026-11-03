import type {ReconstructionConfig} from '../../platform/research.ts';

// Result metadata includes optional stroke defaults which the request omits.
// Compare display state structurally without rewriting either scientific snapshot.
function equalValue(left:unknown,right:unknown):boolean{
 if(left===right)return true;
 if(Array.isArray(left)||Array.isArray(right))return Array.isArray(left)&&Array.isArray(right)&&left.length===right.length&&left.every((value,index)=>equalValue(value,right[index]));
 if(!left||!right||typeof left!=='object'||typeof right!=='object')return false;
 const a=left as Record<string,unknown>,b=right as Record<string,unknown>,keys=Object.keys(a);
 return keys.length===Object.keys(b).length&&keys.every(key=>Object.hasOwn(b,key)&&equalValue(a[key],b[key]));
}
function normalizedStrokes(strokes:ReconstructionConfig['strokes']){
 return (strokes??[]).map(stroke=>({...stroke,size_unit:stroke.size_unit??'legacy',canvas_width:stroke.canvas_width??null,canvas_height:stroke.canvas_height??null,frequency_fraction:stroke.frequency_fraction??null}));
}
export function resultConfigMatches(current:ReconstructionConfig,saved:Partial<ReconstructionConfig>,defaults:ReconstructionConfig):boolean{
 return Object.entries(current).every(([key,value])=>{
  const stored=saved[key as keyof ReconstructionConfig]??defaults[key as keyof ReconstructionConfig];
  return key==='strokes'?equalValue(normalizedStrokes(current.strokes),normalizedStrokes(saved.strokes)):equalValue(value,stored);
 });
}
