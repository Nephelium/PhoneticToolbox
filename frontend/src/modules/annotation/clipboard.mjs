// M12-R6: clipboard carries actual intervals, never estimated durations or incremented tones.
const round=value=>Number(value.toFixed(6));
export function replaceWindow(intervals,start,end,pieces){
 const kept=[];
 for(const item of intervals){
  if(item.xmax<=start||item.xmin>=end)kept.push({...item});
  else{if(item.xmin<start)kept.push({...item,xmax:start});if(item.xmax>end)kept.push({...item,xmin:end});}
 }
 return [...kept,...pieces.map(i=>({...i}))].sort((a,b)=>a.xmin-b.xmin);
}
export function eraseWindows(intervals,windows){
 let result=intervals.map(i=>({...i}));
 for(const [start,end] of windows){
  result=replaceWindow(result,start,end,[{xmin:start,xmax:end,text:''}]);
  let index=result.findIndex(i=>i.xmin===start&&i.xmax===end&&!i.text);
  if(index>0&&!result[index-1].text.trim()&&result[index-1].xmax===start){result[index-1].xmax=end;result.splice(index,1);index--;}
  if(index+1<result.length&&!result[index+1].text.trim()&&result[index+1].xmin===result[index].xmax){result[index].xmax=result[index+1].xmax;result.splice(index+1,1);}
 }
 return result;
}
export function captureAnnotation(intervals,phones,indices,role){
 const selected=[...new Set(indices)].sort((a,b)=>a-b).map(i=>intervals[i]);
 if(!selected.length||selected.some(i=>!i))throw Error('请先选择需要复制、剪切或删除的标注。');
 const start=selected[0].xmin,end=selected.at(-1).xmax,windows=selected.map(i=>[i.xmin,i.xmax]);
 const joined=[];for(const [a,b] of windows){const prev=joined.at(-1);if(prev&&prev[1]===a)prev[1]=b;else joined.push([a,b]);}
 const inside=[];
 if(role==='word'&&phones)for(const phone of phones){
  const intersect=joined.filter(([a,b])=>phone.xmin<b&&phone.xmax>a);
  if(phone.text.trim()&&intersect.length&&!intersect.some(([a,b])=>phone.xmin>=a&&phone.xmax<=b))throw Error('音素跨越了所选音节的外边界，请先对齐边界或扩大选择。');
  for(const [a,b] of intersect)inside.push({...phone,xmin:Math.max(a,phone.xmin),xmax:Math.min(b,phone.xmax)});
 }
 const relative=items=>items.map(i=>({...i,xmin:round(i.xmin-start),xmax:round(i.xmax-start)}));
 return {role,span:round(end-start),intervals:relative(selected),phones:role==='word'&&phones?relative(inside):null,windows};
}
export function pasteIntervals(intervals,pieces,start,span,duration){
 if(!Number.isFinite(start)||start<0||start+span>duration+1e-9)throw Error('粘贴会超出录音范围，标注保持不变。');
 start=round(start);const end=round(start+span);
 if(intervals.some(i=>i.text.trim()&&i.xmin<end&&i.xmax>start))throw Error('粘贴范围与已有标注重叠，请选择足够长的空白位置。');
 const inserted=[];let cursor=start;
 for(const item of pieces){const next={...item,xmin:round(start+item.xmin),xmax:round(start+item.xmax)};if(next.xmin>cursor)inserted.push({xmin:cursor,xmax:next.xmin,text:''});if(next.xmax<=next.xmin)throw Error('粘贴区间小于文件时间精度。');inserted.push(next);cursor=next.xmax;}
 if(cursor<end)inserted.push({xmin:cursor,xmax:end,text:''});
 return replaceWindow(intervals,start,end,inserted);
}
