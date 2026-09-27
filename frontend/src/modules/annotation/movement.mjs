// R1: translate selected syllables and their phones without scaling durations.
export function translateWords(words, phones, indices, delta, end, fillGaps) {
  if(!Number.isFinite(delta))throw Error('标注微调步长无效。');
  const selected=new Set(indices),moving=words.filter((w,i)=>selected.has(i)&&w.text.trim());
  if(!moving.length)throw Error('请先在音节层选中需要移动的标注。');
  const moved=moving.map(w=>({...w,xmin:w.xmin+delta,xmax:w.xmax+delta}));
  if(moved.some(w=>w.xmin<0||w.xmax>end))throw Error('移动会超出录音起止范围，标注保持原位。');
  const kept=words.filter((w,i)=>w.text.trim()&&!selected.has(i));
  if(moved.some(w=>kept.some(k=>w.xmin<k.xmax&&w.xmax>k.xmin)))throw Error('移动会与未选中的音节重叠，标注保持原位。');
  const windows=[];
  for(const w of [...moving].sort((a,b)=>a.xmin-b.xmin)){
    const last=windows.at(-1);
    if(last&&w.xmin<=last[1]+1e-6)last[1]=Math.max(last[1],w.xmax);else windows.push([w.xmin,w.xmax]);
  }
  const movingPhones=[],keptPhones=[];
  for(const phone of phones){
    const intersect=windows.filter(([a,b])=>phone.xmin<b&&phone.xmax>a);
    if(!intersect.length){if(phone.text.trim())keptPhones.push({...phone});continue;}
    if(phone.text.trim()){
      if(!intersect.some(([a,b])=>phone.xmin>=a-1e-6&&phone.xmax<=b+1e-6))throw Error('音素跨越了所选音节的外边界，请扩大选择或先对齐边界。');
      movingPhones.push({...phone,xmin:phone.xmin+delta,xmax:phone.xmax+delta});
    }else for(const [a,b] of intersect)movingPhones.push({xmin:Math.max(a,phone.xmin)+delta,xmax:Math.min(b,phone.xmax)+delta,text:''});
  }
  if(movingPhones.some(p=>p.xmin<0||p.xmax>end||keptPhones.some(k=>p.xmin<k.xmax&&p.xmax>k.xmin)))throw Error('移动会与未选中的音素重叠或越界，标注保持原位。');
  const nextWords=fillGaps([...kept,...moved],end),nextPhones=fillGaps([...keptPhones,...movingPhones],end);
  return {words:nextWords,phones:nextPhones,indices:moved.map(w=>nextWords.findIndex(v=>v.text===w.text&&v.xmin===w.xmin&&v.xmax===w.xmax))};
}
