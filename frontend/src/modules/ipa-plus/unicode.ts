export const codePoints=(text:string)=>Array.from(text,c=>'U+'+c.codePointAt(0)!.toString(16).toUpperCase().padStart(4,'0'));
const segmenter=new Intl.Segmenter('und',{granularity:'grapheme'});
export function graphemes(text:string){return Array.from(segmenter.segment(text),s=>({text:s.segment,index:s.index}));}
export function safeSelection(text:string,start:number,end:number){
  start=Math.max(0,Math.min(text.length,Number.isFinite(start)?Math.trunc(start):text.length));
  end=Math.max(start,Math.min(text.length,Number.isFinite(end)?Math.trunc(end):start));
  const boundaries=[0,...graphemes(text).map(s=>s.index+s.text.length)];
  if(start===end){const after=boundaries.find(x=>x>=start)??text.length;return {start:after,end:after};}
  return {start:boundaries.filter(x=>x<=start).at(-1)??0,end:boundaries.find(x=>x>=end)??text.length};
}
export function suspiciousCharacters(text:string){
  return Array.from(text).flatMap((c,i)=>{const n=c.codePointAt(0)!;return (n<32&&c!=='\n'&&c!=='\t')||(n>=0x7F&&n<=0x9F)||(n>=0xE000&&n<=0xF8FF)||/[\u202A-\u202E\u2066-\u2069]/u.test(c)?[{character:c,code:codePoints(c)[0],position:i+1}]:[];});
}
