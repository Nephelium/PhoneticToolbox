import type { AudioAsset } from './types.ts';
export function parseWav(buffer:ArrayBuffer, name:string):AudioAsset {
  const v = new DataView(buffer);
  const str=(offset:number,n:number)=>Array.from({length:n},(_,i)=>String.fromCharCode(v.getUint8(offset+i))).join('');
  if (v.byteLength<44 || str(0,4)!=='RIFF' || str(8,4)!=='WAVE') throw Error('文件不是有效的 RIFF WAV，请重新选择 WAV 音频。');
  if (v.getUint32(4,true)+8>v.byteLength) throw Error('WAV 数据不完整，请检查文件是否复制完成。');
  let fmt=0, data=0, length=0, fmtLength=0;
  for(let p=12;p+8<=v.byteLength;) {
    const n=v.getUint32(p+4,true); if(p+8+n>v.byteLength) throw Error('WAV 数据块超出文件范围。');
    if(str(p,4)==='fmt ') { fmt=p+8; fmtLength=n; }
    if(str(p,4)==='data') { data=p+8; length=n; }
    p+=8+n+(n%2);
  }
  if(!fmt || fmtLength<16 || !data || !length) throw Error('WAV 没有可预览的采样点。');
  let format=v.getUint16(fmt,true);
  if(format===65534 && fmtLength>=40) format=v.getUint16(fmt+24,true);
  const count=v.getUint16(fmt+2,true), sampleRate=v.getUint32(fmt+4,true), block=v.getUint16(fmt+12,true), bits=v.getUint16(fmt+14,true);
  if(count<1 || count>32 || sampleRate<1 || block!==count*bits/8 || length%block) throw Error('WAV 音频头或帧长度不受支持。');
  if(!((format===1 && [8,16,24,32].includes(bits)) || (format===3 && [32,64].includes(bits)))) throw Error('目前支持 PCM 8/16/24/32 位和 FLOAT 32/64 位 WAV。');
  const frames=length/block, channels=Array.from({length:count},()=>new Float32Array(frames));
  for(let i=0;i<frames;i++) for(let c=0;c<count;c++) {
    const p=data+i*block+c*bits/8;
    let value=0;
    if(format===3) value=bits===32?v.getFloat32(p,true):v.getFloat64(p,true);
    else if(bits===8) value=(v.getUint8(p)-128)/128;
    else if(bits===16) value=v.getInt16(p,true)/32768;
    else if(bits===32) value=v.getInt32(p,true)/2147483648;
    else { let n=v.getUint8(p)|(v.getUint8(p+1)<<8)|(v.getUint8(p+2)<<16);if(n&0x800000)n-=0x1000000;value=n/8388608; }
    if(!Number.isFinite(value)) throw Error('音频包含非有限采样值，无法安全预览。');
    channels[c][i]=value;
  }
  return {name,sampleRate,frames,channels,duration:frames/sampleRate};
}
export function selection(start:number,end:number,duration:number):[number,number] {
  const a=Math.min(duration,Math.max(0,Number.isFinite(start)?start:0));
  const b=Math.min(duration,Math.max(0,Number.isFinite(end)?end:0));
  return [Math.min(a,b),Math.max(a,b)];
}
export function envelope(samples:Float32Array, start:number,end:number,bins=800):[number,number][] {
  const lo=Math.max(0,Math.floor(start)), hi=Math.min(samples.length,Math.ceil(end));
  const step=Math.max(1,Math.ceil((hi-lo)/bins)); const points:[number,number][]=[];
  for(let i=lo;i<hi;i+=step) { let min=Infinity,max=-Infinity;for(let j=i;j<Math.min(hi,i+step);j++){min=Math.min(min,samples[j]);max=Math.max(max,samples[j]);}points.push([min,max]); }
  return points;
}
