// Display range only. Samples and playback gain are never modified.
export function amplitudeLimit(points:readonly (readonly [number,number])[]):number {
 let peak=0;for(const pair of points)for(const value of pair)if(Number.isFinite(value))peak=Math.max(peak,Math.abs(value));
 return peak>0?peak:1;
}
export function amplitudeLabel(value:number):string {
 if(value===0)return '0';
 return Math.abs(value)<.001||Math.abs(value)>=1000?value.toExponential(2):Number(value.toPrecision(3)).toString();
}
// At detail scale, join original samples at their exact time positions. Keeping
// up to 16 samples/pixel avoids disconnected envelope strokes near the transition.
// Include edge neighbours so fractional view limits do not leave an empty margin.
export function sampleWavePath(samples:Float32Array,start:number,end:number,limit:number,bins:number):string|null {
 if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start||limit<=0)return null;
 const first=Math.max(0,Math.floor(start)),last=Math.min(samples.length-1,Math.ceil(end));
 if(last<first||last-first+1>Math.max(1,Math.min(4096,bins))*16)return null;
 const points:string[]=[];
 for(let i=first;i<=last;i++)points.push(`${i===first?'M':'L'}${(i-start)/(end-start)*1000},${45-samples[i]/limit*37}`);
 return points.join(' ');
}
