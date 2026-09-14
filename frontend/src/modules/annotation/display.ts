// ORIGIN-WEBEDITOR: visible-neighbour padding and original lip display scaling.
export function lipCurve(times:number[],values:(number|null)[],offset:number,start:number,length:number,width:number,height:number){
 if(times.length<2||!values.length)return [];
 const first=Math.max(0,times.findIndex(t=>t>=start-offset)-1);let last=times.length-1;
 for(let i=times.length-1;i>=0;i--)if(times[i]<=start+length-offset){last=Math.min(times.length-1,i+1);break;}
 if(times[last]<start-offset||times[first]>start+length-offset)return [];
 let low=Infinity,high=-Infinity;for(let i=first;i<=last;i++){const v=values[i];if(v!==null&&Number.isFinite(v)){low=Math.min(low,v);high=Math.max(high,v);}}
 if(!Number.isFinite(low))return [];
 const result:({x:number;y:number;move:boolean})[]=[];let move=true;
 for(let i=first;i<=last;i++){const value=values[i];if(value===null||!Number.isFinite(value)){move=true;continue;}result.push({x:(times[i]+offset-start)/length*width,y:height-(value-low)/(high-low||1)*height*.5-4,move});move=false;}
 return result;
}
