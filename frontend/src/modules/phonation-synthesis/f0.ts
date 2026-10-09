export interface F0Track {name:string;axis:number[];values:number[];color:string;dash?:string}
export const stepColor=(index:number)=>`hsl(${(index*137.508+25)%360} 65% 45%)`;

// Published groups concatenate equally sized PCM columns in step order.
export function playingStep(position:number,sampleRate:number,frames:number,steps:number){
 if(!Number.isFinite(position)||position<0||!Number.isFinite(sampleRate)||sampleRate<=0||!Number.isInteger(frames)||frames<=0||!Number.isInteger(steps)||steps<2||frames%steps!==0)return '';
 const frame=Math.floor(position*sampleRate+1e-7);
 if(frame>=frames)return '';
 return 'step'+String(Math.floor(frame/(frames/steps))+1).padStart(2,'0');
}
