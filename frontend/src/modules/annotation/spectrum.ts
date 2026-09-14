import {hann,fft} from './fft.mjs';
// Same V2 column grid, Hann/1024 FFT and 3.2 log-amplitude display range.
export function spectrum(data:Float32Array,sr:number,start:number,length:number,cols:number,height:number,dark:boolean,points=cols){
 const size=1024,maxBin=Math.floor(Math.min(5000,sr/2)/sr*size),window=hann(size),spectra:Float64Array[]=[];
 let maximum=-Infinity;const rgba=new Uint8ClampedArray(cols*height*4),rms=new Float64Array(points);let maxRms=1e-9;
 const frame=Math.max(1,Math.floor(.03*sr));
 for(let col=0;col<cols;col++){
  const center=Math.floor((start+col/Math.max(1,cols-1)*length)*sr),re=new Float64Array(size),im=new Float64Array(size);
  for(let i=0;i<size;i++)re[i]=(data[center-Math.floor(size/2)+i]||0)*window[i];fft(re,im);
  const mags=new Float64Array(maxBin+1);for(let bin=1;bin<=maxBin;bin++){mags[bin]=Math.log10(Math.hypot(re[bin],im[bin])+1e-8);maximum=Math.max(maximum,mags[bin]);}spectra.push(mags);
 }
 for(let x=0;x<points;x++){const center=Math.floor((start+x/Math.max(1,points-1)*length)*sr);let sum=0;for(let i=0;i<frame;i++){const v=data[center-Math.floor(frame/2)+i]||0;sum+=v*v;}rms[x]=Math.sqrt(sum/frame);maxRms=Math.max(maxRms,rms[x]);}
 const floor=maximum-3.2;
 for(let col=0;col<cols;col++)for(let y=0;y<height;y++){
  const bin=Math.max(1,Math.min(maxBin,Math.floor((1-y/height)*maxBin))),norm=Math.min(1,Math.max(0,(spectra[col][bin]-floor)/(maximum-floor||1)));
  const shade=dark?25+Math.floor(norm*190):255-Math.floor(norm*220),index=(y*cols+col)*4;rgba[index]=rgba[index+1]=rgba[index+2]=shade;rgba[index+3]=255;
 }
 return {rgba,intensity:[...rms].map(r=>Math.min(1,Math.max(0,((100+20*Math.log10(Math.max(r,1e-10)/maxRms))-40)/60)))};
}
