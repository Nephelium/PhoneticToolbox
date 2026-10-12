export interface IconCrop {x:number;y:number;width:number;height:number}
export interface IconBackgroundRemoval {enabled:boolean;color:string;tolerance:number}
/** Remove edge-connected colour only, preserving isolated details of that colour. */
export function removeIconBackground(pixels:Uint8ClampedArray,width:number,height:number,color:[number,number,number],tolerance:number){
 const visited=new Uint8Array(width*height),queue=new Int32Array(width*height);let begin=0,end=0;
 const visit=(index:number)=>{if(visited[index])return;visited[index]=1;const p=index*4;if(pixels[p+3]&&Math.max(...color.map((channel,i)=>Math.abs(channel-pixels[p+i])))>tolerance)return;queue[end++]=index;};
 for(let x=0;x<width;x++){visit(x);visit((height-1)*width+x);}for(let y=0;y<height;y++){visit(y*width);visit(y*width+width-1);}
 while(begin<end){const i=queue[begin++];pixels[i*4+3]=0;if(i%width)visit(i-1);if(i%width<width-1)visit(i+1);if(i>=width)visit(i-width);if(i+width<width*height)visit(i+width);}
 return pixels;
}
export function initialIconCrop(width:number,height:number,ratio:number|null=1):IconCrop {
 const w=ratio?Math.min(width,height*ratio):width,h=ratio?w/ratio:height;
 return {x:(width-w)/2,y:(height-h)/2,width:w,height:h};
}
export function moveIconCrop(crop:IconCrop,dx:number,dy:number,width:number,height:number):IconCrop {
 return {...crop,x:Math.max(0,Math.min(width-crop.width,crop.x+dx)),y:Math.max(0,Math.min(height-crop.height,crop.y+dy))};
}
export function resizeIconCrop(crop:IconCrop,w:number,h:number,width:number,height:number,ratio:number|null):IconCrop {
 const availableWidth=width-crop.x,availableHeight=height-crop.y;
 if(ratio){const dw=w-crop.width,dh=(h-crop.height)*ratio;w=crop.width+(Math.abs(dw)>=Math.abs(dh)?dw:dh);w=Math.max(Math.min(8,availableWidth,availableHeight*ratio),Math.min(w,availableWidth,availableHeight*ratio));h=w/ratio;}
 else{w=Math.max(Math.min(8,availableWidth),Math.min(w,availableWidth));h=Math.max(Math.min(8,availableHeight),Math.min(h,availableHeight));}
 return {...crop,width:w,height:h};
}
export function renderIconCrop(image:HTMLImageElement,crop:IconCrop,canvas:HTMLCanvasElement,removal?:IconBackgroundRemoval){
 canvas.width=canvas.height=256;
 const context=canvas.getContext('2d');if(!context)throw Error('当前环境无法处理图片。');
 const scale=256/Math.max(crop.width,crop.height),w=crop.width*scale,h=crop.height*scale;
 context.clearRect(0,0,256,256);context.drawImage(image,crop.x,crop.y,crop.width,crop.height,(256-w)/2,(256-h)/2,w,h);
 if(removal?.enabled){const data=context.getImageData(0,0,256,256),color=[1,3,5].map(i=>parseInt(removal.color.slice(i,i+2),16)) as [number,number,number];removeIconBackground(data.data,256,256,color,removal.tolerance);context.putImageData(data,0,0);}
 return canvas.toDataURL('image/png');
}
