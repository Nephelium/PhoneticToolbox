import {lipMetrics,pixelPoints,type Point,type Metrics} from './metrics.ts';
export interface Detection {points:Point[]|null;metrics:Metrics|null;inference_ms:number;time_ms:number;width:number;height:number}
export class LipInference {
  private worker:Worker;
  private next=0;
  private pending=new Map<number,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  private busy=false;
  constructor(){
    this.worker=new Worker(new URL(`${import.meta.env.BASE_URL}m05/worker.js`,location.href));
    this.worker.onmessage=({data})=>{const item=this.pending.get(data.id);if(!item)return;clearTimeout(item.timer);this.pending.delete(data.id);data.ok?item.resolve(data.value):item.reject(Error(data.error));};
    this.worker.onerror=event=>this.fail(Error(event.message||'Worker 初始化失败'));
  }
  private fail(error:Error){for(const item of this.pending.values()){clearTimeout(item.timer);item.reject(error);}this.pending.clear();}
  private call(op:string,extra:Record<string,unknown>={},transfer:Transferable[]=[]):Promise<any>{
    const id=++this.next;
    return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.pending.delete(id);reject(Error('推理 Worker 超时，请停止后重试'));this.worker.terminate();this.fail(Error('Worker 已停止'));},60000);this.pending.set(id,{resolve,reject,timer});this.worker.postMessage({id,op,...extra},transfer);});
  }
  initialize(delegate:'CPU'|'GPU'='CPU',mode:'IMAGE'|'VIDEO'='VIDEO'){return this.call('init',{delegate,mode});}
  async detect(frame:ImageBitmap,time_ms:number,mode:'IMAGE'|'VIDEO'='VIDEO'):Promise<Detection>{
    if(this.busy){frame.close();throw Error('inference_busy');}
    this.busy=true;const width=frame.width,height=frame.height;
    try{const value=await this.call('frame',{frame,time_ms,mode},[frame]);const points=value.points?pixelPoints(value.points,width,height):null;return {...value,width,height,points,metrics:points?lipMetrics(points):null};}
    finally{this.busy=false;}
  }
  close(){this.worker.terminate();this.fail(Error('推理已停止'));}
}
