// PCM capture only. Scientific editing and processing remain separate operations.
class PtbRecorder extends AudioWorkletProcessor {
 constructor(options){super();this.limit=options.processorOptions.limit;this.frames=0;this.done=false;this.port.onmessage=e=>{if(e.data==='stop')this.finish();};}
 finish(limit=false){if(this.done)return;this.done=true;this.port.postMessage({done:true,limit});}
 process(inputs){
  const input=inputs[0];if(this.done||!input?.length)return true;
  const length=Math.min(input[0].length,this.limit-this.frames);
  if(length>0){const block=input.map(c=>c.slice(0,length));this.frames+=length;this.port.postMessage({block},block.map(c=>c.buffer));}
  if(this.frames>=this.limit)this.finish(true);return true;
 }
}
registerProcessor('ptb-recorder',PtbRecorder);
