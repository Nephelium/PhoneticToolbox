// Live streams may report mediaTime=0 for every new frame (video-rvfc spec).
// Keep valid media PTS unchanged; observations are only an estimated fallback.
export class LiveFrameClock {
  estimated=false;
  private previousFrame:number|null=null;
  private previousTime:number|null=null;
  private previousNow:number|null=null;
  private estimateOrigin:{time:number;now:number}|null=null;

  next(now:number,mediaTime:number,presentedFrames:number):number|null {
    const hasFrame=Number.isFinite(presentedFrames);
    if(hasFrame&&this.previousFrame!==null&&presentedFrames<=this.previousFrame)return null;
    const validMedia=Number.isFinite(mediaTime)&&mediaTime>=0;
    let time:number;
    if(this.previousTime===null){
      time=validMedia?mediaTime:0;
      this.estimated=!validMedia;
      if(this.estimated){if(!Number.isFinite(now))return null;this.estimateOrigin={time,now};}
    }else if(!this.estimated&&validMedia&&mediaTime>this.previousTime){
      time=mediaTime;
    }else{
      if(!Number.isFinite(now)||this.previousNow===null||now<=this.previousNow)return null;
      this.estimateOrigin??={time:this.previousTime,now:this.previousNow};
      time=this.estimateOrigin.time+(now-this.estimateOrigin.now)/1000;
      if(time<=this.previousTime)return null;
      this.estimated=true;
    }
    if(hasFrame)this.previousFrame=presentedFrames;
    this.previousNow=Number.isFinite(now)?now:null;
    this.previousTime=time;
    return time;
  }
}
