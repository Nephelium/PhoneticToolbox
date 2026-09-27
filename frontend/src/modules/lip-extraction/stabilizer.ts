// Lightweight implementation of the frozen V2 filter specification; never modifies raw coordinates.
import {metricSpec,type Point} from './metrics.ts';
const f=Math.fround;
const norm=(x:number,y:number)=>f(Math.sqrt(f(f(x*x)+f(y*y))));
export class LandmarkStabilizer {
  private estimate:Point[]|null=null;
  private previous:Point[]|null=null;
  private derivative:Point[]|null=null;
  private time:number|null=null;
  cutoff:number;
  constructor(cutoff=15){this.cutoff=cutoff;}
  reset(){this.estimate=this.previous=this.derivative=null;this.time=null;}
  filter(input:Point[],time:number):Point[]{
    const points=input.map(p=>[f(p[0]),f(p[1])] as Point);
    if(!this.estimate||!this.previous||!this.derivative||this.time===null){
      this.estimate=points.map(p=>[...p]);this.previous=points.map(p=>[...p]);this.derivative=points.map(()=>[0,0]);this.time=time;return points;
    }
    if(time<=this.time)return this.estimate.map(p=>[...p]);
    const dt=Math.min(time-this.time,metricSpec.filter.max_dt),ad=f(dt/(1/(2*Math.PI*metricSpec.filter.d_cutoff_hz)+dt));
    const derivative=points.map((p,i)=>p.map((v,axis)=>f(f(ad*f(f(v-this.previous![i][axis])/f(dt)))+f(f(1-ad)*this.derivative![i][axis]))) as Point);
    const speeds=derivative.map(p=>norm(p[0],p[1]));
    const averaged=speeds.map((v,i)=>f((v+metricSpec.neighbors[i].reduce((sum,n)=>sum+speeds[n],0))/(metricSpec.neighbors[i].length+1)));
    const x=this.estimate.map(p=>p[0]),y=this.estimate.map(p=>p[1]),scale=norm(f(Math.max(...x)-Math.min(...x)),f(Math.max(...y)-Math.min(...y)));
    const result=points.map((p,i)=>{
      const speed=f(Math.max(f(averaged[i]-f(metricSpec.filter.noise_speed_scale*scale)),0));
      const cutoff=f(f(this.cutoff)+f(f(metricSpec.filter.beta)*speed));
      const tau=f(1/f(f(2*Math.PI)*Math.max(cutoff,1e-6))),alpha=f(f(dt)/f(tau+f(dt)));
      const dx=f(p[0]-this.estimate![i][0]),dy=f(p[1]-this.estimate![i][1]);
      const residual=f(norm(dx,dy)-f(metricSpec.filter.gate_low*scale));
      const gateSpan=f(Math.max((metricSpec.filter.gate_high-metricSpec.filter.gate_low)*scale,1e-6));
      const gate=f(Math.max(0,Math.min(1,f(residual/gateSpan))));
      const gain=f(Math.max(alpha,gate));
      return [f(this.estimate![i][0]+f(gain*dx)),f(this.estimate![i][1]+f(gain*dy))] as Point;
    });
    this.estimate=result;this.previous=points;this.derivative=derivative;this.time=time;
    return result.map(p=>[...p]);
  }
}
