// Fit the reconstructed lateral tissue envelope to the native dorsal tongue.
// Midline contact coordinates remain fixed except for exported rounding error.
const cross=(a,b,c)=>(b[0]-a[0])*(c[2]-a[2])-(b[2]-a[2])*(c[0]-a[0]);
const bary=(p,t)=>{const d=cross(...t);if(Math.abs(d)<1e-12)return null;const a=cross(p,t[1],t[2])/d,b=cross(t[0],p,t[2])/d,c=1-a-b;return Math.min(a,b,c)>=-1e-8?[a,b,c]:null;};
const at=(v,i)=>v.slice(i*3,i*3+3);
const bounds=t=>[Math.min(...t.map(p=>p[0])),Math.max(...t.map(p=>p[0])),Math.min(...t.map(p=>p[2])),Math.max(...t.map(p=>p[2]))];
export function fitSoftTissue(vertices,indices,tongue,locked=()=>false){
  const native=[];
  for(let i=0;i<tongue.triangles.length;i+=3){const ids=tongue.triangles.slice(i,i+3);if(ids.some(j=>j>=34*tongue.points))continue;const t=ids.map(j=>at(tongue.vertices,j));native.push({t,box:bounds(t)});}
  for(let f=0;f<indices.length;f+=3){
    const ids=indices.slice(f,f+3),a=ids.map(i=>at(vertices,i)),box=bounds(a),free=a.map((p,k)=>Math.abs(p[2])>1e-8&&!locked(ids[k]));let lift=0,rounding=0;
    const check=(w,y)=>{
      const current=w.reduce((s,v,k)=>s+v*a[k][1],0),gap=y-current;
      if(gap<=1e-7)return;
      const movable=w.reduce((s,v,k)=>s+(free[k]?Math.max(0,v):0),0);
      if(movable>1e-7)lift=Math.max(lift,gap/movable);
      else {
        if(gap>2e-5)throw Error('Rigid uvula contact does not match native surface');
        rounding=Math.max(rounding,gap);
      }
    };
    for(const {t:b,box:q} of native){
      if(box[1]<q[0]-1e-8||q[1]<box[0]-1e-8||box[3]<q[2]-1e-8||q[3]<box[2]-1e-8)continue;
      for(let k=0;k<3;k++){
        const w=bary(a[k],b);if(w)check([0,1,2].map(j=>j===k?1:0),w.reduce((s,v,j)=>s+v*b[j][1],0));
        const u=bary(b[k],a);if(u)check(u,b[k][1]);
      }
      for(let i=0;i<3;i++)for(let j=0;j<3;j++){
        const p=a[i],q=a[(i+1)%3],r=b[j],s=b[(j+1)%3],dx=q[0]-p[0],dz=q[2]-p[2],ex=s[0]-r[0],ez=s[2]-r[2],den=dx*ez-dz*ex;
        if(Math.abs(den)<1e-12)continue;
        const u=((r[0]-p[0])*ez-(r[2]-p[2])*ex)/den,v=((r[0]-p[0])*dz-(r[2]-p[2])*dx)/den;
        if(u< -1e-8||u>1+1e-8||v< -1e-8||v>1+1e-8)continue;
        const w=[0,0,0];w[i]=1-u;w[(i+1)%3]=u;check(w,r[1]+v*(s[1]-r[1]));
      }
    }
    // Raising neighboring faces cannot introduce a new penetration. A single
    // monotone pass satisfies all previously fitted contact half-spaces too.
    for(let k=0;k<3;k++)vertices[ids[k]*3+1]+=rounding+(free[k]?lift:0);
  }
}
