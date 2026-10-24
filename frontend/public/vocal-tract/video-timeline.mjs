// Final commanded F0, already composed by the shared scientific sampler.
export function timelinePitch(points,seconds){
  if(!points?.length)return null;
  let lo=0,hi=points.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(points[mid].time<=seconds)lo=mid+1;else hi=mid;}
  const a=points[Math.max(0,lo-1)],b=points[Math.min(points.length-1,lo)];
  if(a.f0===null||b.f0===null)return a.f0;
  const u=a===b?0:Math.max(0,Math.min(1,(seconds-a.time)/(b.time-a.time)));
  return a.f0+(b.f0-a.f0)*u;
}
