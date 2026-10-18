/** Own two-finger model gestures without turning them into parameter drags. */
export function installModelPinch(surface,{begin,move,end=()=>{},propagateEnd=false}){
 const points=new Map();let blocked=false,initialDistance=1;
 const geometry=()=>{const [a,b]=[...points.values()];return {clientX:(a.x+b.x)/2,clientY:(a.y+b.y)/2,distance:Math.hypot(a.x-b.x,a.y-b.y)};};
 const suppress=event=>{event.preventDefault();event.stopImmediatePropagation();};
 function down(event){
  if(event.pointerType!=='touch')return;
  points.set(event.pointerId,{x:event.clientX,y:event.clientY});
  if(points.size===2&&!blocked){blocked=true;const g=geometry();initialDistance=Math.max(1,g.distance);begin(g);}
  if(blocked){surface.setPointerCapture(event.pointerId);suppress(event);}
 }
 function update(event){
  if(!points.has(event.pointerId))return;
  points.set(event.pointerId,{x:event.clientX,y:event.clientY});
  if(blocked){if(points.size>=2){const g=geometry();move({...g,scale:g.distance/initialDistance});}suppress(event);}
 }
 function finish(event){
  if(!points.has(event.pointerId))return;
  points.delete(event.pointerId);
  if(blocked){if(propagateEnd)event.preventDefault();else suppress(event);if(!points.size){blocked=false;end();}}
 }
 function reset(){points.clear();if(blocked){blocked=false;end();}}
 for(const [name,handler] of [['pointerdown',down],['pointermove',update],['pointerup',finish],['pointercancel',finish],['lostpointercapture',finish]])surface.addEventListener(name,handler,true);
 surface.ownerDocument?.defaultView?.addEventListener('blur',reset);
 return ()=>{for(const [name,handler] of [['pointerdown',down],['pointermove',update],['pointerup',finish],['pointercancel',finish],['lostpointercapture',finish]])surface.removeEventListener(name,handler,true);surface.ownerDocument?.defaultView?.removeEventListener('blur',reset);reset();};
}
