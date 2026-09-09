// Same normalized-time editing convention as the desktop phonation F0 canvas.
export function setupPitch({initialCurve=[],getFrames,onChange,isBusy}){
  const $=id=>document.getElementById(id),canvas=$('pitchCanvas');
  let curve=initialCurve,undo=[],drag=null,cursor=0;
  const total=()=>getFrames().reduce((sum,f)=>sum+f.duration,0);
  const copy=()=>curve.map(p=>[...p]);
  function frameF0(x){
    const frames=getFrames();if(!frames.length)return 125;let t=x*total();
    for(let i=0;i<frames.length-1;i++){if(t<=frames[i].duration){let u=t/frames[i].duration;u=u*u*(3-2*u);return frames[i].f0*(1-u)+frames[i+1].f0*u;}t-=frames[i].duration;}
    return frames.at(-1).f0;
  }
  function value(x){if(!curve.length)return frameF0(x);let i=1;while(i<curve.length-1&&curve[i][0]<x)i++;const a=curve[i-1],b=curve[i],t=(x-a[0])/(b[0]-a[0]);return a[1]+t*(b[1]-a[1]);}
  function draw(){
    const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const d=Math.min(devicePixelRatio,2);canvas.width=w*d;canvas.height=h*d;const c=canvas.getContext('2d');c.scale(d,d);
    const x=t=>34+t*(w-43),y=f=>8+(350-f)/290*(h-30);c.font='9px "Microsoft YaHei"';
    for(const f of [60,150,250,350]){c.strokeStyle='#e3e5da';c.beginPath();c.moveTo(34,y(f));c.lineTo(w-9,y(f));c.stroke();c.fillStyle='#7b877a';c.fillText(f,3,y(f)+3);}
    for(let i=0;i<=2;i++)c.fillText((total()*i/2).toFixed(1)+' s',x(i/2)-8,h-3);
    let elapsed=0;for(const f of getFrames().slice(0,-1)){elapsed+=f.duration;c.strokeStyle='#d7c4aa';c.setLineDash([2,3]);c.beginPath();c.moveTo(x(elapsed/total()),8);c.lineTo(x(elapsed/total()),h-22);c.stroke();}c.setLineDash([]);
    c.strokeStyle=curve.length?'#226f65':'#a88973';c.lineWidth=1.6;c.setLineDash(curve.length?[]:[4,3]);c.beginPath();for(let i=0;i<=200;i++){const xx=x(i/200),yy=y(value(i/200));if(i)c.lineTo(xx,yy);else c.moveTo(xx,yy);}c.stroke();c.setLineDash([]);
    if(document.activeElement===canvas){c.fillStyle='#226f65';c.beginPath();c.arc(x(cursor),y(value(cursor)),3,0,7);c.fill();}
    $('pitchStatus').textContent=curve.length?'实线：手绘 F0 · 改变时长会等比例伸缩':'虚线：逐帧 F0 · 在图内拖动绘制';
    $('pitchUndo').disabled=isBusy()||!undo.length;$('pitchReset').disabled=isBusy()||!curve.length;
    canvas.setAttribute('aria-valuenow',Math.round(value(cursor)));canvas.setAttribute('aria-valuetext',`${(cursor*total()).toFixed(2)} 秒，${Math.round(value(cursor))} Hz`);
  }
  const point=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left-34)/(r.width-43))),Math.max(60,Math.min(350,350-(e.clientY-r.top-8)/(r.height-30)*290))];};
  function paint(a,b){
    if(curve.length!==201)curve=Array.from({length:201},(_,i)=>[i/200,value(i/200)]);
    const lo=Math.round(Math.min(a[0],b[0])*200),hi=Math.round(Math.max(a[0],b[0])*200);
    for(let i=lo;i<=hi;i++){const t=a[0]===b[0]?1:Math.max(0,Math.min(1,(i/200-a[0])/(b[0]-a[0])));curve[i][1]=Math.round(a[1]+t*(b[1]-a[1]));}cursor=b[0];draw();
  }
  canvas.onpointerdown=e=>{if(isBusy()||getFrames().length<2)return;e.preventDefault();canvas.focus();undo.push(copy());undo=undo.slice(-20);drag=point(e);paint(drag,drag);canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;const p=point(e);paint(drag,p);drag=p;};
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{if(drag){drag=null;onChange();draw();}});
  canvas.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.stopPropagation();if(!isBusy())$('pitchUndo').click();return;}if(isBusy()||getFrames().length<2||!e.key.startsWith('Arrow'))return;e.preventDefault();if(['ArrowLeft','ArrowRight'].includes(e.key)){cursor=Math.max(0,Math.min(1,cursor+(e.key==='ArrowRight'?.01:-.01)));draw();return;}undo.push(copy());const p=[cursor,Math.max(60,Math.min(350,value(cursor)+(e.key==='ArrowUp'?5:-5)))];paint(p,p);onChange();};
  canvas.onfocus=draw;canvas.onblur=draw;
  $('pitchUndo').onclick=()=>{if(isBusy()||!undo.length)return;curve=undo.pop();onChange();draw();};
  $('pitchReset').onclick=()=>{if(isBusy())return;undo.push(copy());curve=[];onChange();draw();};
  $('pitchToggle').onclick=()=>{const open=$('pitchEditor').hidden;$('pitchEditor').hidden=!open;$('pitchToggle').setAttribute('aria-expanded',open);document.dispatchEvent(new Event('pitch-layout'));draw();};
  new ResizeObserver(draw).observe(canvas);
  return {draw,getCurve:copy};
}
