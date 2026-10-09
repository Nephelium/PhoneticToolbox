import {canvasFont} from './fonts.js';
import {tint} from './theme.js';
// Same normalized-time editing convention as the desktop phonation F0 canvas.
import {frameIntervals,silentAt} from './keyframes.js';
export function setupPitch({initialCurve=[],getFrames,onChange,isBusy}){
  const $=id=>document.getElementById(id),canvas=$('pitchCanvas');
  let curve=initialCurve,undo=[],drag=null,cursor=0;
  const total=()=>getFrames().reduce((sum,f)=>sum+f.duration,0);
  const copy=()=>curve.map(p=>[...p]);
  function frameF0(x){
    const frames=getFrames();if(!frames.length)return 150;let t=x*total();
    for(let i=0;i<frames.length-1;i++){if(t<frames[i].duration){let u=t/frames[i].duration;u=u*u*(3-2*u);return frames[i].f0*(1-u)+(frames[i+1].silent?frames[i].f0:frames[i+1].f0)*u;}t-=frames[i].duration;}
    return frames.at(-1).f0;
  }
  function value(x){if(!curve.length)return frameF0(x);let i=1;while(i<curve.length-1&&curve[i][0]<x)i++;const a=curve[i-1],b=curve[i],t=(x-a[0])/(b[0]-a[0]);return a[1]+t*(b[1]-a[1]);}
  function draw(){
    const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const d=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);}const c=canvas.getContext('2d',{alpha:false});c.setTransform(d,0,0,d,0,0);c.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();c.fillRect(0,0,w,h);
    const top=48,bottom=25;
    const x=t=>42+t*(w-54),y=f=>top+(350-f)/290*(h-top-bottom);c.font=canvasFont(11);
    for(const f of [60,150,250,350]){c.strokeStyle=tint('#e3e5da');c.beginPath();c.moveTo(34,y(f));c.lineTo(w-9,y(f));c.stroke();c.fillStyle=tint('#7b877a');c.fillText(f,3,y(f)+3);}
    for(let i=0;i<=2;i++){c.textAlign=i===0?'left':i===2?'right':'center';c.fillText((total()*i/2).toFixed(2)+' s',x(i/2),h-3);}c.textAlign='left';
    for(const f of frameIntervals(getFrames())){
      const left=x(f.start/total()),right=x(f.end/total());
      c.fillStyle=f.silent?(tint('#7b877a')+'40'):tint(f.index%2?'#39877916':'#39877908');c.fillRect(left,0,right-left,h-bottom);
      c.strokeStyle=tint('#adbaaf');c.setLineDash([2,3]);c.beginPath();c.moveTo(left,0);c.lineTo(left,h-bottom);c.moveTo(right,0);c.lineTo(right,h-bottom);c.stroke();c.setLineDash([]);
      c.save();c.beginPath();c.rect(left+2,0,Math.max(0,right-left-4),42);c.clip();c.fillStyle=tint('#51756b');c.textAlign='center';c.font=canvasFont(11,true);c.fillText(f.name,(left+right)/2,16);c.font=canvasFont(10);c.fillText(`${f.start.toFixed(2)}–${f.end.toFixed(2)} s`,(left+right)/2,32);c.restore();
    }
    c.strokeStyle=tint(curve.length?'#226f65':'#a88973');c.lineWidth=1.6;c.setLineDash(curve.length?[]:[4,3]);c.beginPath();let pen=false;for(let i=0;i<=200;i++){if(silentAt(getFrames(),i/200)){pen=false;continue;}const xx=x(i/200),yy=y(value(i/200));if(pen)c.lineTo(xx,yy);else c.moveTo(xx,yy);pen=true;}c.stroke();c.setLineDash([]);
    if(document.activeElement===canvas&&!silentAt(getFrames(),cursor)){c.fillStyle=tint('#226f65');c.beginPath();c.arc(x(cursor),y(value(cursor)),3,0,7);c.fill();}
    $('pitchStatus').textContent=(curve.length?'实线：手绘 F0 · 改变时长会等比例伸缩':'虚线：逐帧 F0 · 在图内拖动绘制')+' · 灰色静音段无基频，不能绘制';
    $('pitchUndo').disabled=isBusy()||!undo.length;$('pitchReset').disabled=isBusy()||!curve.length;
    if(silentAt(getFrames(),cursor))canvas.removeAttribute('aria-valuenow');else canvas.setAttribute('aria-valuenow',Math.round(value(cursor)));canvas.setAttribute('aria-valuetext',`${(cursor*total()).toFixed(2)} 秒，${silentAt(getFrames(),cursor)?'静音，无基频':Math.round(value(cursor))+' Hz'}`);
  }
  const point=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left-42)/(r.width-54))),Math.max(60,Math.min(350,350-(e.clientY-r.top-48)/(r.height-73)*290))];};
  function paint(a,b){
    if(curve.length!==201)curve=Array.from({length:201},(_,i)=>[i/200,value(i/200)]);
    const lo=Math.round(Math.min(a[0],b[0])*200),hi=Math.round(Math.max(a[0],b[0])*200);
    for(let i=lo;i<=hi;i++){if(silentAt(getFrames(),i/200))continue;const t=a[0]===b[0]?1:Math.max(0,Math.min(1,(i/200-a[0])/(b[0]-a[0])));curve[i][1]=Math.round(a[1]+t*(b[1]-a[1]));}cursor=b[0];draw();
  }
  canvas.onpointerdown=e=>{if(isBusy()||getFrames().length<2||silentAt(getFrames(),point(e)[0]))return;e.preventDefault();canvas.focus();undo.push(copy());undo=undo.slice(-20);drag=point(e);paint(drag,drag);canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;const p=point(e);paint(drag,p);drag=p;};
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{if(drag){drag=null;onChange();draw();}});
  canvas.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.stopPropagation();if(!isBusy())$('pitchUndo').click();return;}if(isBusy()||getFrames().length<2||!e.key.startsWith('Arrow'))return;e.preventDefault();if(['ArrowLeft','ArrowRight'].includes(e.key)){cursor=Math.max(0,Math.min(1,cursor+(e.key==='ArrowRight'?.01:-.01)));draw();return;}if(silentAt(getFrames(),cursor))return;undo.push(copy());const p=[cursor,Math.max(60,Math.min(350,value(cursor)+(e.key==='ArrowUp'?5:-5)))];paint(p,p);onChange();};
  document.addEventListener('m10-theme',draw);
  canvas.onfocus=draw;canvas.onblur=draw;
  $('pitchUndo').onclick=()=>{if(isBusy()||!undo.length)return;curve=undo.pop();onChange();draw();};
  $('pitchReset').onclick=()=>{if(isBusy())return;undo.push(copy());curve=[];onChange();draw();};
  $('pitchToggle').onclick=()=>{const open=$('pitchEditor').hidden;$('pitchEditor').hidden=!open;$('pitchToggle').setAttribute('aria-expanded',open);document.dispatchEvent(new Event('pitch-layout'));draw();};
  new ResizeObserver(draw).observe(canvas);
  const editor=$('pitchEditor'),home=editor.parentElement,dialog=$('pitchDialog');
  $('pitchExpand').onclick=()=>{editor.hidden=false;$('pitchDialogBody').append(editor);dialog.showModal();draw();};
  $('pitchClose').onclick=()=>dialog.close();
  dialog.addEventListener('close',()=>{home.append(editor);draw();});
  return {draw,getCurve:copy,setCurve:value=>{curve=value.map(p=>[...p]);undo=[];draw();}};
}
