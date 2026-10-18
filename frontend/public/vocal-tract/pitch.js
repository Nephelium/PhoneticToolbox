import {canvasFont} from './fonts.js';
import {tint,dark} from './theme.js';
import {frameIntervals,silentAt,framePitch,framePressure} from './keyframes.js';
export function setupPitch({initialCurve=[],initialPressureCurve=[],getFrames,onChange,isBusy}){
 const $=id=>document.getElementById(id),canvas=$('pitchCanvas');
 const specs={f0:{lo:60,hi:350,unit:'Hz',name:'基频',step:5,base:framePitch},pressure:{lo:0,hi:1600,unit:'Pa',name:'驱动气压',step:20,base:framePressure}};
 const curves={f0:initialCurve.map(p=>[...p]),pressure:initialPressureCurve.map(p=>[...p])},undo={f0:[],pressure:[]};
 let mode='f0',drag=null,cursor=0;
 const total=()=>getFrames().reduce((sum,f)=>sum+f.duration,0),copy=key=>curves[key].map(p=>[...p]);
 const blocked=(x,key=mode)=>silentAt(getFrames(),x,key==='f0');
 const color=key=>key==='f0'?(dark()?'#8db9ff':'#225db6'):(dark()?'#f4b077':'#ac5319');
 function value(x,key=mode){const curve=curves[key];if(!curve.length)return specs[key].base(getFrames(),x*total());let i=1;while(i<curve.length-1&&curve[i][0]<x)i++;const a=curve[i-1],b=curve[i],t=(x-a[0])/(b[0]-a[0]);return a[1]+t*(b[1]-a[1]);}
 function draw(){
  const empty=!getFrames().length;$('pitchEmpty').hidden=!empty;canvas.dataset.empty=String(empty);canvas.setAttribute('aria-disabled',String(empty||isBusy()));$('pitchAddFrame').disabled=isBusy();
  if(empty){$('pitchStatus').textContent='';$('pitchUndo').disabled=$('pitchReset').disabled=true;return;}
  const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const d=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);}const c=canvas.getContext('2d',{alpha:false});c.setTransform(d,0,0,d,0,0);c.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();c.fillRect(0,0,w,h);
  const top=48,bottom=25,x=t=>42+t*(w-94),y=(v,key)=>top+(specs[key].hi-v)/(specs[key].hi-specs[key].lo)*(h-top-bottom);c.font=canvasFont(10);
  for(const f of [60,150,250,350]){const yy=y(f,'f0');c.strokeStyle=tint('#e3e5da');c.beginPath();c.moveTo(38,yy);c.lineTo(w-50,yy);c.stroke();c.textAlign='left';c.fillStyle=color('f0');c.fillText(f,2,yy+3);c.textAlign='right';c.fillStyle=color('pressure');c.fillText(Math.round((f-60)/290*1600),w-2,yy+3);}
  c.textAlign='left';c.fillStyle=color('f0');c.fillText('Hz',2,top-10);c.textAlign='right';c.fillStyle=color('pressure');c.fillText('Pa',w-2,top-10);c.fillStyle=tint('#7b877a');
  for(let i=0;i<=2;i++){c.textAlign=i===0?'left':i===2?'right':'center';c.fillText((total()*i/2).toFixed(2)+' s',x(i/2),h-3);}
  for(const f of frameIntervals(getFrames())){const left=x(f.start/total()),right=x(f.end/total());c.fillStyle=f.silent?tint('#7b877a')+'40':f.aspiration?color('pressure')+'12':tint(f.index%2?'#39877916':'#39877908');c.fillRect(left,0,right-left,h-bottom);c.strokeStyle=tint('#adbaaf');c.setLineDash([2,3]);c.beginPath();c.moveTo(left,0);c.lineTo(left,h-bottom);c.moveTo(right,0);c.lineTo(right,h-bottom);c.stroke();c.setLineDash([]);c.save();c.beginPath();c.rect(left+2,0,Math.max(0,right-left-4),42);c.clip();c.fillStyle=tint('#51756b');c.textAlign='center';c.font=canvasFont(11,true);c.fillText(f.name,(left+right)/2,16);c.font=canvasFont(10);c.fillText(`${f.start.toFixed(2)}–${f.end.toFixed(2)} s`,(left+right)/2,32);c.restore();}
  for(const key of ['f0','pressure']){c.strokeStyle=color(key);c.lineWidth=key===mode?2:1.3;c.setLineDash(curves[key].length?[]:[4,3]);c.beginPath();let pen=false;for(let i=0;i<=200;i++){if(blocked(i/200,key)){pen=false;continue;}const xx=x(i/200),yy=y(value(i/200,key),key);pen?c.lineTo(xx,yy):c.moveTo(xx,yy);pen=true;}c.stroke();c.setLineDash([]);}
  if(document.activeElement===canvas&&!blocked(cursor)){c.fillStyle=color(mode);c.beginPath();c.arc(x(cursor),y(value(cursor),mode),3,0,7);c.fill();}
  $('pitchStatus').textContent=`正在绘制${specs[mode].name} · 实线为手绘，虚线为逐帧值 · 改变时长等比例伸缩。静音段不能绘制；送气段可绘制气压。`;
  $('pitchUndo').disabled=isBusy()||!undo[mode].length;$('pitchReset').disabled=isBusy()||!curves[mode].length;$('pitchReset').textContent='恢复逐帧'+specs[mode].name;
  for(const [id,key] of [['drawF0','f0'],['drawPressure','pressure']]){$(id).setAttribute('aria-pressed',mode===key);$(id).disabled=isBusy();}
  canvas.setAttribute('aria-label','手绘'+specs[mode].name+'曲线');canvas.setAttribute('aria-valuemin',specs[mode].lo);canvas.setAttribute('aria-valuemax',specs[mode].hi);if(blocked(cursor))canvas.removeAttribute('aria-valuenow');else canvas.setAttribute('aria-valuenow',Math.round(value(cursor)));canvas.setAttribute('aria-valuetext',`${(cursor*total()).toFixed(2)} 秒，${blocked(cursor)?'当前段不能绘制':Math.round(value(cursor))+' '+specs[mode].unit}`);
 }
 const point=e=>{const r=canvas.getBoundingClientRect(),m=specs[mode];return [Math.max(0,Math.min(1,(e.clientX-r.left-42)/(r.width-94))),Math.max(m.lo,Math.min(m.hi,m.hi-(e.clientY-r.top-48)/(r.height-73)*(m.hi-m.lo)))];};
 function remember(){undo[mode].push(copy(mode));undo[mode]=undo[mode].slice(-20);}
 function paint(a,b){if(curves[mode].length!==201)curves[mode]=Array.from({length:201},(_,i)=>[i/200,value(i/200)]);const lo=Math.round(Math.min(a[0],b[0])*200),hi=Math.round(Math.max(a[0],b[0])*200);for(let i=lo;i<=hi;i++){if(blocked(i/200))continue;const t=a[0]===b[0]?1:Math.max(0,Math.min(1,(i/200-a[0])/(b[0]-a[0])));curves[mode][i][1]=Math.round(a[1]+t*(b[1]-a[1]));}cursor=b[0];draw();}
 canvas.onpointerdown=e=>{if(isBusy()||drag||getFrames().length<1||blocked(point(e)[0]))return;e.preventDefault();canvas.focus();remember();drag={id:e.pointerId,point:point(e)};paint(drag.point,drag.point);canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!drag||drag.id!==e.pointerId)return;const p=point(e);paint(drag.point,p);drag.point=p;};
 for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,e=>{if(drag?.id===e.pointerId){drag=null;if(event==='pointercancel')curves[mode]=undo[mode].pop();else onChange();draw();}});
 canvas.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.stopPropagation();if(!isBusy())$('pitchUndo').click();return;}if(isBusy()||getFrames().length<1||!e.key.startsWith('Arrow'))return;e.preventDefault();if(['ArrowLeft','ArrowRight'].includes(e.key)){cursor=Math.max(0,Math.min(1,cursor+(e.key==='ArrowRight'?.01:-.01)));draw();return;}if(blocked(cursor))return;remember();const m=specs[mode],p=[cursor,Math.max(m.lo,Math.min(m.hi,value(cursor)+(e.key==='ArrowUp'?m.step:-m.step)))];paint(p,p);onChange();};
 for(const [id,key] of [['drawF0','f0'],['drawPressure','pressure']])$(id).onclick=()=>{if(isBusy()||drag)return;mode=key;draw();};
 document.addEventListener('m10-theme',draw);canvas.onfocus=draw;canvas.onblur=draw;
 $('pitchUndo').onclick=()=>{if(isBusy()||!undo[mode].length)return;curves[mode]=undo[mode].pop();onChange();draw();};$('pitchReset').onclick=()=>{if(isBusy())return;remember();curves[mode]=[];onChange();draw();};
 $('pitchToggle').onclick=()=>{const open=$('pitchEditor').hidden;$('pitchEditor').hidden=!open;$('pitchToggle').setAttribute('aria-expanded',open);document.dispatchEvent(new Event('pitch-layout'));draw();};new ResizeObserver(draw).observe(canvas);
 const editor=$('pitchEditor'),home=editor.parentElement,dialog=$('pitchDialog');$('pitchExpand').onclick=()=>{editor.hidden=false;$('pitchDialogBody').append(editor);dialog.showModal();draw();};$('pitchClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{home.append(editor);draw();});
 $('pitchAddFrame').onclick=()=>{if(!isBusy())$('captureFrame').click();};
 const set=(key,value)=>{curves[key]=(value||[]).map(p=>[...p]);undo[key]=[];draw();};return {draw,getCurve:()=>copy('f0'),setCurve:value=>set('f0',value),getPressureCurve:()=>copy('pressure'),setPressureCurve:value=>set('pressure',value)};
}
