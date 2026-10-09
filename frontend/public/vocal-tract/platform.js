// M10 platform port. This module cannot open sockets or choose native paths.
import {applyFonts} from './fonts.js';
const pending=new Map();let sequence=0;
let resolveChart;
export const ipaChart=new Promise(resolve=>{resolveChart=resolve;});
export function request(op,body={}){
  return new Promise((resolve,reject)=>{
    const id=String(++sequence),timer=setTimeout(()=>{pending.delete(id);reject(Error('声道引擎请求超时'));},160000);
    pending.set(id,{resolve,reject,timer});parent.postMessage({type:'m10-request',id,op,body},'*');
  });
}
window.addEventListener('message',event=>{
  if(event.source!==parent)return;
  const data=event.data;
  if(data?.type==='m10-ipa-chart'&&data.chart)resolveChart(data.chart);
  if(data?.type==='m10-fonts')void applyFonts(data.fonts);
  if(data?.type==='m10-response'){
    const item=pending.get(data.id);if(!item)return;pending.delete(data.id);clearTimeout(item.timer);
    data.ok?item.resolve(data.value):item.reject(Error(data.error));
  }
  if(data?.type==='m10-theme'){
    const root=document.documentElement;
    const map={'--app':'--bg','--panel':'--panel','--text':'--ink','--muted':'--muted','--border':'--line','--accent':'--accent','--selected':'--selected','--on-accent':'--on-accent','--warning':'--warning','--danger':'--danger','--sidebar':'--sidebar','--waveform-color':'--waveform-color'};
    for(const [key,target] of Object.entries(map)){const value=data.colors?.[key];if(typeof value==='string'&&/^#[0-9a-f]{3,8}$/i.test(value))root.style.setProperty(target,value);else root.style.removeProperty(target);}
    root.style.setProperty('--green',root.style.getPropertyValue('--accent')||'#0969da');
    root.dataset.theme=data.theme==='dark'?'dark':'light';root.style.colorScheme=root.dataset.theme;document.dispatchEvent(new Event('m10-theme'));
    if(data.buttons){
      root.dataset.buttonStyle=['auto','all','plain'].includes(data.buttons.mode)?data.buttons.mode:'auto';
      root.dataset.buttonEffects=data.buttons.effects==='off'?'off':'on';
      if(typeof data.buttons.css==='string'){
        let style=document.getElementById('ptb-button-styles');
        if(!style){style=document.createElement('style');style.id='ptb-button-styles';document.head.append(style);}
        if(style.textContent!==data.buttons.css)style.textContent=data.buttons.css;
      }
    }
  }
});
parent.postMessage({type:'m10-ready'},'*');
export const response=async(op,body)=>{const value=await request(op,body);return {ok:true,json:async()=>value};};
