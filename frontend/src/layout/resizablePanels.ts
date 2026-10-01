import type {ObjectDirective} from 'vue';
import {projects} from '../platform/browser.ts';
import {fitPanels,layoutKey,panelWidth,type PanelLimit} from './panelWidths.ts';
import styles from './resizablePanels.css?inline';

interface Panel extends PanelLimit { selector:string; side:'left'|'right'; label:string; variable?:string }
export interface PanelLayout { key:string; center:string; centerMin?:number; panels:Panel[]; disabled?:boolean }
type Store={read<T>(key:string,fallback:T):T;write(key:string,value:unknown):boolean};
const controllers=new WeakMap<HTMLElement,ReturnType<typeof resizePanels>>();

/** Also used by the same-origin M10 frame. Never reads scientific/page state. */
export function resizePanels(root:HTMLElement,options:PanelLayout,store:Store=projects) {
  const doc=root.ownerDocument,win=doc.defaultView!;
  if(!doc.querySelector('style[data-panel-resize]')){const style=doc.createElement('style');style.dataset.panelResize='';style.textContent=styles;doc.head.append(style);}
  root.classList.add('resizable-panels');
  const key=layoutKey(options.key),raw=store.read<unknown>(key,{});
  const saved=raw&&typeof raw==='object'?raw as Record<string,unknown>:{};
  const entries=options.panels.map(panel=>{
    const variable=panel.variable??'--panel-'+panel.side,handle=doc.createElement('div');
    handle.className='panel-resize-handle';handle.setAttribute('role','separator');handle.setAttribute('aria-orientation','vertical');handle.setAttribute('aria-label',panel.label+'宽度');handle.tabIndex=0;
    handle.title='拖动边界调整宽度；方向键微调，Home / End 调至最小 / 最大';
    root.append(handle);
    const wanted=panelWidth(saved[variable],panel);
    // Each nested layout owns its variables; never inherit the navigation width.
    root.style.setProperty(variable,wanted+'px');
    return {panel,variable,handle,wanted,width:panel.initial,max:panel.max,visible:false};
  });
  const message=doc.createElement('span');message.className='panel-resize-message';message.setAttribute('role','status');root.append(message);
  let frame=0,dead=false,disabled=!!options.disabled;
  type Entry=typeof entries[number];
  let drag:{entry:Entry;id:number;x:number;width:number;scale:number;shield:HTMLElement;wanted:number}|undefined;
  const schedule=()=>{if(!dead){win.cancelAnimationFrame(frame);frame=win.requestAnimationFrame(measure);}};
  function measure(){
    if(dead)return;
    // Vue may replace the bound class after a reactive update. The handles
    // must stay relative to this layout, including after toggling its sidebar.
    if(!root.classList.contains('resizable-panels'))root.classList.add('resizable-panels');
    const outer=root.getBoundingClientRect(),center=root.querySelector<HTMLElement>(options.center),cr=center?.getBoundingClientRect();
    const scale=outer.width/root.offsetWidth||1;
    for(const e of entries){
      const el=root.querySelector<HTMLElement>(e.panel.selector),r=el?.getBoundingClientRect();
      e.visible=!disabled&&!!r&&!!cr&&outer.width>0&&r.width/scale>=e.panel.min-2&&r.height>0&&cr.height>0&&win.getComputedStyle(el!).display!=='none'&&Math.min(r.bottom,cr.bottom)>Math.max(r.top,cr.top)&&
        (e.panel.side==='left'?r.right<=cr.left+2:r.left>=cr.right-2);
      e.handle.hidden=!e.visible;
    }
    const active=entries.filter(e=>e.visible);
    if(!cr||!active.length)return;
    const gap=parseFloat(win.getComputedStyle(root).columnGap)||0;
    const widths=fitPanels(active.map(e=>e.wanted),active.map(e=>e.panel.min),root.clientWidth-gap*active.length-(options.centerMin??280));
    active.forEach((e,i)=>{const value=widths[i]+'px';if(root.style.getPropertyValue(e.variable)!==value)root.style.setProperty(e.variable,value);});
    const content=root.querySelector<HTMLElement>(options.center)!.getBoundingClientRect();
    for(const e of active){
      const r=root.querySelector<HTMLElement>(e.panel.selector)!.getBoundingClientRect();
      e.width=r.width/scale;e.max=Math.max(e.panel.min,Math.min(e.panel.max,e.width+content.width/scale-(options.centerMin??280)));
      const edge=e.panel.side==='left'?r.right:r.left;
      Object.assign(e.handle.style,{left:((edge-outer.left)/scale+root.scrollLeft-root.clientLeft-4)+'px',top:((Math.min(r.top,content.top)-outer.top)/scale+root.scrollTop-root.clientTop)+'px',height:((Math.max(r.bottom,content.bottom)-Math.min(r.top,content.top))/scale)+'px'});
      e.handle.setAttribute('aria-valuemin',String(e.panel.min));e.handle.setAttribute('aria-valuemax',String(Math.round(e.max)));e.handle.setAttribute('aria-valuenow',String(Math.round(e.width)));e.handle.setAttribute('aria-valuetext',Math.round(e.width)+' 像素');
    }
  }
  function persist(){
    const widths=Object.fromEntries(entries.map(e=>[e.variable,e.wanted]));
    message.textContent=store.write(key,widths)?'':'栏宽未能保存，下次打开可能恢复默认宽度。';
  }
  function finish(cancel=false){
    if(!drag)return;const d=drag;drag=undefined;
    d.shield.remove();d.entry.handle.classList.remove('dragging');
    if(d.entry.handle.hasPointerCapture(d.id))d.entry.handle.releasePointerCapture(d.id);
    if(cancel)d.entry.wanted=d.wanted;else persist();schedule();
  }
  for(const e of entries){
    e.handle.addEventListener('pointerdown',event=>{
      if(event.button!==0||!e.visible)return;event.preventDefault();finish();measure();
      const shield=doc.createElement('div');shield.className='panel-resize-shield';doc.body.append(shield);
      drag={entry:e,id:event.pointerId,x:event.clientX,width:e.width,scale:root.getBoundingClientRect().width/root.offsetWidth||1,shield,wanted:e.wanted};
      e.handle.focus({preventScroll:true});e.handle.setPointerCapture(event.pointerId);e.handle.classList.add('dragging');
    });
    e.handle.addEventListener('pointermove',event=>{
      if(!drag||drag.entry!==e||drag.id!==event.pointerId)return;
      e.wanted=panelWidth(Math.min(e.max,drag.width+(event.clientX-drag.x)/drag.scale*(e.panel.side==='left'?1:-1)),e.panel);measure();
    });
    e.handle.addEventListener('pointerup',()=>finish());e.handle.addEventListener('pointercancel',()=>finish(true));e.handle.addEventListener('lostpointercapture',()=>finish(true));
    e.handle.addEventListener('keydown',event=>{
      if(event.key==='Escape'){finish(true);return;}
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)||!e.visible)return;
      event.preventDefault();measure();const direction=(event.key==='ArrowRight'?1:-1)*(e.panel.side==='left'?1:-1);
      e.wanted=panelWidth(event.key==='Home'?e.panel.min:event.key==='End'?e.max:Math.min(e.max,e.width+direction*(event.shiftKey?40:10)),e.panel);measure();persist();
    });
  }
  const blur=()=>finish(true),resize=new ResizeObserver(schedule),mutation=new MutationObserver(schedule);
  resize.observe(root);root.querySelectorAll<HTMLElement>([options.center,...options.panels.map(p=>p.selector)].join(',')).forEach(el=>resize.observe(el));
  // Attribute changes cover responsive mode, collapsed panels and v-show tabs.
  mutation.observe(root,{attributes:true,attributeFilter:['class']});
  mutation.observe(doc.documentElement,{attributes:true,attributeFilter:['style','data-columns']});
  for(const p of options.panels){const el=root.querySelector(p.selector);if(el)mutation.observe(el,{attributes:true,attributeFilter:['style','class']});}
  win.addEventListener('resize',schedule);win.addEventListener('blur',blur);root.addEventListener('scroll',schedule);schedule();
  return {update(value:PanelLayout){if(!root.classList.contains('resizable-panels'))root.classList.add('resizable-panels');disabled=!!value.disabled;schedule();},destroy(){finish(true);dead=true;win.cancelAnimationFrame(frame);resize.disconnect();mutation.disconnect();win.removeEventListener('resize',schedule);win.removeEventListener('blur',blur);root.removeEventListener('scroll',schedule);entries.forEach(e=>e.handle.remove());message.remove();}};
}
export const vResizablePanels:ObjectDirective<HTMLElement,PanelLayout>={mounted(el,{value}){controllers.set(el,resizePanels(el,value));},updated(el,{value}){controllers.get(el)?.update(value);},unmounted(el){controllers.get(el)?.destroy();controllers.delete(el);}};
