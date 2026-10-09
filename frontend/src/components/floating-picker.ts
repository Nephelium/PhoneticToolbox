import {ref,watch,nextTick,onMounted,onUnmounted,type Ref} from 'vue';

// Teleport escapes the settings card's scrolling container. Coordinates remain
// in CSS pixels when the workbench uses its shared page zoom setting.
export function useFloatingPicker(anchor:Ref<HTMLElement|undefined>,panel:Ref<HTMLElement|undefined>){
 const open=ref(false),position=ref<Record<string,string>>({});
 function place(){
  if(!open.value||!anchor.value)return;
  const scale=parseFloat(document.documentElement.style.zoom)||1,rect=anchor.value.getBoundingClientRect();
  const width=Math.min(Math.max(rect.width/scale,240),window.innerWidth/scale-16);
  const below=window.innerHeight/scale-rect.bottom/scale-12,above=rect.top/scale-12;
  const up=below<180&&above>below,height=Math.max(60,Math.min(360,up?above:below));
  position.value={left:Math.max(8,Math.min(rect.left/scale,window.innerWidth/scale-width-8))+'px',width:width+'px',maxHeight:height+'px',
   ...(up?{bottom:(window.innerHeight-rect.top)/scale+4+'px'}:{top:rect.bottom/scale+4+'px'})};
 }
 function outside(event:PointerEvent){const target=event.target as Node;if(!anchor.value?.contains(target)&&!panel.value?.contains(target))open.value=false;}
 function escape(event:KeyboardEvent){if(event.key==='Escape'&&open.value){event.preventDefault();open.value=false;anchor.value?.querySelector<HTMLElement>('input,button')?.focus();}}
 watch(open,async value=>{if(value){await nextTick();place();}});
 onMounted(()=>{document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);window.addEventListener('resize',place);window.addEventListener('scroll',place,true);});
 onUnmounted(()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);window.removeEventListener('resize',place);window.removeEventListener('scroll',place,true);});
 return {open,position,place};
}
