import type {ObjectDirective} from 'vue';

export type TimeUnit = 's' | 'ms';
const digits = (unit:TimeUnit) => unit === 's' ? 5 : 2;

// Only the displayed value is formatted. Scientific state is never written here.
export function timeText(value:string|number,unit:TimeUnit):string {
 const text=String(value);
 if(!text.trim()||!Number.isFinite(Number(text)))return text;
 const rounded=Number(text).toFixed(digits(unit));
 return rounded.includes('.')?rounded.replace(/0+$/,'').replace(/\.$/,'').replace(/^-0$/,'0'):rounded;
}
export function limitTimeInput(value:string,unit:TimeUnit):string {
 // Keep partially typed decimals and permitted trailing zeros while editing.
 const fraction=value.match(/\.(\d*)/);
 return /e/i.test(value)||(fraction?.[1].length??0)>digits(unit)?timeText(value,unit):value;
}

type State={unit:TimeUnit|undefined;input:()=>void;commit:()=>void};
const states=new WeakMap<HTMLInputElement,State>();
function display(el:HTMLInputElement){
 const unit=states.get(el)?.unit;
 if(unit)el.value=document.activeElement===el?limitTimeInput(el.value,unit):timeText(el.value,unit);
}
function unit(el:HTMLInputElement,value:TimeUnit|undefined){
 states.get(el)!.unit=value;
 if(value)el.dataset.timeUnit=value;else delete el.dataset.timeUnit;
}
export const vTimePrecision:ObjectDirective<HTMLInputElement,TimeUnit|undefined>={
 created(el,binding){
  const normalize=(commit:boolean)=>{const current=states.get(el)?.unit;if(current)el.value=commit?timeText(el.value,current):limitTimeInput(el.value,current);};
  const state:State={unit:binding.value,input:()=>normalize(false),commit:()=>normalize(true)};
  states.set(el,state);unit(el,binding.value);
  // Capture runs before v-model and existing @input/@change handlers, including paste.
  el.addEventListener('input',state.input,true);el.addEventListener('change',state.commit,true);el.addEventListener('blur',state.commit,true);
 },
 mounted(el){queueMicrotask(()=>display(el));},
 updated(el,binding){unit(el,binding.value);display(el);},
 beforeUnmount(el){const state=states.get(el);if(state){el.removeEventListener('input',state.input,true);el.removeEventListener('change',state.commit,true);el.removeEventListener('blur',state.commit,true);states.delete(el);}},
};
