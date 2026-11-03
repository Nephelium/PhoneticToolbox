import {computed,shallowRef,toRaw,type Ref} from 'vue';
import {messageText,errorMessageDescriptor,receiveDisplayError,type MessageDescriptor,type MessageValues} from '../i18n/index.ts';
import {RuntimeDisplayError} from '../i18n/runtime-errors.ts';

type MessagePart=string|number|boolean|null|undefined|MessageDescriptor;
export type MessageSnapshotPart=string|number|boolean|null|MessageSnapshot;
export type MessageSnapshot={type:'message';key:string;values?:Record<string,MessageSnapshotPart>}|{type:'joined';parts:MessageSnapshotPart[]};
const messageTrees=new WeakMap<MessageDescriptor,{type:'message';values:Record<string,MessagePart>}|{type:'joined';parts:MessagePart[]}>();
const partText=(part:MessagePart):string=>typeof part==='object'&&part!==null?messageText(part):String(part??'');

/** Structured composition keeps current-locale labels alongside captured event data. */
export function uiMessage(key:string,values:Record<string,MessagePart>={}):MessageDescriptor {
 const named:MessageValues={};
 for(const [name,value]of Object.entries(values))Object.defineProperty(named,name,{enumerable:true,get:()=>typeof value==='object'&&value!==null?messageText(value):value});
 const descriptor={key,values:named};messageTrees.set(descriptor,{type:'message',values:{...values}});return descriptor;
}
export function joinMessages(...parts:MessagePart[]):MessageDescriptor {
 const values:MessageValues={};Object.defineProperty(values,'text',{enumerable:true,get:()=>parts.map(partText).join('')});
 const descriptor={key:'modules.display.composed',values};messageTrees.set(descriptor,{type:'joined',parts});return descriptor;
}
/** Capture message structure for an owned bridge, without evaluating nested labels. */
export function messageSnapshot(value:string|MessageDescriptor):MessageSnapshotPart {
 let remaining=512;const active=new Set<object>();
 const visit=(part:MessagePart,depth:number):MessageSnapshotPart=>{
  if(--remaining<0||depth>16)throw new RangeError('Message snapshot exceeds structural limits');
  if(part===null||part===undefined)return null;
  if(typeof part!=='object')return part;
  part=toRaw(part);if(active.has(part))throw new TypeError('Cyclic message descriptor');active.add(part);
  try{
   const tree=messageTrees.get(part);
   if(tree?.type==='joined')return {type:'joined',parts:tree.parts.map(child=>visit(child,depth+1))};
   const values=tree?.type==='message'?tree.values:part.values;
   return {type:'message',key:part.key,...(values?{values:Object.fromEntries(Object.entries(values).map(([key,child])=>[key,visit(child,depth+1)]))}:{})};
  }finally{active.delete(part);}
 };
 return visit(value,0);
}
export class ModuleDisplayError extends RuntimeDisplayError {
 constructor(value:MessageDescriptor|string){const display=typeof value==='string'?errorMessageDescriptor(value)??joinMessages(value):value;super(display,messageText(display));}
}
export const uiError=(display:MessageDescriptor|string)=>new ModuleDisplayError(display);
/** Decode owned bridge failures at the boundary; unknown diagnostics stay literal. */
export const responseError=(raw:unknown,display?:unknown)=>{
 if(display){const error=receiveDisplayError(display,raw);if(error instanceof RuntimeDisplayError)return error;}
 return uiError(typeof raw==='string'&&raw?raw:uiMessage('common.status.failed'));
};
export function uiErrorMessage(error:unknown):string|MessageDescriptor {
 return error instanceof RuntimeDisplayError?error.display:error instanceof Error?error.message:String(error);
}
const messageProperties=new WeakMap<object,Map<PropertyKey,Ref<string|MessageDescriptor>>>();
/** Adds a translated display getter while retaining the public string field. */
export function messageProperty<T extends object,K extends keyof T>(target:T,key:K):T {
 target=toRaw(target);
 if(messageProperties.get(target)?.has(key))return target;
 const value=shallowRef<string|MessageDescriptor>(String(target[key]??''));
 let fields=messageProperties.get(target);if(!fields){fields=new Map();messageProperties.set(target,fields);}fields.set(key,value);
 Object.defineProperty(target,key,{enumerable:true,configurable:true,get:()=>messageText(value.value),set:(next:string|MessageDescriptor)=>{value.value=next;}});
 return target;
}
export function setMessageProperty(target:object,key:PropertyKey,value:string|MessageDescriptor):void {
 target=toRaw(target);
 if(!messageProperties.get(target)?.has(key))messageProperty(target as Record<PropertyKey,unknown>,key);
 Reflect.set(target,key,value);
}
export function getMessageProperty(target:object,key:PropertyKey):string|MessageDescriptor {
 target=toRaw(target);
 return messageProperties.get(target)?.get(key)?.value??String(Reflect.get(target,key)??'');
}

/** Lazy, cached labels only. Never wrap scientific configuration or persisted data. */
export function localized<T extends object>(factory:()=>T):T {
 const value=computed(factory);
 const target=(Array.isArray(value.value)?[]:{}) as T;
 return new Proxy(target,{
  get(_target,key){return Reflect.get(value.value,key);},
  has(_target,key){return Reflect.has(value.value,key);},
  ownKeys(){return Reflect.ownKeys(value.value);},
  getOwnPropertyDescriptor(_target,key){
   if(Array.isArray(target)&&key==='length')return {...Object.getOwnPropertyDescriptor(target,key)!,value:(value.value as unknown[]).length};
   const descriptor=Object.getOwnPropertyDescriptor(value.value,key);
   return descriptor?{...descriptor,configurable:true}:undefined;
  },
 });
}
