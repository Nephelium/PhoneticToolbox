import type {PaperAnnotation} from '../../platform/papers.ts';

export interface AnnotationChange {before:PaperAnnotation|null;after:PaperAnnotation|null}
export type AnnotationDirection='undo'|'redo';
type Entry={label:string;changes:AnnotationChange[]};
type Write=(item:PaperAnnotation,remove:boolean)=>Promise<boolean>;

export function copyAnnotation(item:PaperAnnotation):PaperAnnotation{
 return {id:item.id,page:item.page,kind:item.kind,color:item.color,text:item.text,quote:item.quote,rects:item.rects.map(r=>[...r])};
}

// Session history stores confirmed edits only. Each PDF/language owns its own instance.
export class AnnotationHistory {
 private past:Entry[]=[];
 private future:Entry[]=[];
 private limit:number;
 constructor(limit=100){this.limit=limit;}
 get canUndo(){return this.past.length>0;}
 get canRedo(){return this.future.length>0;}
 get undoLabel(){return this.past.at(-1)?.label??'';}
 get redoLabel(){return this.future.at(-1)?.label??'';}
 clear(){this.past=[];this.future=[];}
 record(label:string,changes:AnnotationChange[]){
  const copy=changes.map(c=>({before:c.before?copyAnnotation(c.before):null,after:c.after?copyAnnotation(c.after):null}))
   .filter(c=>JSON.stringify(c.before)!==JSON.stringify(c.after));
  if(!copy.length)return;
  this.past.push({label,changes:copy});if(this.past.length>this.limit)this.past.shift();this.future=[];
 }
 async travel(direction:AnnotationDirection,write:Write){
  const source=direction==='undo'?this.past:this.future,destination=direction==='undo'?this.future:this.past;
  const entry=source.at(-1);if(!entry)return {changed:0,complete:true};
  const ordered=direction==='undo'?[...entry.changes].reverse():entry.changes;
  let done=0;
  try{
   for(const change of ordered){
    const target=direction==='undo'?change.before:change.after;
    const item=target??(direction==='undo'?change.after:change.before)!;
    if(!await write(copyAnnotation(item),target===null))break;
    done++;
   }
  }finally{
   // A multi-page edit can fail partway through. Keep both stacks aligned with
   // the writes actually acknowledged by the existing revisioned document port.
   if(done){
    source.pop();
    const split=direction==='undo'?entry.changes.length-done:done;
    const completed=direction==='undo'?entry.changes.slice(split):entry.changes.slice(0,split);
    const remaining=direction==='undo'?entry.changes.slice(0,split):entry.changes.slice(split);
    if(remaining.length)source.push({label:entry.label,changes:remaining});
    destination.push({label:entry.label,changes:completed});
   }
  }
  return {changed:done,complete:done===ordered.length};
 }
}
