import type {EditorSnapshot,SymbolEntry} from './types.ts';
import {graphemes,safeSelection} from './unicode.ts';
export function insertSymbol(current:EditorSnapshot,entry:Pick<SymbolEntry,'insertText'|'insertionMode'|'prefix'|'suffix'>):EditorSnapshot {
  const {start,end}=safeSelection(current.text,current.start,current.end);const selected=current.text.slice(start,end);
  let value=entry.insertText,caret=start+value.length,selectionEnd=caret;
  if(entry.insertionMode==='paired-span'){
    const before=entry.prefix??'',after=entry.suffix??'';
    value=before+selected+after;caret=start+before.length;selectionEnd=caret+selected.length;
  }else if(entry.insertionMode==='bridge'&&selected){
    const units=graphemes(selected);if(units.length!==2)throw Error('连线操作需要恰好选中两个字素（字母及其附加符号）。');
    value=units[0]!.text+entry.insertText+units[1]!.text;caret=start+value.length;selectionEnd=caret;
  }
  return {text:current.text.slice(0,start)+value+current.text.slice(end),start:caret,end:selectionEnd};
}
/** One history owns native input and symbol commands. IME commits are one record. */
export class EditorHistory {
  private past:EditorSnapshot[]=[];private future:EditorSnapshot[]=[];
  current:EditorSnapshot;
  constructor(initial:EditorSnapshot={text:'',start:0,end:0}){this.current={...initial};}
  get canUndo(){return this.past.length>0;}get canRedo(){return this.future.length>0;}
  select(start:number,end:number){this.current={...this.current,start,end};}
  commit(next:EditorSnapshot){
    if(next.text===this.current.text){this.current={...next};return false;}
    this.past.push({...this.current});this.future=[];this.current={...next};
    // Cap retained text, never silently truncate the user's current document.
    while(this.past.length>120||this.past.reduce((n,s)=>n+s.text.length,0)>8_000_000)this.past.shift();
    return true;
  }
  undo(){const previous=this.past.pop();if(!previous)return null;this.future.push({...this.current});this.current=previous;return {...previous};}
  redo(){const next=this.future.pop();if(!next)return null;this.past.push({...this.current});this.current=next;return {...next};}
}
