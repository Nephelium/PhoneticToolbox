export function editableTarget(target:EventTarget|null){if(!(target instanceof Element))return false;return !!target.closest('input,textarea,select,[contenteditable="true"],[role="dialog"]');}
export function shortcut(event:KeyboardEvent,active:boolean,capturing:boolean,canvasFocused:boolean):string|null{
 if(!active||capturing||event.repeat||event.isComposing||editableTarget(event.target))return null;
 if(event.code==='Space'&&!event.ctrlKey&&!event.metaKey&&!event.altKey)return 'play';
 if(!canvasFocused)return null;
 const modifier=event.ctrlKey||event.metaKey,key=event.key.toLowerCase();
 if(modifier){if(key==='x')return 'cut';if(key==='c')return 'copy';if(key==='v')return 'paste';if(key==='z')return event.shiftKey?'redo':'undo';if(key==='y')return 'redo';if(key==='a')return 'all';}
 if(key==='delete'||key==='backspace')return 'delete';return null;
}
