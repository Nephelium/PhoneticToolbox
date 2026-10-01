type Key = Pick<KeyboardEvent,'key'|'ctrlKey'|'metaKey'|'altKey'|'shiftKey'|'isComposing'>;
export function annotationShortcut(event:Key){
  if(event.isComposing||event.altKey||event.shiftKey||!(event.ctrlKey||event.metaKey))return;
  return ({c:'copy',x:'cut',v:'paste',z:'undo',s:'save'} as const)[event.key.toLowerCase() as 'c'|'x'|'v'|'z'|'s'];
}
