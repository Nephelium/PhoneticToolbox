// Audio selection buttons opt in; other controls keep native keyboard actions.
export function audioShortcutTarget(target:EventTarget|null){
 if(!(target instanceof Element))return true;
 if(target.closest('input,textarea,select,[contenteditable]'))return false;
 const control=target.closest('button,a,summary,[role="button"],[role="link"]');
 return !control||control.matches('button[data-audio-selection],[role="button"][data-audio-selection]');
}
export function spacePlaybackShortcut(event:KeyboardEvent,transport?:Element|null){
 const modal=[...document.querySelectorAll('dialog[open],[role="dialog"][aria-modal="true"]')].at(-1);
 // A dialog may contain its own audio, e.g. EGG inverse-filtering results.
 // Only that dialog's transport can respond while it is open.
 if(modal&&(!transport||!modal.contains(transport)||!(event.target instanceof Node)||!modal.contains(event.target)))return false;
 return event.code==='Space'&&!event.defaultPrevented&&!event.repeat&&!event.isComposing&&
  !event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey&&
  audioShortcutTarget(event.target);
}
