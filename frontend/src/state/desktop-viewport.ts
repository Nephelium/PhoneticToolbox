/** The server workbench keeps desktop geometry on handheld browsers. */
export interface ViewportDevice {userAgent:string;platform:string;touchPoints:number;coarsePointer:boolean;finePointer:boolean;screenWidth:number;}
export function workbenchViewportWidth(device:ViewportDevice):number|null {
 const mobile=/Android|iPhone|iPad|iPod/i.test(device.userAgent);
 const ipad=device.platform==='MacIntel'&&device.touchPoints>1;
 const touchTablet=device.touchPoints>0&&device.coarsePointer&&!device.finePointer;
 return mobile||ipad||touchTablet?Math.max(1440,Math.round(device.screenWidth)||0):null;
}
function browserDevice():ViewportDevice {
 return {userAgent:navigator.userAgent,platform:navigator.platform,touchPoints:navigator.maxTouchPoints,coarsePointer:matchMedia('(pointer: coarse)').matches,finePointer:matchMedia('(any-pointer: fine)').matches,screenWidth:screen.width};
}
/** Also covers iPad's desktop Safari UA and attached trackpads. */
export function isHandheldBrowser():boolean {return workbenchViewportWidth(browserDevice())!==null;}
export function installDesktopViewport(){
 const meta=document.querySelector<HTMLMetaElement>('meta[name="viewport"]');if(!meta)return ()=>{};
 const original=meta.content;let applied='';
 function update(){
  const width=workbenchViewportWidth(browserDevice());
  if(width===null)return;
  // Let the browser fit the desktop canvas initially. No maximum/user-scalable
  // restriction: users can pinch to inspect controls at their preferred size.
  const content='width='+width;if(meta!.content!==content)meta!.content=content;applied=content;
  document.documentElement.dataset.workbenchLayout='desktop';
 }
 update();window.addEventListener('orientationchange',update);window.addEventListener('resize',update);
 return ()=>{window.removeEventListener('orientationchange',update);window.removeEventListener('resize',update);if(applied&&meta.content===applied)meta.content=original;delete document.documentElement.dataset.workbenchLayout;};
}
