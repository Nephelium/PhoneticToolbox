import {tr} from '../i18n/index.ts';
/** Explicit playback ownership. Native media is coordinated with shared Web Audio. */
export interface ManualMedia {pause():void}
export interface PlaybackBridge {assertAllowed():void; pauseWorkbench():void; announce?():void}
export function manualPlaybackError(key:string):Error&{manualMessageKey:string}{return Object.assign(Error(tr(key)),{manualMessageKey:key});}
export class ManualMediaCoordinator {
  private active:ManualMedia|null=null;
  private bridge:PlaybackBridge;
  constructor(bridge:PlaybackBridge){this.bridge=bridge;}
  activate(media:ManualMedia,allowed=true):void {
    try{if(!allowed)throw manualPlaybackError('manual.reader.audioBusy');this.bridge.assertAllowed();}
    catch(error){media.pause();throw error;}
    if(this.active&&this.active!==media)this.active.pause();
    this.bridge.pauseWorkbench();this.active=media;this.bridge.announce?.();
  }
  release(media:ManualMedia):void {if(this.active===media)this.active=null;}
  pause():void {this.active?.pause();this.active=null;}
}
