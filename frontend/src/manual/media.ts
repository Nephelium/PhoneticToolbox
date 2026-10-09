/** Explicit playback ownership. Native media is coordinated with shared Web Audio. */
export interface ManualMedia {pause():void}
export interface PlaybackBridge {assertAllowed():void; pauseWorkbench():void; announce?():void}
export class ManualMediaCoordinator {
  private active:ManualMedia|null=null;
  private bridge:PlaybackBridge;
  constructor(bridge:PlaybackBridge){this.bridge=bridge;}
  activate(media:ManualMedia,allowed=true):void {
    try{if(!allowed)throw Error('当前任务正在使用音频，请结束或暂停后再试听说明书。');this.bridge.assertAllowed();}
    catch(error){media.pause();throw error;}
    if(this.active&&this.active!==media)this.active.pause();
    this.bridge.pauseWorkbench();this.active=media;this.bridge.announce?.();
  }
  release(media:ManualMedia):void {if(this.active===media)this.active=null;}
  pause():void {this.active?.pause();this.active=null;}
}
