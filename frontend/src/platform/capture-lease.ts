/** One active input capture in this workbench. The owner releases only its lease. */
export type CaptureOwner='M05'|'M16';
let owner:CaptureOwner|undefined;
export function claimCapture(next:CaptureOwner):void {
  if(owner&&owner!==next)throw Error(`${owner==='M05'?'唇形采集':'录音模块'}正在使用输入设备，请先停止采集。`);
  owner=next;
}
export function releaseCapture(current:CaptureOwner):void {if(owner===current)owner=undefined;}
export function captureOwner():CaptureOwner|undefined {return owner;}
export function assertPlaybackAllowed():void {
  if(owner==='M16')throw Error('录音或录前检测正在进行，请先停止后再播放音频。');
}
