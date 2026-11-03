import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
/** One active input capture in this workbench. The owner releases only its lease. */
export type CaptureOwner='M05'|'M16';
let owner:CaptureOwner|undefined;
export function claimCapture(next:CaptureOwner):void {
  if(owner&&owner!==next)throw uiError(uiMessage('platform.value_is_using_the_input_device_stop_capture_first',{v0:(owner==='M05'?'唇形采集':'录音模块')}));
  owner=next;
}
export function releaseCapture(current:CaptureOwner):void {if(owner===current)owner=undefined;}
export function captureOwner():CaptureOwner|undefined {return owner;}
export function assertPlaybackAllowed():void {
  if(owner==='M16')throw uiError(uiMessage('platform.recording_or_input_testing_is_active_stop_it_before'));
}
