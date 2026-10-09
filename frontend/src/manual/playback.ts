import {watch} from 'vue';
import {pause,playback} from '../state/audio.ts';
import {assertPlaybackAllowed,captureOwner} from '../platform/capture-lease.ts';
import {ManualMediaCoordinator} from './media.ts';

let channel:BroadcastChannel|null=null;
export const manualPlayback=new ManualMediaCoordinator({
  assertAllowed:()=>{assertPlaybackAllowed();if(captureOwner())throw Error('唇形采集正在进行，请先停止采集后再试听说明书。');},
  pauseWorkbench:pause,
  announce:()=>channel?.postMessage('pause-other-windows')
});
// This module is browser-only; the pure coordinator remains independently testable.
if(typeof window!=='undefined'&&typeof BroadcastChannel!=='undefined'){
  channel=new BroadcastChannel('ptb-v3-audio');channel.onmessage=()=>manualPlayback.pause();
}
watch(()=>playback.playing,playing=>{if(playing)manualPlayback.pause();},{flush:'sync'});
