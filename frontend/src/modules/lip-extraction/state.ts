export type CaptureMode='preview'|'realtime'|'raw'|'record_then_analyze';
export type CapturePhase='idle'|'opening'|'previewing'|'recording'|'stopping'|'finalizing'|'ready'|'failed';
export interface CaptureStats {presented:number;processed:number;detected:number;skippedInference:number;observedPresentationGaps:number;encodedBytes:number;started:number;stopped:number|null;inferenceMs:number;hiddenEvents:number;firstPresentedMs:number|null;lastPresentedMs:number|null;firstOverlayMs:number|null;lastOverlayMs:number|null;}
export function newStats():CaptureStats{return {presented:0,processed:0,detected:0,skippedInference:0,observedPresentationGaps:0,encodedBytes:0,started:0,stopped:null,inferenceMs:0,hiddenEvents:0,firstPresentedMs:null,lastPresentedMs:null,firstOverlayMs:null,lastOverlayMs:null};}
export function offsetChoice(action:'apply'|'save_without_offset'|'cancel',value:number):number|null{
 if(action==='cancel')return null;if(!Number.isFinite(value)||Math.abs(value)>2)throw Error('offset 必须在 ±2 秒内');return action==='apply'?value:0;
}
export function frameRate(count:number,first:number,last:number):number|null{return count>1&&last>first?(count-1)*1000/(last-first):null;}
export const MAX_LOCAL_RECORDING_BYTES=128_000_000;
export const MAX_LOCAL_RESULT_BYTES=32_000_000;
export const MAX_RECORDING_SECONDS=1800;
export function verifyLocalBudget(bytes:number,additional:number,limit=MAX_LOCAL_RECORDING_BYTES){if(!Number.isSafeInteger(additional)||additional<0||bytes+additional>limit)throw Error('本地录制达到内存预算，已停止。请保存现有录制，长录制使用桌面文件路径。');}
export function preserveDeviceSelection(previous:string,devices:MediaDeviceInfo[],kind:MediaDeviceKind):string{
 return previous&&devices.some(d=>d.kind===kind&&d.deviceId===previous)?previous:previous||'';
}
