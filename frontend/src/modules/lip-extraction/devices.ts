/** Unauthorised browsers expose an incomplete list with empty device names. */
export function audioDevicesHidden(devices:Pick<MediaDeviceInfo,'kind'|'label'>[]){
 return !devices.some(d=>d.kind==='audioinput'&&d.label);
}

/** Only an explicit user action may briefly open audio to reveal the list. */
export async function readCaptureDevices(media:Pick<MediaDevices,'enumerateDevices'|'getUserMedia'>,options:{authorizeAudio?:boolean;prepareAudio?:()=>Promise<void>;signal?:AbortSignal}={}){
 const listed=await media.enumerateDevices();
 options.signal?.throwIfAborted();
 if(!options.authorizeAudio||!audioDevicesHidden(listed))return listed;
 await options.prepareAudio?.();
 options.signal?.throwIfAborted();
 const stream=await media.getUserMedia({audio:true,video:false});
 try{
  options.signal?.throwIfAborted();
  return await media.enumerateDevices();
 }finally{stream.getTracks().forEach(track=>track.stop());}
}

export function deviceAccessError(error:unknown){
 const e=error as Error;
 if(e.name==='NotAllowedError'||e.name==='SecurityError')return '未获得麦克风权限。请检查系统或浏览器权限，再点击授权并刷新设备并选择允许。';
 if(e.name==='NotFoundError')return '未找到音频输入，请连接麦克风或声卡后重新刷新设备。';
 if(e.name==='NotReadableError')return '无法读取麦克风，请检查设备是否被其他程序占用后重试。';
 if(e.name==='AbortError'||e.name==='InvalidStateError')return '音频设备读取已中断，请重新点击授权并刷新设备。';
 return e.message||'读取设备失败，请重新刷新设备。';
}
