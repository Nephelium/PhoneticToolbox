import {uiMessage,uiErrorMessage} from '../module-localization.ts';
import {messageText} from '../../i18n/index.ts';
import {tr} from '../../i18n/index.ts';
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

export function deviceAccessErrorMessage(error:unknown){
 const e=error as Error;
 if(e.name==='NotAllowedError'||e.name==='SecurityError')return uiMessage("modules.lip-extraction.microphone_permission_was_not_granted_check_system_or_browser");
 if(e.name==='NotFoundError')return uiMessage("modules.lip-extraction.no_audio_input_found_connect_a_microphone_or_sound");
 if(e.name==='NotReadableError')return uiMessage("modules.lip-extraction.cannot_read_microphone_check_whether_another_program_is_using");
 if(e.name==='AbortError'||e.name==='InvalidStateError')return uiMessage("modules.lip-extraction.audio_device_reading_was_interrupted_click_authorize_and_refresh");
 return e.message?uiErrorMessage(e):uiMessage("modules.lip-extraction.failed_to_read_devices_refresh_devices_again");
}

export const deviceAccessError=(error:unknown):string=>messageText(deviceAccessErrorMessage(error));
