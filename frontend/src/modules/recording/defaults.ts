import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr,type MessageDescriptor} from '../../i18n/index.ts';
import type {Config,Device} from './types.ts';
export function recordingDefaults():Config{return {device:'',sample_rate:48000,channels:2,roles:['microphone','microphone'],gain_db:0};}
export function recordingSettings(config:Config){const {device,sample_rate,channels,roles,gain_db}=config;return {device,sample_rate,channels,roles,gain_db};}
export function bindInput(config:Config,device:Device):{config:Config;message:string;messageDisplay:MessageDescriptor}{
 const channels=Math.min(2,device.inputs);
 if(channels<1)throw displayError(displayMessage('modules.recording.text.d70b8edb1338'));
 const key=channels===1?'modules.recording.text.6f3ba9a3c8fa':'modules.recording.text.e53dcc8a7f83';
 return {config:{...config,device:device.id,channels,roles:Array.from({length:channels},()=> 'microphone')},get message(){return tr(key);},messageDisplay:displayMessage(key)};
}
