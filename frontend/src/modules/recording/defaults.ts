import type {Config,Device} from './types.ts';
export function recordingDefaults():Config{return {device:'',sample_rate:48000,channels:2,roles:['microphone','microphone'],gain_db:0};}
export function bindInput(config:Config,device:Device):{config:Config;message:string}{
 const channels=Math.min(2,device.inputs);
 if(channels<1)throw Error('该设备没有音频输入');
 return {config:{...config,device:device.id,channels,roles:Array.from({length:channels},()=> 'microphone')},message:channels===1?'该设备仅有 1 路输入，已切为单声道音频。':'已选择双声道音频；若接入 EGG，请主动修改对应物理输入的角色。'};
}
