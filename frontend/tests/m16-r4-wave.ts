// Browser-only layout fixture. Import through Vite so all components share Vue.
import {createApp,h,ref} from 'vue';
import RecordingWave from '../src/modules/recording/RecordingWave.vue';
export function mountEightChannels(){
 const all=ref(true),spectrum=ref(true),element=document.createElement('div');
 element.id='m16-eight';element.style.cssText='position:fixed;inset:20px;z-index:1000;height:400px;background:white';
 document.body.append(element);
 const preview={wave:Array.from({length:8},()=>[[0,.2],[-.4,.4],[0,.1]]),window_frames:48000,frames:48000,start_frame:0,sample_rate:48000,spectrum:null};
 createApp({render:()=>h(RecordingWave,{preview,roles:Array(8).fill('microphone'),selection:[0,0],spectrum:spectrum.value,allChannels:all.value,disabled:false,scale:1})}).mount(element);
 return {setAll:(value:boolean)=>all.value=value,setSpectrum:(value:boolean)=>spectrum.value=value};
}
