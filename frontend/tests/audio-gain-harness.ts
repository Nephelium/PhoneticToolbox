import {createApp,h} from 'vue';
import ManualMedia from '../src/manual/ManualMedia.vue';
export function mountMedia(url:string){
 const root=document.createElement('div');document.body.append(root);
 const app=createApp({render:()=>h(ManualMedia,{kind:'audio',assetId:'gain-test',assets:[{id:'gain-test',kind:'audio',path:'quiet.wav',title:'增益验证'}] as any,assetResolver:()=>url})});app.mount(root);
 const button=document.createElement('button');button.textContent='播放验证例音';button.onclick=()=>{void root.querySelector('audio')!.play();};root.append(button);
 return {unmount(){app.unmount();root.remove();URL.revokeObjectURL(url);}};
}
