import {createApp,h,nextTick,ref} from 'vue';
import ManualReader from '../../src/manual/ManualReader.vue';
import ManualDocument from '../../src/manual/ManualDocument.vue';
import {manualPlayback} from '../../src/manual/playback.ts';
import type {ManualChapter,ManualLocation,ManualProject,ManualTarget} from '../../src/manual/types.ts';

/** Browser-only fixture: controlled requests exercise cancellation and project changes. */
export async function verifyManualNavigation(){
  const check=(value:unknown,message:string)=>{if(!value)throw Error(message);};
  const wait=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
  async function until(test:()=>boolean){const end=performance.now()+5000;while(!test()){check(performance.now()<end,'fixture wait timed out');await wait(10);}}
  const project=ref<ManualProject>({schemaVersion:'ptb-manual/1',id:'qa',version:'1',title:'Navigation QA',assets:[],chapters:['a','b'].map(id=>({id,title:id,path:id+'.json',sections:[{id:'one',title:'One',level:2},{id:'two',title:'Two',level:2}]}))});
  const reader=ref<{navigate:(target:ManualTarget)=>Promise<void>}>();
  const requests:{id:string;resolve:(chapter:ManualChapter)=>void}[]=[],locations:ManualLocation[]=[];
  const content=(id:string,title:string):ManualChapter=>({schemaVersion:'ptb-manual-chapter/1',id,title,body:{type:'doc',content:['one','two'].flatMap(section=>[
    {type:'heading',attrs:{id:section,level:2},content:[{type:'text',text:section}]},
    ...Array.from({length:20},()=>({type:'paragraph',content:[{type:'text',text:'Fixture paragraph for scroll restoration.'}]}))
  ])}});
  const container=document.createElement('div');container.style.cssText='position:fixed;inset:0;z-index:99999;background:white';document.body.append(container);
  const app=createApp({render:()=>h(ManualReader,{ref:reader,project:project.value,chapterLoader:descriptor=>new Promise(resolve=>requests.push({id:descriptor.id,resolve})),onLocation:(location:ManualLocation)=>locations.push(location)})});
  let mounted=false;
  const checks:string[]=[];
  try{
    app.mount(container);mounted=true;
    await until(()=>requests.length===1);requests[0].resolve(content('a','A version one'));
    await until(()=>!!container.querySelector('.manual-document[data-manual-chapter=a]'));
    const original=container.querySelector('.manual-document');
    await reader.value!.navigate({chapterId:'a',targetId:'two'});
    check(requests.length===1&&container.querySelector('.manual-document')===original,'same-chapter jump reloaded or remounted the chapter');
    checks.push('same chapter keeps its document and avoids the loader');

    project.value={...project.value,version:'2'};
    await until(()=>requests.length===2);requests[1].resolve(content('a','A version two'));
    await until(()=>!!container.querySelector('.manual-document')?.textContent?.includes('A version two'));
    check(container.querySelector('.manual-document')!==original,'new project reused stale chapter content');
    checks.push('project revision invalidates the same-chapter fast path');

    const pending=reader.value!.navigate({chapterId:'b'});
    await until(()=>requests.length===3);
    await reader.value!.navigate({chapterId:'a',targetId:'one'});
    requests[2].resolve(content('b','Late B response'));await pending;await nextTick();
    check(container.querySelector('.manual-document')?.getAttribute('data-manual-chapter')==='a','late chapter request replaced the current chapter');
    checks.push('returning to the cached chapter cancels a late different-chapter response');

    await Promise.all([reader.value!.navigate({chapterId:'a',targetId:'one'}),reader.value!.navigate({chapterId:'a',targetId:'two'})]);
    check(locations.at(-1)?.targetId==='two','rapid same-chapter jumps saved an outdated target');
    checks.push('rapid same-chapter requests keep the latest target');

    const sc=container.querySelector<HTMLElement>('.manual-reading-scroll')!;
    sc.scrollTop=240;sc.dispatchEvent(new Event('scroll'));await wait(40);
    const position=sc.scrollTop;app.unmount();mounted=false;
    check(locations.at(-1)?.scrollTop===position,'unmount lost the latest unflushed scroll position');
    const count=locations.length;await wait(300);
    check(locations.length===count,'unmounted reader emitted a delayed location');
    checks.push('unmount flushes the latest scroll and cancels pending writes');
    return checks;
  }finally{if(mounted)app.unmount();container.remove();}
}

/** Changing reader visibility must still stop native media without replacing text. */
export async function verifyManualPlaybackContext(){
  const allowed=ref(true),container=document.createElement('div');document.body.append(container);
  const chapter:ManualChapter={schemaVersion:'ptb-manual-chapter/1',id:'media-qa',title:'Media QA',body:{type:'doc',content:[
    {type:'paragraph',content:[{type:'text',text:'Stable chapter text'}]},
    {type:'audio',attrs:{assetId:'qa-audio'}}
  ]}};
  const source=URL.createObjectURL(new Blob([],{type:'audio/wav'}));
  const app=createApp({render:()=>h(ManualDocument,{chapter,assets:[{id:'qa-audio',kind:'audio',path:'qa.wav',distribution:'public'}],assetResolver:()=>source,playbackAllowed:allowed.value})});
  try{
    app.mount(container);await nextTick();
    const text=container.querySelector('p'),media=container.querySelector('audio')!;
    let pauses=0;media.pause=()=>{pauses++;};
    media.dispatchEvent(new Event('play'));allowed.value=false;await nextTick();
    if(pauses!==1||container.querySelector('audio')!==media||container.querySelector('p')!==text)throw Error('reader visibility failed to stop and preserve its mounted media/text');
    media.dispatchEvent(new Event('play'));await nextTick();
    if(pauses!==2||!container.textContent?.includes('当前任务正在使用音频'))throw Error('inactive reader allowed native media playback');
    allowed.value=true;await nextTick();media.dispatchEvent(new Event('play'));await nextTick();
    if(pauses!==2||container.textContent?.includes('当前任务正在使用音频'))throw Error('reactivated reader failed to restore playback permission');
    return ['reader visibility pauses its native media and preserves mounted content','inactive media rejects play and active media regains permission'];
  }finally{manualPlayback.pause();app.unmount();container.remove();URL.revokeObjectURL(source);}
}
