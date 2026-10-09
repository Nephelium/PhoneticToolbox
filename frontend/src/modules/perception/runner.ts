import {copy,uid,roles,keys,keyRange,isCleanKey,type Session,type Attempt,type Role} from './model.ts';
import {MediaBank,timing,type Prepared} from './media.ts';
import {LocalStore,lockSession} from './storage.ts';
export type Phase='intro'|'notice'|'interstitial'|'ready'|'preparing'|'playing'|'responding'|'saving'|'paused'|'saving-error'|'completed';
export class Runner {
 phase:Phase='intro';message='';visual:Prepared|null=null;activeRole:Role|null=null;session:Session;
 private release:(()=>Promise<void>)|null=null;private epoch=0;private timer:ReturnType<typeof setTimeout>|null=null;private nodes:AudioScheduledSourceNode[]=[];private current:Attempt|null=null;private held=new Set<string>();private pending=false;private saveQueue:Promise<boolean>=Promise.resolve(true);private ending:Promise<boolean>|null=null;
 readonly bank:MediaBank;readonly store:LocalStore;private changed:()=>void;
 constructor(session:Session,bank:MediaBank,store:LocalStore,changed:()=>void){this.session=copy(session);this.bank=bank;this.store=store;this.changed=changed;}
 async acquire(){this.release=await lockSession(this.session.id);}
 get focused(){return !['paused','saving-error','completed','intro'].includes(this.phase);}
 private set(p:Phase){this.phase=p;this.changed();}
 private stop(){this.epoch++;if(this.timer)clearTimeout(this.timer);for(const n of this.nodes)try{n.stop();n.disconnect();}catch{}this.nodes=[];this.activeRole=null;}
 event(kind:string,detail=''){this.session.events.push({kind,detail,at:new Date().toISOString(),perfMs:performance.now(),timeOrigin:performance.timeOrigin});}
 get terminal(){return this.session.status==='completed'||this.session.status==='ended';}
 get canConfirmExport(){return this.session.exportRequestedRevision===this.session.revision&&this.session.exportConfirmedRevision!==this.session.revision;}
 private save(metadata?:'request'|'confirm'){const operation=this.saveQueue.then(async()=>{try{
  // Export bookkeeping is the only write that leaves the exported data unchanged.
  if(metadata==='request'){this.session.exportRequestedRevision=this.session.revision+1;this.session.exportConfirmedRevision=null;}
  else if(metadata==='confirm'){if(!this.canConfirmExport)throw Error('结果已更新或尚未导出，请重新导出当前结果后确认');this.session.exportRequestedRevision=this.session.revision+1;this.session.exportConfirmedRevision=this.session.revision+1;}
  else{this.session.exportRequestedRevision=null;this.session.exportConfirmedRevision=null;}
  await this.store.commit(this.session);this.changed();return true;
 }catch(e){this.message=String(e);this.stop();this.set('saving-error');return false;}});this.saveQueue=operation;return operation;}
 async retrySave(){if(this.phase!=='saving-error'||this.pending)return;this.pending=true;const ok=await this.save();this.pending=false;if(ok){this.message='保存已恢复，请显式继续';this.set(this.terminal?'completed':'paused');}}
 async start(){if(this.terminal)return;if(this.phase!=='intro'&&this.phase!=='paused')return;if(this.session.nextIndex>=this.session.project.trials.length){this.set('completed');return;}this.message='';this.visual=null;const i=this.session.nextIndex,r=keyRange(this.session.project,i);this.set(r?.start===i+1&&r.notice.trim()?'notice':'interstitial');this.scheduleInterstitial();}
 private scheduleInterstitial(){if(this.phase==='interstitial'&&this.session.project.config.advanceMode==='auto'){this.timer=setTimeout(()=>{if(this.phase==='interstitial')this.enter();},this.session.project.config.interTrialInterval);}}
 continue(){if(this.phase==='notice'){this.set('interstitial');this.scheduleInterstitial();}else if(this.phase==='interstitial'&&this.session.project.config.advanceMode==='manual')this.enter();else if(this.phase==='ready')void this.begin();}
 private enter(){if(this.session.project.config.autoPlay)void this.begin();else this.set('ready');}
 async begin(){if(!['interstitial','ready'].includes(this.phase))return;this.stop();const epoch=this.epoch;this.visual=null;this.set('preparing');const s=this.session,i=s.nextIndex,p=s.project;
 const a:Attempt={id:uid(),trialIndex:i,attempt:s.attempts.filter(a=>a.trialIndex===i).length+1,status:'running',roles:{},order:roles(p.paradigm),key:null,keyLabel:null,rtMs:null,rtDefinition:'response-open-to-handler-performance-ms/v1',timing:null,reason:null,startedAt:new Date().toISOString(),finishedAt:null};
 for(const r of a.order){const asset=p.assets.find(x=>x.id===p.trials[i].stimuli[r]?.id);if(asset)a.roles[r]=copy(asset);}s.attempts.push(a);this.current=a;s.status='running';this.pending=true;const saved=await this.save();this.pending=false;if(!saved||epoch!==this.epoch)return;
 try{if(this.bank.context.state!=='running')throw Error('音频 context 未运行');const prepared=await this.bank.trial(p,i);if(epoch!==this.epoch)return;
 a.timing=timing(this.bank.context);for(const {role,media} of prepared)a.roles[role]=copy(media.asset);
 const ctx=this.bank.context;let cursor=ctx.currentTime+.05;
 if(p.config.useBeep){const osc=ctx.createOscillator(),gain=ctx.createGain();osc.frequency.value=600;gain.gain.setValueAtTime(1,cursor);gain.gain.exponentialRampToValueAtTime(.00001,cursor+.15);osc.connect(gain).connect(ctx.destination);osc.start(cursor);osc.stop(cursor+.15);osc.onended=()=>{osc.disconnect();gain.disconnect();};this.nodes.push(osc);cursor+=.5;}
 this.set('playing');
 if(prepared[0].media.buffer){for(const [n,{role,media}] of prepared.entries()){if(!media.buffer)throw Error('混合媒体序列不兼容');if(n)cursor+=p.config.isi/1000;const node=ctx.createBufferSource();node.buffer=media.buffer;node.connect(ctx.destination);const stamp=a.timing.outputTimestamp;a.timing.planned.push({role,startAudioSeconds:cursor,endAudioSeconds:cursor+media.buffer.duration,scheduledAtAudioSeconds:ctx.currentTime,scheduledAtPerfMs:performance.now(),scheduleLateByMs:Math.max(0,(ctx.currentTime-cursor)*1000),outputStartEstimatePerfMs:stamp?stamp.performanceTime+(cursor-stamp.contextTime)*1000:null});node.onended=()=>{a.timing?.observedEnded.push({role,performanceMs:performance.now(),audioSeconds:ctx.currentTime});node.disconnect();};node.start(cursor);this.nodes.push(node);cursor+=media.buffer.duration;}
  const poll=()=>{if(epoch!==this.epoch)return;const now=ctx.currentTime;this.activeRole=a.timing!.planned.find(x=>now>=x.startAudioSeconds&&now<x.endAudioSeconds)?.role??null;if(now>=cursor){a.timing!.endDetectedPerfMs=performance.now();const m=a.timing!.audioMapping;this.openResponse();a.timing!.responseDelayFromMappedEndMs=a.timing!.responseOpenPerfMs!-(m.performanceMs+(cursor-m.audioSeconds)*1000);}else{this.changed();this.timer=setTimeout(poll,8);}};poll();
 }else{const show=()=>{if(epoch!==this.epoch)return;if(ctx.currentTime<cursor){this.timer=setTimeout(show,8);return;}this.visual=prepared[0].media;this.changed();requestAnimationFrame(t=>{if(epoch!==this.epoch)return;a.timing!.visualFramePerfMs=t;requestAnimationFrame(()=>{if(epoch===this.epoch)this.openResponse();});});};show();}
 }catch(e){if(epoch===this.epoch)await this.interrupt('playback-or-decode-failure',String(e),'invalid');}
 }
 private openResponse(){if(!this.current?.timing)return;this.current.timing.responseOpenPerfMs=performance.now();this.activeRole=null;this.set('responding');}
 keyup(e:KeyboardEvent){this.held.delete(e.code||e.key.toLowerCase());}
 keydown(e:KeyboardEvent){const code=e.code||e.key.toLowerCase(),held=this.held.has(code);this.held.add(code);if(held||!isCleanKey(e))return;
 if(['notice','interstitial','ready'].includes(this.phase)){this.continue();return;}if(this.phase!=='responding'||!this.current?.timing)return;
 const choice=keys(this.session.project,this.session.nextIndex).find(k=>k.key.toLowerCase()===e.key.toLowerCase());if(!choice)return;const now=performance.now(),a=this.current;
 // Reject events queued before the response window opened. Epoch-based timestamps are recorded as unavailable.
 const eventTime=Number.isFinite(e.timeStamp)&&e.timeStamp>=0&&e.timeStamp<now+1000?e.timeStamp:null;if(eventTime!==null&&eventTime<a.timing!.responseOpenPerfMs!)return;
 this.set('saving');a.timing!.keyEventPerfMs=eventTime;a.timing!.keyHandledPerfMs=now;a.key=choice.key.toLowerCase();a.keyLabel=choice.label;a.rtMs=now-a.timing!.responseOpenPerfMs!;a.status='completed';a.finishedAt=new Date().toISOString();void this.finishTrial();
 }
 private async finishTrial(){this.stop();const epoch=this.epoch;this.pending=true;this.session.nextIndex++;this.session.status=this.session.nextIndex===this.session.project.trials.length?'completed':'ready';const saved=await this.save();this.pending=false;this.current=null;if(!saved||epoch!==this.epoch)return;if(this.session.status==='completed')this.set('completed');else{this.set('paused');await this.start();}}
 async interrupt(kind:string,detail='',status:'interrupted'|'invalid'='interrupted'){if(this.phase==='completed')return true;this.event(kind,detail);this.stop();this.held.clear();this.message=detail||kind;
 if(this.current?.status==='running'){this.current.status=status;this.current.reason=kind;this.current.finishedAt=new Date().toISOString();this.current.rtMs=null;}
 if(!this.terminal)this.session.status='paused';this.set('paused');const saved=await this.save();if(saved&&this.terminal)this.set('completed');return saved;
 }
 async end(){if(this.ending)return this.ending;if(this.terminal){const saved=await this.saveQueue;return saved&&this.phase==='completed';}
  this.event('explicit-end','实验已结束；保留部分结果');this.stop();this.held.clear();
  if(this.current?.status==='running'){this.current.status='interrupted';this.current.reason='explicit-end';this.current.finishedAt=new Date().toISOString();this.current.rtMs=null;}
  this.session.status='ended';this.set('saving');this.ending=this.save().then(saved=>{if(saved)this.set('completed');return saved;});
  try{return await this.ending;}finally{this.ending=null;}
 }
 async exportRequested(){return this.save('request');}
 async confirmExport(){if(!this.canConfirmExport)throw Error('结果已更新或尚未导出，请重新导出当前结果后确认');return this.save('confirm');}
 async dispose(force=false){let saved=await this.saveQueue;if(!['completed','intro','saving-error'].includes(this.phase))saved=await this.interrupt('leave-runner');
  if((!saved||this.phase==='saving-error')&&!force)return false;this.stop();await this.release?.();this.release=null;return true;
 }
}
