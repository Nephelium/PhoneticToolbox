/** Synthetic inputs only. Install in an isolated verification page, never a user page. */
window.installM05R2Capture=async function(portrait){
 const image=new Image();image.src=portrait;await image.decode();
 const state=window.m05R2={requests:[],ended:0,created:0,permission:false,low:false,missing:false,deny:false};
 const info=(deviceId,label)=>({kind:'audioinput',deviceId,label,groupId:'qa',toJSON(){return this;}});
 navigator.mediaDevices.enumerateDevices=async()=>state.permission?[info('default','默认 - 立体声混音'),info('mix','立体声混音 (Realtek)'),info('mic','Microphone USB (QA)')]:[info('default','')];
 navigator.mediaDevices.getUserMedia=async constraints=>{
  state.requests.push(constraints);if(state.deny)throw new DOMException('controlled denial','NotAllowedError');state.permission=true;
  const stream=new MediaStream();
  const owned=(track,release)=>{state.created++;const stop=track.stop.bind(track);let done=false;track.stop=()=>{if(!done){done=true;release();state.ended++;}stop();};return track;};
  if(constraints.video){
   const canvas=document.createElement('canvas');canvas.width=960;canvas.height=540;const context=canvas.getContext('2d');context.drawImage(image,0,-80,960,960);
   const timer=setInterval(()=>context.drawImage(image,0,-80,960,960),33),track=canvas.captureStream(30).getVideoTracks()[0];stream.addTrack(owned(track,()=>clearInterval(timer)));
  }
  if(constraints.audio&&!state.missing){
   const id=constraints.audio.deviceId?.exact||'mix';
   if(!['mix','mic','default'].includes(id)){stream.getTracks().forEach(t=>t.stop());throw new DOMException('unavailable','OverconstrainedError');}
   const context=new AudioContext(),oscillator=context.createOscillator(),gain=context.createGain(),destination=context.createMediaStreamDestination();
   gain.gain.value=state.low?0.0001:0.1;oscillator.connect(gain);gain.connect(destination);oscillator.start();await context.resume();
   const track=destination.stream.getAudioTracks()[0],original=track.getSettings.bind(track);
   Object.defineProperty(track,'label',{value:id==='mic'?'Microphone USB (QA)':'立体声混音 (Realtek)'});
   track.getSettings=()=>({...original(),deviceId:id});
   stream.addTrack(owned(track,()=>{oscillator.stop();void context.close();}));
  }
  return stream;
 };
};
