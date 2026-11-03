// Five-route output comparison. The preview lives above transformed reveal cards.
(() => {
  'use strict';
  let active=null,anchor=null,timer=0;
  function position(){
    if(!active||!anchor)return;
    active.style.maxHeight='calc(100dvh - 24px)';
    const r=anchor.getBoundingClientRect(),w=active.offsetWidth,gap=14,margin=12;
    let x=r.right+gap;
    if(x+w>innerWidth-margin)x=r.left-w-gap;
    let y;
    if(x<margin){
      x=Math.min(innerWidth-w-margin,Math.max(margin,r.left));
      const below=innerHeight-r.bottom-margin-gap,above=r.top-margin-gap;
      const useBelow=below>=above;
      active.style.maxHeight=Math.max(100,useBelow?below:above)+'px';
      y=useBelow?r.bottom+gap:r.top-active.offsetHeight-gap;
    }else y=Math.max(margin,Math.min(r.top-30,innerHeight-active.offsetHeight-margin));
    active.style.left=x+'px';active.style.top=y+'px';
  }
  function close(){clearTimeout(timer);if(active)active.hidden=true;if(anchor)anchor.setAttribute('aria-expanded','false');active=null;anchor=null;}
  function deferClose(){clearTimeout(timer);timer=setTimeout(()=>{if(!active?.contains(document.activeElement)&&document.activeElement!==anchor)close();},160);}
  document.querySelectorAll('.next-example').forEach(card=>{
    const preview=card.querySelector('.next-spectrum-preview');if(!preview)return;
    // Escape ancestor transforms/overflow so viewport positioning remains exact.
    document.body.append(preview);
    card.querySelectorAll('[data-spectrum-route]').forEach(button=>{
      function show(){
        clearTimeout(timer);if(active!==preview)close();if(anchor&&anchor!==button)anchor.setAttribute('aria-expanded','false');
        active=preview;anchor=button;preview.hidden=false;button.setAttribute('aria-expanded','true');
        preview.querySelectorAll('[data-spectrum-kind]').forEach(row=>row.classList.toggle('is-active',row.dataset.spectrumKind===button.dataset.spectrumRoute));
        preview.querySelectorAll('img').forEach(img=>{img.loading='eager';});position();
      }
      button.addEventListener('pointerenter',show);button.addEventListener('focus',show);
      button.addEventListener('pointerleave',deferClose);button.addEventListener('blur',deferClose);
    });
    preview.addEventListener('pointerenter',()=>clearTimeout(timer));preview.addEventListener('pointerleave',deferClose);
    preview.addEventListener('focusin',()=>clearTimeout(timer));preview.addEventListener('focusout',deferClose);
    preview.querySelector('.next-spectrum-close').addEventListener('click',close);
  });
  addEventListener('keydown',event=>{if(event.key==='Escape'&&active){event.preventDefault();close();}});
  addEventListener('pointerdown',event=>{if(active&&!active.contains(event.target)&&event.target!==anchor&&!anchor.contains(event.target))close();});
  addEventListener('scroll',position,{passive:true});addEventListener('resize',position,{passive:true});
})();

(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const hero = document.querySelector('.hero');
  const canvas = document.getElementById('hero-spectrum');
  // Deterministic illustrative texture: deliberately has no scientific axes/data labels.
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const w=canvas.width,h=canvas.height,pixels=ctx.createImageData(w,h);
      for(let x=0;x<w;x++)for(let y=0;y<h;y++){
        const t=x/w,f=1-y/h;
        const lines=[.13+.045*Math.sin(t*8),.33+.08*Math.sin(t*5+1),.59+.06*Math.cos(t*7),.8+.03*Math.sin(t*9)];
        let energy=0;for(let i=0;i<lines.length;i++)energy+=Math.exp(-Math.pow((f-lines[i])/(.024+i*.007),2))*(1-i*.13);
        const grain=(Math.sin(x*12.9898+y*78.233)*43758.5453)%1;
        energy=Math.min(1,Math.max(0,energy*(.63+.37*Math.pow(Math.cos(t*310),2))+.06*grain));
        const index=(y*w+x)*4;pixels.data[index]=240-energy*125;pixels.data[index+1]=231-energy*151;pixels.data[index+2]=255-energy*39;pixels.data[index+3]=230;
      }
      ctx.putImageData(pixels,0,0);
    }
  }
  const wave=document.getElementById('annotation-path');
  if(wave){let d='';for(let x=0;x<480;x++){const t=x/480;const envelope=Math.pow(Math.sin(Math.PI*t),2)*(.5+.4*Math.sin(t*21)**2);const y=48+envelope*(Math.sin(t*270)+.42*Math.sin(t*470))*31;d+=(x?'L':'M')+x+' '+y.toFixed(2);}wave.setAttribute('d',d);}

  let pointerFrame=0;
  hero?.addEventListener('pointermove',event=>{if(reduced.matches||event.pointerType==='touch')return;cancelAnimationFrame(pointerFrame);pointerFrame=requestAnimationFrame(()=>{const r=hero.getBoundingClientRect();hero.style.setProperty('--pointer-x',((event.clientX-r.left)/r.width-.5)*2);hero.style.setProperty('--pointer-y',((event.clientY-r.top)/r.height-.5)*2);});},{passive:true});
  function resetPointer(){cancelAnimationFrame(pointerFrame);hero?.style.setProperty('--pointer-x',0);hero?.style.setProperty('--pointer-y',0);}
  hero?.addEventListener('pointerleave',resetPointer);
  reduced.addEventListener('change',resetPointer);

  const stages=[...document.querySelectorAll('.screen-stage')];
  let frame=0;
  function positionScreens(){
    frame=0;if(reduced.matches)return;
    const height=innerHeight;
    for(const stage of stages){const r=stage.getBoundingClientRect();if(r.bottom<0||r.top>height)continue;const distance=Math.abs((r.top+r.height/2-height/2)/height);stage.style.setProperty('--angle',Math.min(1.25,.16+distance*1.35).toFixed(3));}
  }
  function queuePosition(){if(!frame)frame=requestAnimationFrame(positionScreens);}
  // Small groups enter in order; the page stays readable without animation support.
  document.querySelectorAll('.chapter-divider,.faq-list details,.site-stats').forEach(e=>e.classList.add('reveal'));
  for(const selector of ['.reference-grid','.lab-grid','.faq-list','.contact-cards','.upcoming-pipeline']){
    const group=document.querySelector(selector);if(!group)continue;
    [...group.children].forEach((e,i)=>{e.classList.add('reveal');e.style.setProperty('--reveal-delay',Math.min(i,3)*65+'ms');});
  }
  // Pause ambient motion when its illustration is offscreen or the tab is hidden.
  let heroVisible=true;
  function syncAmbient(){hero?.classList.toggle('motion-paused',document.hidden||!heroVisible||reduced.matches);}
  if(hero&&'IntersectionObserver' in window){const visibility=new IntersectionObserver(entries=>{heroVisible=entries[0].isIntersecting;syncAmbient();});visibility.observe(hero);}
  document.addEventListener('visibilitychange',syncAmbient);reduced.addEventListener('change',syncAmbient);syncAmbient();
  // Content remains visible without JS, unsupported observers, or reduced motion.
  if('IntersectionObserver' in window&&!reduced.matches){
    const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('is-visible');observer.unobserve(entry.target);}},{threshold:.07,rootMargin:'0px 0px -20px 0px'});
    document.documentElement.classList.add('motion-ready');document.querySelectorAll('.reveal').forEach(e=>observer.observe(e));
    reduced.addEventListener('change',e=>{if(e.matches){document.documentElement.classList.remove('motion-ready');observer.disconnect();}queuePosition();});
  }
  addEventListener('scroll',queuePosition,{passive:true});addEventListener('resize',queuePosition,{passive:true});queuePosition();

  const dialog=document.getElementById('screen-viewer'),viewer=document.getElementById('viewer-image'),caption=document.getElementById('viewer-caption');
  let opener=null;
  document.querySelectorAll('[data-screen]').forEach(link=>link.addEventListener('click',event=>{
    if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||typeof dialog?.showModal!=='function')return;
    event.preventDefault();opener=link;viewer.src=link.href;viewer.alt=link.querySelector('img').alt;caption.textContent=link.dataset.caption;dialog.showModal();document.body.style.overflow='hidden';
  }));
  dialog?.querySelector('.viewer-close').addEventListener('click',()=>dialog.close());
  dialog?.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
  dialog?.addEventListener('close',()=>{document.body.style.overflow='';opener?.focus({preventScroll:true});});

  const sounds=[...document.querySelectorAll('.sound-option audio')];
  const nativeVideos=[...document.querySelectorAll('.vocal-tract-demo video,.release-video video')],media=[...sounds,...nativeVideos];
  const pauseOthers=active=>media.forEach(other=>{if(other!==active&&!other.paused)other.pause();});
  const defaultSoundStatus=document.querySelector('.audio-status');
  const statusFor=audio=>audio.closest('.upcoming-section')?.querySelector('.audio-status')||defaultSoundStatus;
  if(sounds.length){
    document.documentElement.classList.add('audio-ready');
    for(const audio of sounds){
      const option=audio.closest('.sound-option'),button=option.querySelector('.sound-play');
      const label=audio.getAttribute('aria-label');
      const soundStatus=statusFor(audio);
      const playbackVolume=Number(audio.dataset.playbackVolume);
      if(Number.isFinite(playbackVolume)&&playbackVolume>0&&playbackVolume<=1)audio.volume=playbackVolume;
      function show(playing){button.setAttribute('aria-pressed',String(playing));button.setAttribute('aria-label',(playing?"Pause":"Play")+label);button.querySelector('.sound-symbol').textContent=playing?'Ⅱ':'▶';option.classList.toggle('is-playing',playing);}
      button.addEventListener('click',async()=>{
        if(!audio.paused){audio.pause();soundStatus.textContent="Paused: "+label;return;}
        pauseOthers(audio);
        if(audio.ended)audio.currentTime=0;
        try{await audio.play();}catch(error){if(error.name!=='AbortError')soundStatus.textContent="This sound could not be opened. Please try again later.";}
      });
      audio.addEventListener('play',()=>{pauseOthers(audio);show(true);soundStatus.textContent="Playing: "+label;});
      audio.addEventListener('pause',()=>{show(false);if(!audio.ended&&!sounds.some(other=>!other.paused&&statusFor(other)===soundStatus))soundStatus.textContent="Paused: "+label;});
      audio.addEventListener('ended',()=>{show(false);soundStatus.textContent="Finished: "+label+". Choose another to compare?";});
      audio.addEventListener('error',()=>{show(false);soundStatus.textContent="This sound cannot be played right now. Please try again later.";});
    }
    for(const video of nativeVideos){
      const soundStatus=video.closest('#lab')?.querySelector('.audio-status'),label=video.getAttribute('aria-label');
      video.addEventListener('play',()=>{pauseOthers(video);if(soundStatus)soundStatus.textContent="Playing: "+label;});
      video.addEventListener('pause',()=>{if(soundStatus&&!video.ended&&!media.some(other=>!other.paused))soundStatus.textContent="Paused: "+label;});
      video.addEventListener('ended',()=>{if(soundStatus)soundStatus.textContent="Finished: "+label;});
      video.addEventListener('error',()=>{if(soundStatus)soundStatus.textContent="This video cannot be played right now. Please try again later.";});
    }
    document.addEventListener('visibilitychange',()=>{if(document.hidden)media.forEach(item=>item.pause());});
  }
  const releaseTabs=[...document.querySelectorAll('[data-release-tab]')];
  function selectRelease(tab){releaseTabs.forEach(button=>{const selected=button===tab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;const panel=document.getElementById(button.dataset.releaseTab);panel.hidden=!selected;if(selected)panel.classList.add('is-visible');});}
  releaseTabs.forEach((tab,index)=>{tab.addEventListener('click',()=>selectRelease(tab));tab.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(index+1)%releaseTabs.length;else if(event.key==='ArrowLeft')next=(index-1+releaseTabs.length)%releaseTabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=releaseTabs.length-1;else return;event.preventDefault();selectRelease(releaseTabs[next]);releaseTabs[next].focus();});});
  document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
    const status=document.querySelector('.copy-status');
    try{await navigator.clipboard.writeText(button.dataset.copy);status.textContent="Group number copied: "+button.dataset.copy+". Search for it in QQ.";}
    catch{status.textContent="QQ group: "+button.dataset.copy+". Select or long-press the number to copy it.";}
  }));
  // Mac download help shares one panel across mouse, keyboard and touch.
  const macGroup=document.getElementById('macos-download');
  if(macGroup){
    const help=macGroup.querySelector('.macos-help'),button=macGroup.querySelector('.download.macos');
    if(help&&button){
      const panel=help.querySelector('.macos-install-popover'),summary=help.querySelector('summary');
      let pinned=false,hovering=false,closing=false,closeTimer;
      function positionHelp(){
        if(!help.open)return;
        const box=macGroup.getBoundingClientRect(),anchor=button.getBoundingClientRect();
        const above=anchor.top-20,below=innerHeight-anchor.bottom-20,up=above>=300||above>below;
        panel.style.maxHeight=Math.max(80,Math.min(480,up?above:below))+'px';
        const width=panel.getBoundingClientRect().width,left=Math.max(16,Math.min(box.right-width,innerWidth-width-16));
        panel.style.right='auto';panel.style.left=(left-box.left)+'px';
        panel.style.bottom='auto';panel.style.top=(up?-panel.getBoundingClientRect().height-12:anchor.bottom-box.top+12)+'px';
      }
      function showHelp(open){help.open=open;if(open)positionHelp();}
      macGroup.addEventListener('pointerenter',event=>{if(event.pointerType==='mouse'){clearTimeout(closeTimer);hovering=true;showHelp(true);}});
      macGroup.addEventListener('pointerleave',event=>{if(event.pointerType==='mouse'){hovering=false;closeTimer=setTimeout(()=>{if(!hovering&&!pinned&&!macGroup.contains(document.activeElement))showHelp(false);},150);}});
      macGroup.addEventListener('focusin',()=>{if(!closing)showHelp(true);});
      macGroup.addEventListener('focusout',()=>setTimeout(()=>{if(!pinned&&!hovering&&!macGroup.contains(document.activeElement))showHelp(false);},0));
      summary.addEventListener('click',event=>{event.preventDefault();pinned=!pinned;showHelp(pinned||hovering);});
      document.addEventListener('keydown',event=>{if(event.key==='Escape'&&help.open){event.preventDefault();pinned=false;hovering=false;closing=true;showHelp(false);button.focus({preventScroll:true});closing=false;}});
      addEventListener('resize',positionHelp,{passive:true});addEventListener('scroll',positionHelp,{passive:true});
      document.fonts?.ready.then(positionHelp);
      function linkedHelp(){if(location.hash==='#macos-download'){pinned=true;showHelp(true);}}
      addEventListener('hashchange',linkedHelp);linkedHelp();
    }
  }
})();
// Same-origin client for the existing, email-verified feedback service.
(() => {
  'use strict';
  const $=id=>document.getElementById('feedback-'+id),panel=$('paper');if(!panel)return;
  const sessionKey='ptb-website-feedback-session-v1',pendingKey='ptb-website-feedback-pending-v1';
  const storage={get(key){try{return JSON.parse(sessionStorage.getItem(key));}catch{return null;}},set(key,value){try{if(value===null)sessionStorage.removeItem(key);else sessionStorage.setItem(key,JSON.stringify(value));}catch{/* Current page still works with storage disabled. */}}};
  let identity=storage.get(sessionKey),challenge='',retryAt=0,busy=false,cursor=0,pending=storage.get(pendingKey),images=[];
  if(!identity||typeof identity.token!=='string'||!/^[A-Za-z0-9_-]{43}$/.test(identity.token)||typeof identity.email!=='string')identity=null;
  const errors={invalid_email:"Enter a valid email address.",mail_not_configured:"Email is temporarily unavailable. Please try again later.",verification_rate_limited:"Too many code requests. Please try again later.",verification_expired:"This code has expired or reached its attempt limit. Request a new one.",verification_invalid:"Incorrect code. Please check your email.",feedback_unauthorized:"Your session has expired. Verify your email again.",invalid_image:"Choose PNG, JPEG or WebP images, no larger than 2 MB each.",invalid_message:"Write a message or add an image before sending.",message_key_conflict:"The retry content has changed. Refresh the conversation before sending.",message_rate_limited:"Messages are being sent too frequently. Please try again later.",conversation_budget:"This conversation has reached its attachment limit. Please use the email address below.",attachment_budget:"The total image size exceeds the limit. Remove some images and try again."};
  function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('is-error',error);}
  function render(){
    $('auth').hidden=!!identity;$('identity').hidden=!identity;$('history').hidden=!identity;
    if(identity)$('email-label').textContent=identity.email;
    $('code-row').hidden=!challenge;
    panel.querySelectorAll('button,input,textarea').forEach(e=>e.disabled=busy);
    $('submit').disabled=busy||!identity;
    const seconds=Math.max(0,Math.ceil((retryAt-Date.now())/1000));
    $('code-send').disabled=busy||seconds>0;$('code-send').textContent=seconds?seconds+"  seconds before resending":challenge?"Resend code":"Send code";
    $('email').disabled=busy||!!challenge;
    $('images-clear').hidden=!images.length;
  }
  function forget(){identity=null;challenge='';cursor=0;pending=null;storage.set(sessionKey,null);storage.set(pendingKey,null);$('messages').replaceChildren();$('code').value='';$('email').value='';render();}
  async function api(route,body,authenticated=true){
    const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),35000);
    try{
      const headers={Accept:'application/json'};if(body!==undefined)headers['Content-Type']='application/json';
      if(authenticated&&identity)headers.Authorization='Bearer '+identity.token;
      const response=await fetch('/api/v1/feedback/'+route,{method:body===undefined?'GET':'POST',headers,body:body===undefined?undefined:JSON.stringify(body),credentials:'omit',cache:'no-store',redirect:'error',signal:abort.signal});
      const data=await response.json();
      if(!response.ok){if(response.status===401)forget();throw Error(errors[data.detail]||(response.status===429?"Too many requests. Please try again later.":"The feedback service is temporarily unavailable. Keep your message and try again later."));}
      return data;
    }catch(error){if(error instanceof TypeError||error.name==='AbortError')throw Error("The network is unavailable. Your message is still here; refresh the conversation and retry.");throw error;}finally{clearTimeout(timer);}
  }
  async function run(action){if(busy)return;busy=true;render();try{await action();}catch(error){status(error.message,true);}finally{busy=false;render();}}
  function appendMessage(message){
    if($('messages').querySelector('[data-id="'+Number(message.id)+'"]'))return;
    const item=document.createElement('article');item.className='feedback-message '+(message.role==='admin'?'from-author':'from-visitor');item.dataset.id=Number(message.id);
    const heading=document.createElement('header'),name=document.createElement('strong'),time=document.createElement('time');name.textContent=message.role==='admin'?"Shaozi Jing":"Your note";time.textContent=new Date(message.created*1000).toLocaleString("en-US",{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});heading.append(name,time);
    const text=document.createElement('p');text.textContent=message.text;item.append(heading,text);
    for(const picture of message.images||[]){if(!['image/png','image/jpeg','image/webp'].includes(picture.mime)||typeof picture.data!=='string')continue;const img=document.createElement('img');img.src='data:'+picture.mime+';base64,'+picture.data;img.alt="Feedback attachment";img.loading='lazy';item.append(img);}
    $('messages').append(item);
  }
  async function refresh(reset=false){
    const data=await api('conversation?after='+(reset?0:cursor));
    if(reset){cursor=0;$('messages').replaceChildren();}
    if(data.messages.length)$('messages').querySelector('.feedback-empty')?.remove();
    data.messages.forEach(appendMessage);cursor=data.cursor;$('more').hidden=!data.hasMore;$('mail').checked=data.mailNotifications;
    if(!cursor){const p=document.createElement('p');p.className='feedback-empty';p.textContent="Your messages and replies will appear here.";$('messages').replaceChildren(p);}
    if($('history').open&&cursor)await api('read',{through:cursor});
    return data;
  }
  $('code-send').addEventListener('click',()=>run(async()=>{
    if(Date.now()<retryAt)return;if(!$('email').reportValidity())return;
    const email=$('email').value.trim(),data=await api('verification',{email},false);
    challenge=data.challenge;retryAt=Date.now()+data.retryAfter*1000;render();status("A verification email has been queued. Please check your inbox.");$('code').focus();
  }));
  // A cooldown changes only button text; no background verification or polling requests.
  setInterval(()=>{if(!busy&&retryAt)render();},1000);
  $('code-verify').addEventListener('click',()=>run(async()=>{
    if(!/^[0-9]{6}$/.test($('code').value)){status("Enter the 6-digit code from your email.",true);return;}
    identity=await api('verification/complete',{challenge,code:$('code').value},false);storage.set(sessionKey,identity);challenge='';$('code').value='';render();
    await refresh(true);status("Email verified. You can now send your note.");
  }));
  $('email-change').addEventListener('click',()=>{challenge='';$('code').value='';render();status("Enter a new email address and request a new code.");$('email').focus();});
  $('switch').addEventListener('click',()=>{forget();status("Your conversation is stored on the server. Verify the original email again to retrieve it.");$('email').focus();});
  $('email').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('code-send').click();}});
  $('code').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('code-verify').click();}});
  const accepted=['image/png','image/jpeg','image/webp'];
  $('images').addEventListener('change',()=>run(async()=>{
    const files=[...$('images').files];images=[];$('previews').replaceChildren();
    if(files.length>4||files.some(f=>!accepted.includes(f.type)||f.size>2000000)){$('images').value='';throw Error("Choose up to 4 PNG, JPEG or WebP images, no larger than 2 MB each.");}
    const prepared=await Promise.all(files.map(file=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({data:String(reader.result).split(',')[1],mime:file.type});reader.onerror=()=>reject(Error("The image could not be read. Please select it again."));reader.readAsDataURL(file);})));images=prepared;
    for(const file of images){const img=document.createElement('img');img.src='data:'+file.mime+';base64,'+file.data;img.alt="Screenshot to send";$('previews').append(img);}status(images.length?"Selected  "+images.length+"  screenshots. You can remove them before sending.":"Screenshots removed.");
  }));
  $('images-clear').addEventListener('click',()=>{images=[];$('images').value='';$('previews').replaceChildren();render();});
  $('mail').addEventListener('change',()=>{if(!identity)return;const checked=$('mail').checked;run(async()=>{try{await api('preferences',{mailNotifications:checked});status(checked?"Reply notifications by email are enabled.":"Email notifications are disabled. Replies will still appear here.");}catch(error){$('mail').checked=!checked;throw error;}});});
  $('submit').addEventListener('click',()=>run(async()=>{
    if(!identity)return;const module=$('module').value.trim(),message=$('text').value.trim();
    if(!message&&!images.length){status("Write a message or add a screenshot before sending your note.",true);$('text').focus();return;}
    const body={text:"From the website"+(module?' · '+module:'')+'\n\n'+message,images:images.map(({data})=>({data}))};
    const fingerprint=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify({owner:identity.owner,...body}))))).map(x=>x.toString(16).padStart(2,'0')).join('');
    if(!pending||pending.fingerprint!==fingerprint){pending={fingerprint,requestId:crypto.randomUUID()};storage.set(pendingKey,pending);}
    await api('preferences',{mailNotifications:$('mail').checked});
    await api('messages',{...body,requestId:pending.requestId});
    pending=null;storage.set(pendingKey,null);$('text').value='';images=[];$('images').value='';$('previews').replaceChildren();
    status("Your note has reached Shaozi's feedback inbox. Replies will appear here too.");
    $('history').open=true;
    try{await refresh();}catch{status("Your note was sent, but the conversation has not refreshed. Click Check for replies later.");}
  }));
  $('refresh').addEventListener('click',()=>run(async()=>{await refresh();status("Latest replies loaded.");}));
  $('more').addEventListener('click',()=>run(()=>refresh()));
  $('history').addEventListener('toggle',()=>{if($('history').open&&identity&&cursor&&!busy)run(()=>api('read',{through:cursor}));});
  render();if(identity)run(async()=>{await refresh(true);status("Restored the feedback conversation for this tab.");});
})();
// Anonymous server aggregates. Never substitute invented totals or past samples.
(() => {
  const panel=document.querySelector('.site-stats');if(!panel)return;
  const keys=['views','downloads','web','stars'],number=n=>n.toLocaleString("en-US");
  const valid=n=>Number.isSafeInteger(n)&&n>=0;
  const date=t=>new Date(t*1000).toLocaleString("en-US",{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  let history=[],fetching=false,lastUpdate=0;
  let selectedTime=null;
  const alignPlots=()=>panel.querySelectorAll('[data-chart]').forEach(card=>{
    card.style.setProperty('--chart-top',card.querySelector('.stats-chart-total').offsetTop+'px');
    const svg=card.querySelector('svg'),rect=svg.getBoundingClientRect();
    if(rect.width&&rect.height)card.querySelectorAll('.chart-dot').forEach(dot=>{dot.setAttribute('rx',String(3.5*svg.viewBox.baseVal.width/rect.width));dot.setAttribute('ry',String(3.5*svg.viewBox.baseVal.height/rect.height));});
  });
  const plotResize=new ResizeObserver(alignPlots);panel.querySelectorAll('[data-chart]').forEach(card=>plotResize.observe(card));alignPlots();
  // Only the all-history drawing is reduced. Queries and time bounds use original records.
  function* plotSamples(points,column,start,span,reduce){
    const selected=reduce&&points.length>2048?new Set():null,breaks=new Set();
    let bucket=-1,first=0,last=0,min=0,max=0;
    const keep=()=>{if(bucket<0)return;for(const index of [first,min,max,last])selected.add(index);};
    for(let i=0;i<points.length;i++){
      const p=points[i],age=Date.now()/1000-p[0],gap=age<2*86400?1800:age<90*86400?10800:3*86400;
      if(i===0||p[0]-points[i-1][0]>gap){breaks.add(i);if(selected){selected.add(i);if(i)selected.add(i-1);}}
      if(!selected)continue;
      const next=Math.min(255,Math.floor((p[0]-start)/span*256));
      if(next!==bucket){keep();bucket=next;first=last=min=max=i;}
      else{last=i;if(p[column]<points[min][column])min=i;if(p[column]>points[max][column])max=i;}
    }
    if(selected)keep();
    for(const i of selected?[...selected].sort((a,b)=>a-b):points.keys())yield {point:points[i],move:breaks.has(i)};
  }
  function charts(){
    const range=panel.querySelector('[data-stats-range]').value,cutoff=range==='all'?0:Date.now()/1000-Number(range)*86400;
    const values=history.filter(p=>p[0]>=cutoff);
    const start=values[0]?.[0],end=values.at(-1)?.[0],span=Math.max(1,end-start);
    const dates=panel.querySelectorAll('[data-stats-dates] span'),selectors=[];
    dates[0].textContent=values.length?date(start):'';dates[1].textContent=values.length?date(end):'';
    function nearest(stamp){
      let low=0,high=values.length-1;
      while(low<high){const mid=Math.floor((low+high)/2);if(values[mid][0]<stamp)low=mid+1;else high=mid;}
      return low>0&&stamp-values[low-1][0]<=values[low][0]-stamp?low-1:low;
    }
    function select(position,remember=false){
      const p=values[position];if(!p)return;
      selectors.forEach(apply=>apply(p));
      if(remember)selectedTime=p[0]===end?null:p[0];
    }
    function inspect(card){panel.querySelectorAll('[data-chart]').forEach(c=>c.classList.toggle('is-inspecting',c===card));}
    panel.querySelectorAll('[data-chart]').forEach((card,index)=>{
      const columns=index===1?[{column:2,name:'Windows',platform:'windows'},{column:5,name:'macOS',platform:'macos'},{column:6,name:"IPA input method",platform:'ime'}]:[{column:index+1,name:'',platform:''}];
      const series=columns.map(s=>({...s,values:values.filter(p=>valid(p[s.column]))}));
      const hasRecords=series.some(s=>s.values.length);
      const svg=card.querySelector('svg'),empty=card.querySelector('.chart-empty'),slider=card.querySelector('input'),output=card.querySelector('output');
      svg.replaceChildren();empty.hidden=hasRecords;slider.disabled=values.length<2||!hasRecords;
      slider.oninput=null;slider.onkeydown=null;slider.onpointermove=null;slider.onpointerdown=null;
      if(!values.length){slider.min='0';slider.max='1';slider.value='0';slider.removeAttribute('aria-valuetext');output.textContent="No records yet";return;}
      const max=series.reduce((maximum,s)=>s.values.reduce((value,p)=>Math.max(value,p[s.column]),maximum),1);
      slider.min=String(start);slider.max=String(start+span);slider.step='1';
      const xy=(p,column)=>[38+(p[0]-start)/span*426,153-p[column]/max*125];
      const node=(tag,attrs,parent=svg)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));parent.append(e);return e;};
      for(const v of [0,max/2]){const y=153-v/max*125;node('path',{d:`M38 ${y}H464`,class:'chart-grid'});}
      const cursor=node('line',{x1:464,x2:464,y1:28,y2:153,class:'chart-cursor'});
      for(const s of series){
        if(!s.values.length)continue;
        let d='',area='',segment='',firstX=0;
        for(const {point:p,move} of plotSamples(s.values,s.column,start,span,range==='all')){
          const [x,y]=xy(p,s.column),command=move?`M${x},${y}`:`H${x}V${y}`;d+=command;
          if(move){if(segment)area+=segment+`V153H${firstX}Z`;segment=command;firstX=x;}else segment+=command;
        }
        if(segment)area+=segment+`V153H${firstX}Z`;
        if(series.length===1){
          const defs=node('defs',{}),gradient=node('linearGradient',{id:'stats-fill-'+keys[index],x1:0,y1:0,x2:0,y2:1},defs);
          node('stop',{offset:'0%','stop-color':'var(--chart-color)','stop-opacity':'.12'},gradient);node('stop',{offset:'100%','stop-color':'var(--chart-color)','stop-opacity':'.01'},gradient);
          node('path',{d:area,class:'chart-area',fill:'url(#stats-fill-'+keys[index]+')'});
        }
        node('path',{d,class:'chart-line','data-platform':s.platform});
        s.marker=node('ellipse',{cx:0,cy:0,rx:3.5,ry:3.5,class:'chart-dot','data-platform':s.platform});
      }
      selectors.push(p=>{
        slider.value=String(p[0]);const labels=[],cursorX=38+(p[0]-start)/span*426;
        cursor.setAttribute('x1',String(cursorX));cursor.setAttribute('x2',String(cursorX));
        cursor.style.display=p[0]===end?'none':'';
        for(const s of series){
          let low=0,high=s.values.length;
          while(low<high){const mid=Math.floor((low+high)/2);if(s.values[mid][0]<=p[0])low=mid+1;else high=mid;}
          const sample=s.values[low-1];
          if(s.marker){s.marker.style.display=sample?'':'none';if(sample){const [x,y]=xy(sample,s.column);s.marker.setAttribute('cx',String(x));s.marker.setAttribute('cy',String(y));}}
          labels.push((s.name?s.name+' ':'')+(sample?number(sample[s.column])+(index===3?" stars":" requests"):"No records yet"));
        }
        output.textContent=date(p[0])+' · '+labels.join(' · ');slider.setAttribute('aria-valuetext',output.textContent);
      });
      slider.oninput=()=>select(nearest(Number(slider.value)),true);
      slider.onkeydown=event=>{
        if(event.key==='Escape'){selectedTime=null;select(values.length-1);card.classList.remove('is-inspecting');return;}
        const offsets={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1,PageDown:-10,PageUp:10};
        if(!(event.key in offsets)&&event.key!=='Home'&&event.key!=='End')return;
        event.preventDefault();const current=nearest(Number(slider.value));
        select(event.key==='Home'?0:event.key==='End'?values.length-1:Math.max(0,Math.min(values.length-1,current+offsets[event.key])),true);
      };
      const pointAt=event=>{
        if(slider.disabled)return;
        const rect=svg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((event.clientX-rect.left)/rect.width*480-38)/426));
        inspect(card);select(nearest(start+fraction*span),true);
      };
      slider.onpointerdown=pointAt;slider.onpointermove=event=>{if(event.pointerType==='mouse'||event.buttons)pointAt(event);};
      slider.onfocus=()=>inspect(card);
      slider.onblur=()=>card.classList.remove('is-inspecting');
      slider.onpointerleave=event=>{if(event.pointerType==='mouse'&&document.activeElement!==slider)card.classList.remove('is-inspecting');};
      node('title',{}).textContent=card.querySelector('h3').textContent+", with "+values.filter(p=>columns.some(s=>valid(p[s.column]))).length+" recorded samples. Hover, tap or use the arrow keys to inspect them. All four charts share the selected time.";
    });
    if(!values.length)return;
    select(selectedTime==null?values.length-1:nearest(selectedTime));
    alignPlots();
  }
  async function readSnapshot(){
    for(let attempt=0;attempt<2;attempt++){
      const r=await fetch('/site-stats.json',{cache:'no-store',credentials:'omit'});if(!r.ok)throw Error();const data=await r.json();
      const h=await fetch('/site-stats-history.json',{cache:'no-store',credentials:'omit'});if(!h.ok)throw Error();const series=await h.json();
      if(series.schema!==1||!Array.isArray(series.points))throw Error();
      // The collector replaces two files in sequence. Retry a sampling boundary,
      // and keep both displayed values until a matching pair is available.
      if(data.historyUpdatedAt&&series.updatedAt&&data.historyUpdatedAt!==series.updatedAt)continue;
      return {data,series};
    }
    throw Error();
  }
  async function update(){
    if(fetching||document.hidden)return;fetching=true;
    try{
      const {data,series}=await readSnapshot();
      const macos=data.downloads?.macos,ime=data.downloads?.ime;
      const untimed=data.downloads?.untimed?.total??0;if(!valid(untimed))throw Error();
      if(data.schema!==1||![data.pageViews,data.downloads?.total,data.downloads?.portable,data.downloads?.installer].every(valid)||(macos!=null&&!valid(macos))||(ime!=null&&!valid(ime))||data.downloads.total!==data.downloads.portable+data.downloads.installer+(macos??0)+(ime??0))throw Error();
      const counts=[data.pageViews,data.downloads.total,data.webUses,data.github?.count];keys.forEach((key,i)=>panel.querySelector('[data-stat="'+key+'"]').textContent=valid(counts[i])?number(counts[i]):'—');
      panel.querySelector('[data-stat-platform="windows"]').textContent=number(data.downloads.portable+data.downloads.installer);
      panel.querySelector('[data-stat-platform="macos"]').textContent=valid(macos)?number(macos):'—';
      panel.querySelector('[data-stat-platform="ime"]').textContent=valid(ime)?number(ime):'—';
      lastUpdate=Date.parse(data.updatedAt);panel.querySelector('[data-stats-since]').textContent="Recording since  "+new Date(data.startedAt).toLocaleDateString("en-US",{timeZone:'Asia/Shanghai'})+" ";
      panel.querySelector('[data-stats-updated]').textContent="Statistics updated  "+date(lastUpdate/1000)+(Date.now()-lastUpdate>180000?"  · Data not yet updated":'')+(data.github?.updatedAt?"  · Stars sampled  "+date(Date.parse(data.github.updatedAt)/1000):"  · No Star samples yet");
      history=series.points.filter(p=>Array.isArray(p)&&(p.length===5||p.length===6||p.length===7)&&valid(p[0])&&p.slice(1).every(v=>v===null||valid(v))).sort((a,b)=>a[0]-b[0]);charts();
    }catch{panel.querySelector('[data-stats-updated]').textContent=lastUpdate?"Temporarily unable to update; showing records from  "+date(lastUpdate/1000)+" .":"Statistics are temporarily unavailable. Please try again later.";}
    finally{fetching=false;}
  }
  panel.querySelector('[data-stats-range]').addEventListener('change',charts);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)void update();});
  void update();setInterval(update,60000);
})();

// Accessible reading bookmark: ordinary anchors remain usable without JavaScript.
(() => {
  const dock=document.querySelector('.reading-dock');if(!dock)return;
  const toggle=dock.querySelector('.reading-toggle'),menu=dock.querySelector('.reading-menu');
  const links=[...menu.querySelectorAll('a')],sections=links.map(a=>document.querySelector(a.getAttribute('href')));
  let pending=false;
  function close(focus=false){menu.hidden=true;toggle.setAttribute('aria-expanded','false');if(focus)toggle.focus({preventScroll:true});}
  function update(){pending=false;const span=document.documentElement.scrollHeight-innerHeight;const progress=span>0?Math.min(100,Math.max(0,scrollY/span*100)):0;dock.style.setProperty('--reading-progress',progress.toFixed(2));dock.hidden=scrollY<420&&!dock.contains(document.activeElement);let current=0;for(let i=0;i<sections.length;i++)if(sections[i]&&sections[i].getBoundingClientRect().top<innerHeight*.4)current=i;links.forEach((a,i)=>{if(i===current)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}
  function queue(){if(!pending){pending=true;requestAnimationFrame(update);}}
  toggle.addEventListener('click',()=>{const open=menu.hidden;menu.hidden=!open;toggle.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!menu.hidden)close(true);});
  document.addEventListener('pointerdown',e=>{if(!dock.contains(e.target))close();});
  links.forEach((a,i)=>a.addEventListener('click',()=>{close();const target=sections[i];if(target){target.setAttribute('tabindex','-1');target.focus({preventScroll:true});}}));
  dock.querySelector('.reading-top').addEventListener('click',()=>{close();document.querySelector('.site-header .brand').focus({preventScroll:true});window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});});
  addEventListener('scroll',queue,{passive:true});addEventListener('resize',queue,{passive:true});update();
})();

// Load a single official player on selection; changing channels stops the old video.
(() => {
  const channels = [...document.querySelectorAll('[data-tutorial]')];
  if (!channels.length) return;
  const screen = document.getElementById('tutorial-screen');
  const fullscreen = document.getElementById('tutorial-fullscreen');
  const status = document.getElementById('tutorial-status');
  const episodes = document.getElementById('tutorial-episodes');
  const external = document.querySelector('.tutorial-actions a');
  const videos = {
    intro: {bvid:'BV1WWpr67EkX', title:"PhoneticToolbox introduction"},
    recording: {bvid:'BV1xsnJ6nES2', title:"PhoneticToolbox tutorial · 01 Recording"},
    'lip-extraction': {bvid:'BV1NsnJ6nE2Z', title:"PhoneticToolbox tutorial · 02 Lip tracking"},
    'parameter-estimation': {bvid:'BV1NcJd6UEZv', title:"PhoneticToolbox tutorial · 03 Parameter estimation"},
    'parameter-display': {bvid:'BV11cJd6UESc', title:"PhoneticToolbox tutorial · 04 Parameter display"},
    egg: {bvid:'BV1KFJd6JEvg', title:"PhoneticToolbox tutorial · 05 EGG analysis"},
    lpc: {bvid:'BV17mnc6QExx', title:"PhoneticToolbox tutorial · 06 LPC spectrum"},
    'phonation-synthesis': {bvid:'BV1g1ns63EXQ', title:"PhoneticToolbox tutorial · Phonation synthesis"},
    'pitch-manipulation': {bvid:'BV1QCns6kEqY', title:"PhoneticToolbox tutorial · Time and pitch manipulation"},
    spectrogram: {bvid:'BV17mnc6QEYY', title:"PhoneticToolbox tutorial · 11 Spectrogram-to-audio"},
    'ipa-plus': {bvid:'BV17mnc6QEYL', title:"PhoneticToolbox tutorial · 12 IPA Chart Plus"},
    'mandarin-ipa': {bvid:'BV13Cns6kExb', title:"PhoneticToolbox tutorial · Chinese-to-IPA transcription"},
    mfa: {bvid:'BV1g1ns63EyT', title:"PhoneticToolbox tutorial · MFA forced alignment"},
    annotation: {bvid:'BV17mnc6QEJN', title:"PhoneticToolbox tutorial · 13 TextGrid annotation"},
    phonology: {bvid:'BV17mnc6QEJf', title:"PhoneticToolbox tutorial · 14 Phonological analysis"},
    perception: {bvid:'BV1Bmnc6QEjn', title:"PhoneticToolbox tutorial · 15 Perception experiments"},
    'paper-reading': {bvid:'BV1Bmnc6QELa', title:"PhoneticToolbox tutorial · 16 Paper reader"},
    'spectrogram-special': {bvid:'BV1RuJm6gEWE', title:"Spectrogram-to-audio · Overview"},
  };
  let selected = '';
  function play(key) {
    const channelButton = channels.find(button => button.dataset.tutorial === key);
    if (!Object.hasOwn(videos, key)) {
      if (!channelButton || selected === key) return;
      selected = key;
      const placeholder = document.createElement('div');
      placeholder.className = 'tutorial-idle';
      const title = document.createElement('strong');
      title.textContent = channelButton.textContent.trim();
      const message = document.createElement('p');
      message.textContent = "Tutorials are being updated.";
      placeholder.append(title, message);
      screen.replaceChildren(placeholder);
      channels.forEach(button => button.setAttribute('aria-pressed', String(button === channelButton)));
      episodes.hidden = true;
      fullscreen.hidden = true;
      external.hidden = true;
      status.textContent = title.textContent + "  · Tutorial coming soon";
      return;
    }
    const video = videos[key];
    if (!video || selected === key) return;
    selected = key;
    const player = document.createElement('iframe');
    player.title = video.title + "  · Bilibili video";
    player.src = 'https://player.bilibili.com/player.html?bvid='+video.bvid+'&page=1&autoplay=0';
    player.allow = 'fullscreen; autoplay; encrypted-media; picture-in-picture';
    player.allowFullscreen = true;
    player.referrerPolicy = 'strict-origin-when-cross-origin';
    screen.replaceChildren(player);
    const channel = key === 'spectrogram-special' ? 'spectrogram' : key;
    channels.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.tutorial === channel)));
    episodes.hidden = channel !== 'spectrogram';
    episodes.querySelectorAll('[data-tutorial-episode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.tutorialEpisode === key)));
    external.href = 'https://www.bilibili.com/video/'+video.bvid+'/';
    external.hidden = false;
    external.setAttribute('aria-label', "Watch on Bilibili"+video.title);
    status.textContent = video.title;
    if (document.fullscreenEnabled && screen.requestFullscreen) fullscreen.hidden = false;
  }
  channels.forEach(button => button.addEventListener('click', () => play(button.dataset.tutorial)));
  episodes.querySelectorAll('[data-tutorial-episode]').forEach(button => button.addEventListener('click', () => play(button.dataset.tutorialEpisode)));
  function selectLinkedTutorial() {
    const match = /^#tutorial-([a-z0-9-]+)$/.exec(location.hash);
    if (match) play(match[1]);
  }
  addEventListener('hashchange', selectLinkedTutorial);
  selectLinkedTutorial();
  fullscreen.addEventListener('click', async () => {
    try { await screen.requestFullscreen(); status.textContent = ''; }
    catch { status.textContent = "Use the video's full-screen control, or watch on Bilibili."; }
  });
})();

// The full bibliography is static HTML, so it remains available without JavaScript.
(() => {
  document.querySelectorAll('.reference-catalog').forEach(catalog => {
    const query = catalog.querySelector('.reference-query');
    const module = catalog.querySelector('.reference-module');
    const status = catalog.querySelector('.reference-count');
    const results = catalog.querySelector('.reference-results');
    const rows = [...catalog.querySelectorAll('.reference-source')].map(element => ({
      element, text:element.textContent.toLowerCase(), modules:element.dataset.modules.split(' '),
    }));
    catalog.querySelector('.reference-filters').hidden = false;
    function filter() {
      const term = query.value.trim().toLowerCase();
      let count = 0;
      rows.forEach(row => {
        row.element.hidden = !(row.text.includes(term) && (!module.value || row.modules.includes(module.value)));
        if (!row.element.hidden) count++;
      });
      status.textContent = count ? "Showing  "+count+"  sources" : "No matching sources. Try different search terms or change the module filter.";
      results.scrollTop = 0;
    }
    query.addEventListener('input', filter);
    module.addEventListener('change', filter);
    catalog.addEventListener('toggle', () => {
      catalog.querySelector(':scope > summary span').textContent = catalog.open ? "Collapse list" : "Expand to browse";
    });
    catalog.querySelectorAll('.reference-copy').forEach(button => {
      button.hidden = false;
      button.addEventListener('click', async () => {
        const citation = button.closest('.reference-source').querySelector('.reference-citation');
        try { await navigator.clipboard.writeText(citation.textContent); status.textContent = "Citation copied"; }
        catch {
          const range = document.createRange(); range.selectNodeContents(citation);
          const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
          status.textContent = "Citation text selected. Copy it now.";
        }
      });
    });
  });
})();
