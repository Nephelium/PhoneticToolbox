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
      function show(playing){button.setAttribute('aria-pressed',String(playing));button.setAttribute('aria-label',(playing?'暂停':'播放')+label);button.querySelector('.sound-symbol').textContent=playing?'Ⅱ':'▶';option.classList.toggle('is-playing',playing);}
      button.addEventListener('click',async()=>{
        if(!audio.paused){audio.pause();soundStatus.textContent='已暂停：'+label;return;}
        sounds.forEach(other=>{if(other!==audio&&!other.paused)other.pause();});
        if(audio.ended)audio.currentTime=0;
        try{await audio.play();}catch(error){if(error.name!=='AbortError')soundStatus.textContent='这段声音暂时没有打开，请稍后再试。';}
      });
      audio.addEventListener('play',()=>{sounds.forEach(other=>{if(other!==audio)other.pause();});show(true);soundStatus.textContent='正在播放：'+label;});
      audio.addEventListener('pause',()=>{show(false);if(!audio.ended&&!sounds.some(other=>!other.paused&&statusFor(other)===soundStatus))soundStatus.textContent='已暂停：'+label;});
      audio.addEventListener('ended',()=>{show(false);soundStatus.textContent='听完了：'+label+'。再选一个，比较一下？';});
      audio.addEventListener('error',()=>{show(false);soundStatus.textContent='这段声音暂时无法播放，请稍后再试。';});
    }
    document.addEventListener('visibilitychange',()=>{if(document.hidden)sounds.forEach(audio=>audio.pause());});
  }
  const releaseTabs=[...document.querySelectorAll('[data-release-tab]')];
  function selectRelease(tab){releaseTabs.forEach(button=>{const selected=button===tab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;const panel=document.getElementById(button.dataset.releaseTab);panel.hidden=!selected;if(selected)panel.classList.add('is-visible');});}
  releaseTabs.forEach((tab,index)=>{tab.addEventListener('click',()=>selectRelease(tab));tab.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(index+1)%releaseTabs.length;else if(event.key==='ArrowLeft')next=(index-1+releaseTabs.length)%releaseTabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=releaseTabs.length-1;else return;event.preventDefault();selectRelease(releaseTabs[next]);releaseTabs[next].focus();});});
  document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
    const status=document.querySelector('.copy-status');
    try{await navigator.clipboard.writeText(button.dataset.copy);status.textContent='群号已复制：'+button.dataset.copy+'，去 QQ 搜索就好啦。';}
    catch{status.textContent='QQ 群号：'+button.dataset.copy+'。可以长按或选中文字复制。';}
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
      document.addEventListener('keydown',event=>{if(event.key==='Escape'&&help.open){event.preventDefault();pinned=false;hovering=false;closing=true;showHelp(false);summary.focus({preventScroll:true});closing=false;}});
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
  const errors={invalid_email:'请输入有效的邮箱地址。',mail_not_configured:'邮件服务暂时无法使用，请稍后重试。',verification_rate_limited:'验证码发送过于频繁，请稍后再试。',verification_expired:'验证码已过期或尝试次数已用完，请重新获取。',verification_invalid:'验证码不正确，请检查邮件。',feedback_unauthorized:'会话已过期，请重新验证邮箱。',invalid_image:'请选择 PNG、JPEG 或 WebP 图片，每张不超过 2 MB。',invalid_message:'请写下问题或添加图片后再发送。',message_key_conflict:'重试内容发生变化，请刷新会话后再发送。',message_rate_limited:'发送有点频繁，请稍后重试。',conversation_budget:'这段会话的附件已达到限额，请通过下方邮箱联系。',attachment_budget:'图片总量超过限制，请减少图片后再试。'};
  function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('is-error',error);}
  function render(){
    $('auth').hidden=!!identity;$('identity').hidden=!identity;$('history').hidden=!identity;
    if(identity)$('email-label').textContent=identity.email;
    $('code-row').hidden=!challenge;
    panel.querySelectorAll('button,input,textarea').forEach(e=>e.disabled=busy);
    $('submit').disabled=busy||!identity;
    const seconds=Math.max(0,Math.ceil((retryAt-Date.now())/1000));
    $('code-send').disabled=busy||seconds>0;$('code-send').textContent=seconds?seconds+' 秒后重发':challenge?'重新发送':'发送验证码';
    $('email').disabled=busy||!!challenge;
    $('images-clear').hidden=!images.length;
  }
  function forget(){identity=null;challenge='';cursor=0;pending=null;storage.set(sessionKey,null);storage.set(pendingKey,null);$('messages').replaceChildren();$('code').value='';$('email').value='';render();}
  async function api(route,body,authenticated=true){
    const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),35000);
    try{
      const headers={Accept:'application/json'};if(body!==undefined)headers['Content-Type']='application/json';
      if(authenticated&&identity)headers.Authorization='Bearer '+identity.token;
      if(route==='messages'&&body!==undefined)headers['X-PTB-Feedback-Source']='web';
      const response=await fetch('/api/v1/feedback/'+route,{method:body===undefined?'GET':'POST',headers,body:body===undefined?undefined:JSON.stringify(body),credentials:'omit',cache:'no-store',redirect:'error',signal:abort.signal});
      const data=await response.json();
      if(!response.ok){if(response.status===401)forget();throw Error(errors[data.detail]||(response.status===429?'操作有点频繁，请稍后重试。':'反馈服务暂时没连上，请保留内容，稍后重试。'));}
      return data;
    }catch(error){if(error instanceof TypeError||error.name==='AbortError')throw Error('网络暂时没连上。内容仍在这里，刷新会话后可以重试发送。');throw error;}finally{clearTimeout(timer);}
  }
  async function run(action){if(busy)return;busy=true;render();try{await action();}catch(error){status(error.message,true);}finally{busy=false;render();}}
  function appendMessage(message){
    if($('messages').querySelector('[data-id="'+Number(message.id)+'"]'))return;
    const item=document.createElement('article');item.className='feedback-message '+(message.role==='admin'?'from-author':'from-visitor');item.dataset.id=Number(message.id);
    const heading=document.createElement('header'),name=document.createElement('strong'),time=document.createElement('time');name.textContent=message.role==='admin'?'井韶子':'你的便签';time.textContent=new Date(message.created*1000).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});heading.append(name,time);
    const text=document.createElement('p');text.textContent=message.text;item.append(heading,text);
    for(const picture of message.images||[]){if(!['image/png','image/jpeg','image/webp'].includes(picture.mime)||typeof picture.data!=='string')continue;const img=document.createElement('img');img.src='data:'+picture.mime+';base64,'+picture.data;img.alt='反馈会话附图';img.loading='lazy';item.append(img);}
    $('messages').append(item);
  }
  async function refresh(reset=false){
    const data=await api('conversation?after='+(reset?0:cursor));
    if(reset){cursor=0;$('messages').replaceChildren();}
    if(data.messages.length)$('messages').querySelector('.feedback-empty')?.remove();
    data.messages.forEach(appendMessage);cursor=data.cursor;$('more').hidden=!data.hasMore;$('mail').checked=data.mailNotifications;
    if(!cursor){const p=document.createElement('p');p.className='feedback-empty';p.textContent='这里会留下我们的来信往返。';$('messages').replaceChildren(p);}
    if($('history').open&&cursor)await api('read',{through:cursor});
    return data;
  }
  $('code-send').addEventListener('click',()=>run(async()=>{
    if(Date.now()<retryAt)return;if(!$('email').reportValidity())return;
    const email=$('email').value.trim(),data=await api('verification',{email},false);
    challenge=data.challenge;retryAt=Date.now()+data.retryAfter*1000;render();status('验证码已安排发送，请查收邮箱。');$('code').focus();
  }));
  // A cooldown changes only button text; no background verification or polling requests.
  setInterval(()=>{if(!busy&&retryAt)render();},1000);
  $('code-verify').addEventListener('click',()=>run(async()=>{
    if(!/^[0-9]{6}$/.test($('code').value)){status('请填写邮件中的 6 位验证码。',true);return;}
    identity=await api('verification/complete',{challenge,code:$('code').value},false);storage.set(sessionKey,identity);challenge='';$('code').value='';render();
    await refresh(true);status('邮箱已验证，可以把便签寄出去啦。');
  }));
  $('email-change').addEventListener('click',()=>{challenge='';$('code').value='';render();status('填写新的邮箱后，重新获取验证码。');$('email').focus();});
  $('switch').addEventListener('click',()=>{forget();status('会话保存在服务器，再次验证原邮箱即可找回。');$('email').focus();});
  $('email').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('code-send').click();}});
  $('code').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('code-verify').click();}});
  const accepted=['image/png','image/jpeg','image/webp'];
  $('images').addEventListener('change',()=>run(async()=>{
    const files=[...$('images').files];images=[];$('previews').replaceChildren();
    if(files.length>4||files.some(f=>!accepted.includes(f.type)||f.size>2000000)){$('images').value='';throw Error('最多选择 4 张 PNG、JPEG 或 WebP 图片，每张不超过 2 MB。');}
    const prepared=await Promise.all(files.map(file=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({data:String(reader.result).split(',')[1],mime:file.type});reader.onerror=()=>reject(Error('图片读取失败，请重新选择。'));reader.readAsDataURL(file);})));images=prepared;
    for(const file of images){const img=document.createElement('img');img.src='data:'+file.mime+';base64,'+file.data;img.alt='待发送截图';$('previews').append(img);}status(images.length?'已选 '+images.length+' 张截图，发送前可以移除。':'已移除截图。');
  }));
  $('images-clear').addEventListener('click',()=>{images=[];$('images').value='';$('previews').replaceChildren();render();});
  $('mail').addEventListener('change',()=>{if(!identity)return;const checked=$('mail').checked;run(async()=>{try{await api('preferences',{mailNotifications:checked});status(checked?'已开启回复邮件提醒。':'已关闭邮件提醒，回复仍保存在这里。');}catch(error){$('mail').checked=!checked;throw error;}});});
  $('submit').addEventListener('click',()=>run(async()=>{
    if(!identity)return;const module=$('module').value.trim(),message=$('text').value.trim();
    if(!message&&!images.length){status('写下问题或添加截图后，再把便签寄出去吧。',true);$('text').focus();return;}
    const body={text:'来自官网'+(module?' · '+module:'')+'\n\n'+message,images:images.map(({data})=>({data}))};
    const fingerprint=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify({owner:identity.owner,...body}))))).map(x=>x.toString(16).padStart(2,'0')).join('');
    if(!pending||pending.fingerprint!==fingerprint){pending={fingerprint,requestId:crypto.randomUUID()};storage.set(pendingKey,pending);}
    await api('preferences',{mailNotifications:$('mail').checked});
    await api('messages',{...body,requestId:pending.requestId});
    pending=null;storage.set(pendingKey,null);$('text').value='';images=[];$('images').value='';$('previews').replaceChildren();
    status('便签已寄到韶子的反馈收件箱。收到回复后，这里也能看到。');
    $('history').open=true;
    try{await refresh();}catch{status('便签已成功寄出。会话暂时未刷新，稍后点击“查看最新回复”即可。');}
  }));
  $('refresh').addEventListener('click',()=>run(async()=>{await refresh();status('已查看最新回复。');}));
  $('more').addEventListener('click',()=>run(()=>refresh()));
  $('history').addEventListener('toggle',()=>{if($('history').open&&identity&&cursor&&!busy)run(()=>api('read',{through:cursor}));});
  render();if(identity)run(async()=>{await refresh(true);status('已恢复当前标签页的反馈会话。');});
})();
// Anonymous server aggregates. Never substitute invented totals or past samples.
(() => {
  const panel=document.querySelector('.site-stats');if(!panel)return;
  const keys=['views','downloads','web','stars'],number=n=>n.toLocaleString('zh-CN');
  const valid=n=>Number.isSafeInteger(n)&&n>=0;
  const date=t=>new Date(t*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  let history=[],fetching=false,lastUpdate=0;
  const selectedTimes=new Map();
  function charts(){
    const range=panel.querySelector('[data-stats-range]').value,cutoff=range==='all'?0:Date.now()/1000-Number(range)*86400;
    panel.querySelectorAll('[data-chart]').forEach((card,index)=>{
      const columns=index===1?[{column:2,name:'Windows',platform:'windows'},{column:5,name:'macOS',platform:'macos'}]:[{column:index+1,name:'',platform:''}];
      const values=history.filter(p=>p[0]>=cutoff&&columns.some(s=>valid(p[s.column])));
      const series=columns.map(s=>({...s,values:values.filter(p=>valid(p[s.column]))}));
      const svg=card.querySelector('svg'),empty=card.querySelector('.chart-empty'),slider=card.querySelector('input'),output=card.querySelector('output'),labels=card.querySelectorAll('.chart-dates span');
      svg.replaceChildren();empty.hidden=!!values.length;slider.disabled=values.length<2;
      labels[0].textContent=values.length?date(values[0][0]):'';labels[1].textContent=values.length?date(values.at(-1)[0]):'';
      slider.oninput=null;slider.onkeydown=null;
      if(!values.length){slider.min='0';slider.max='1';slider.value='0';slider.removeAttribute('aria-valuetext');output.textContent='暂无记录';return;}
      const max=Math.max(1,...series.flatMap(s=>s.values.map(p=>p[s.column]))),start=values[0][0],end=values.at(-1)[0],span=Math.max(1,end-start);
      slider.min=String(start);slider.max=String(start+span);slider.step='1';
      const xy=(p,column)=>[38+(p[0]-start)/span*426,153-p[column]/max*125];
      const node=(tag,attrs)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));svg.append(e);return e;};
      for(const v of [...new Set([0,Math.round(max/2),max])]){const y=153-v/max*125;node('path',{d:`M38 ${y}H464`,class:'chart-grid'});node('text',{x:31,y:y+4,'text-anchor':'end',class:'chart-axis'}).textContent=number(Math.round(v));}
      for(const s of series){
        if(!s.values.length)continue;
        let d='',previous;
        for(const p of s.values){const [x,y]=xy(p,s.column);const age=Date.now()/1000-p[0],gap=age<2*86400?1800:age<90*86400?10800:3*86400;d+=!previous||p[0]-previous[0]>gap?`M${x},${y}`:`H${x}V${y}`;previous=p;}
        node('path',{d,class:'chart-line','data-platform':s.platform});
        s.marker=node('circle',{cx:0,cy:0,r:3.5,class:'chart-dot','data-platform':s.platform});
      }
      function nearest(stamp){
        let low=0,high=values.length-1;
        while(low<high){const mid=Math.floor((low+high)/2);if(values[mid][0]<stamp)low=mid+1;else high=mid;}
        return low>0&&stamp-values[low-1][0]<=values[low][0]-stamp?low-1:low;
      }
      function select(position,remember=false){
        const p=values[position];slider.value=String(p[0]);const labels=[];
        for(const s of series){
          let low=0,high=s.values.length;
          while(low<high){const mid=Math.floor((low+high)/2);if(s.values[mid][0]<=p[0])low=mid+1;else high=mid;}
          const sample=s.values[low-1];
          if(s.marker){s.marker.style.display=sample?'':'none';if(sample){const [x,y]=xy(sample,s.column);s.marker.setAttribute('cx',String(x));s.marker.setAttribute('cy',String(y));}}
          labels.push((s.name?s.name+' ':'')+(sample?number(sample[s.column])+(index===3?' 个':' 次'):'暂无记录'));
        }
        output.textContent=date(p[0])+' · '+labels.join(' · ');slider.setAttribute('aria-valuetext',output.textContent);
        if(remember)selectedTimes.set(index,p[0]===end?null:p[0]);
      }
      slider.oninput=()=>select(nearest(Number(slider.value)),true);
      slider.onkeydown=event=>{
        const offsets={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1,PageDown:-10,PageUp:10};
        if(!(event.key in offsets)&&event.key!=='Home'&&event.key!=='End')return;
        event.preventDefault();const current=nearest(Number(slider.value));
        select(event.key==='Home'?0:event.key==='End'?values.length-1:Math.max(0,Math.min(values.length-1,current+offsets[event.key])),true);
      };
      const selected=selectedTimes.get(index);select(selected==null?values.length-1:nearest(selected));
      node('title',{}).textContent=card.querySelector('h3').textContent+'，共'+values.length+'个真实采样点。可拖动下方时间轴查看数值。';
    });
  }
  async function update(){
    if(fetching||document.hidden)return;fetching=true;
    try{
      const r=await fetch('/site-stats.json',{cache:'no-store',credentials:'omit'});if(!r.ok)throw Error();const data=await r.json();
      const macos=data.downloads?.macos;
      if(data.schema!==1||![data.pageViews,data.downloads?.total,data.downloads?.portable,data.downloads?.installer].every(valid)||(macos!=null&&!valid(macos))||data.downloads.total!==data.downloads.portable+data.downloads.installer+(macos??0))throw Error();
      const counts=[data.pageViews,data.downloads.total,data.webUses,data.github?.count];keys.forEach((key,i)=>panel.querySelector('[data-stat="'+key+'"]').textContent=valid(counts[i])?number(counts[i]):'—');
      lastUpdate=Date.parse(data.updatedAt);panel.querySelector('[data-stats-since]').textContent='从 '+new Date(data.startedAt).toLocaleDateString('zh-CN',{timeZone:'Asia/Shanghai'})+' 开始记录';
      panel.querySelector('[data-stats-updated]').textContent='统计更新 '+date(lastUpdate/1000)+(Date.now()-lastUpdate>180000?' · 数据暂未更新':'')+(data.github?.updatedAt?' · Star采样 '+date(Date.parse(data.github.updatedAt)/1000):' · Star尚无采样');
      const h=await fetch('/site-stats-history.json',{cache:'no-store',credentials:'omit'});if(!h.ok)throw Error();const series=await h.json();
      if(series.schema!==1||!Array.isArray(series.points))throw Error();
      history=series.points.filter(p=>Array.isArray(p)&&(p.length===5||p.length===6)&&valid(p[0])&&p.slice(1).every(v=>v===null||valid(v))).sort((a,b)=>a[0]-b[0]);charts();
    }catch{panel.querySelector('[data-stats-updated]').textContent=lastUpdate?'暂时无法更新，显示 '+date(lastUpdate/1000)+' 的记录。':'统计暂时无法读取，请稍后再看。';}
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
    intro: {bvid:'BV1WWpr67EkX', title:'PhoneticToolbox 介绍'},
    spectrogram: {bvid:'BV1RuJm6gEWE', title:'语谱图转音频 · 第一集'},
  };
  let selected = '';
  function play(key) {
    const video = videos[key];
    if (!video || selected === key) return;
    selected = key;
    const player = document.createElement('iframe');
    player.title = video.title + ' · B 站视频';
    player.src = 'https://player.bilibili.com/player.html?bvid='+video.bvid+'&page=1&autoplay=0';
    player.allow = 'fullscreen; autoplay; encrypted-media; picture-in-picture';
    player.allowFullscreen = true;
    player.referrerPolicy = 'strict-origin-when-cross-origin';
    screen.replaceChildren(player);
    channels.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.tutorial === key)));
    episodes.hidden = key !== 'spectrogram';
    episodes.querySelector('[data-tutorial-episode]').setAttribute('aria-pressed', String(key === 'spectrogram'));
    external.href = 'https://www.bilibili.com/video/'+video.bvid+'/';
    external.setAttribute('aria-label', '去 B 站播放'+video.title);
    status.textContent = video.title;
    if (document.fullscreenEnabled && screen.requestFullscreen) fullscreen.hidden = false;
  }
  channels.forEach(button => button.addEventListener('click', () => play(button.dataset.tutorial)));
  episodes.querySelectorAll('[data-tutorial-episode]').forEach(button => button.addEventListener('click', () => play(button.dataset.tutorialEpisode)));
  fullscreen.addEventListener('click', async () => {
    try { await screen.requestFullscreen(); status.textContent = ''; }
    catch { status.textContent = '可使用视频内的全屏按钮，或前往 B 站观看。'; }
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
      status.textContent = count ? '显示 '+count+' 条来源' : '没有匹配的来源，可调整搜索词或查看范围。';
      results.scrollTop = 0;
    }
    query.addEventListener('input', filter);
    module.addEventListener('change', filter);
    catalog.addEventListener('toggle', () => {
      catalog.querySelector(':scope > summary span').textContent = catalog.open ? '收起列表' : '展开查看';
    });
    catalog.querySelectorAll('.reference-copy').forEach(button => {
      button.hidden = false;
      button.addEventListener('click', async () => {
        const citation = button.closest('.reference-source').querySelector('.reference-citation');
        try { await navigator.clipboard.writeText(citation.textContent); status.textContent = '引用已复制'; }
        catch {
          const range = document.createRange(); range.selectNodeContents(citation);
          const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
          status.textContent = '引用文字已选中，请复制。';
        }
      });
    });
  });
})();
