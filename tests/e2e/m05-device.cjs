const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','tests/support/m05_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-host.html']},plugins:[{name:'m05-host-transport-test',configureServer(s){s.middlewares.use('/__m05_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter;pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...JSON.parse(raw),id})+'\n')})})}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:false}),page=await browser.newPage({viewport:{width:1440,height:1000},permissions:['camera','microphone']}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(45000);
 const click=name=>page.getByRole('button',{name,exact:true}).click(),open=()=>page.locator('nav').getByRole('button',{name:'唇形提取',exact:true}).click();
 let success=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-host.html');await open();await page.locator('.lip-page').waitFor();
  await page.evaluate(()=>{const banner=document.createElement('div');banner.id='qa-device-instructions';Object.assign(banner.style,{position:'fixed',top:'0',left:'0',right:'0',zIndex:'99999',background:'#142638',color:'white',padding:'12px',textAlign:'center',fontSize:'18px'});document.body.append(banner)});
  const banner=message=>page.evaluate(message=>{document.getElementById('qa-device-instructions').textContent=message},message);
  const devices=await page.evaluate(async()=>{const devices=await navigator.mediaDevices.enumerateDevices();return devices.map(d=>({kind:d.kind,label:d.label}))});
  await fs.writeFile(path.join(out,'device-inventory.json'),JSON.stringify(devices,null,2));
  for(const mode of (process.env.M05_DEVICE_MODES??'raw,realtime,record_then_analyze').split(',')){
   await page.getByLabel('模式',{exact:false}).first().selectOption(mode);
   await page.getByLabel('请求帧率').fill(mode==='record_then_analyze'?'60':'30');
   for(let second=5;second>0;second--){await banner('本地设备验收：'+mode+' 模式，'+second+' 秒后录制。仅本机保存，不上传。');await page.waitForTimeout(1000);}
   await click('开始录制');await page.waitForFunction(()=>document.querySelector('.lip-state')?.textContent.includes('阶段 recording'));
   for(let second=12;second>0;second--){await banner('正在录制 '+mode+'，剩余 '+second+' 秒。正脸说话 → 侧脸 → 快速张合嘴 → 短暂遮挡嘴部。');await page.waitForTimeout(1000);if(second===6&&mode==='realtime')await page.locator('.lip-video').screenshot({path:path.join(out,'corrected-live-overlay.png')});}
   await click('停止并收尾');await page.waitForFunction(()=>document.querySelector('.lip-state')?.textContent.includes('阶段 ready'));
   await click('保存原始录制');await page.getByText('原始录制写入完成并核对大小。',{exact:true}).waitFor();
   await click('保存候选参数与时间记录');await page.getByText('候选参数与时间元数据已保存。',{exact:true}).waitFor();
   checks.push('physical Chrome '+mode+': captured 12 seconds, MediaRecorder finalized, actual native save completed');
   await fs.writeFile(path.join(out,mode+'-state.txt'),await page.locator('.lip-state').innerText());
  }
  await banner('三种模式录制与本地保存完成，即将关闭本任务窗口。');await page.waitForTimeout(1500);
  assert.deepEqual(errors,[]);success=true;
 }catch(e){await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});throw e;}
 finally{await fs.writeFile(path.join(out,'device-report.json'),JSON.stringify({success,checks,errors,scope:'Windows Chrome physical default camera/microphone and actual production capture; substitute QWebChannel transport; private local recording explicitly authorized 2026-09-27'},null,2));console.log(JSON.stringify({success,checks}));await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1});
