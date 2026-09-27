// M12: project-owned isolated Chrome with real file capabilities, no EXE/DDL.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m12_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let counter=0,readyResolve,readyReject;
 const ready=new Promise((r,j)=>{readyResolve=r;readyReject=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)readyResolve(v);else{pending.get(v.id)?.(v);pending.delete(v.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',readyReject);worker.once('exit',code=>readyReject(Error('RPC exited '+code)));
 const {out}=await ready;const rpc=data=>new Promise(resolve=>{const id=++counter;pending.set(id,resolve);worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:46212},optimizeDeps:{entries:['tests/m12-live.html']},plugins:[{name:'m12-owned-rpc',configureServer(s){s.middlewares.use('/__m12',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',async()=>{const value=await rpc(JSON.parse(body));res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),context=await browser.newContext({viewport:{width:1600,height:1100},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[];
 await page.clock.install();
 page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const loaded=()=>page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
 const gridClick=async(time,row=0)=>{const box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*time/2,box.y+box.height*(row===0?.22:.73));};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m12-live.html');await require('./m12-r1-flow.cjs')({page,click,loaded,out,rpc,checks});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({out,checks},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});await fs.writeFile(path.join(out,'failure.txt'),String(error)+'\n'+await page.locator('body').innerText()+'\nErrors:'+JSON.stringify(errors));console.error('M12 evidence:',out);throw error;}
 finally{await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
