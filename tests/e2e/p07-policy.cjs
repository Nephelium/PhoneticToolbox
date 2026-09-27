// Actual built Vue UI with synthetic HTTP responses; no database or credentials.
const path=require('node:path');
const fs=require('node:fs/promises');
const assert=require('node:assert/strict');
const os=require('node:os');

(async()=>{
  const root=path.resolve(__dirname,'../..');
  const out=path.join(root,'output/validation/p07-policy/ui');
  await fs.mkdir(out,{recursive:true});
  const {preview}=await import('../../frontend/node_modules/vite/dist/node/index.js');
  const server=await preview({root:path.join(root,'frontend'),logLevel:'error',preview:{host:'127.0.0.1',port:0}});
  const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
  const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  let browser;
  const errors=[],checks=[];
  try {
    browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
    const owner='00000000-0000-4000-8000-000000000001', project='00000000-0000-4000-8000-000000000002';
    const asset={id:'00000000-0000-4000-8000-000000000003',project_id:project,name:'旧结果 ɑ.zip',kind:'archive',
      state:'ready',size_bytes:32,reserved_bytes:0,expected_bytes:32,sha256:'a'.repeat(64),created_at:100,
      expires_at:Date.now()/1000+604800,error_code:null,policy_version:1};
    for(const state of ['unmigrated','over','ready']) {
      const context=await browser.newContext({viewport:{width:1280,height:900}});
      const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
      // Production host mounts the relative build under /server/. Vite preview
      // mounts at root, so map just those static requests to the same artifacts.
      await page.route('**/server/assets/**',route=>route.continue({url:route.request().url().replace('/server/assets/','/assets/')}));
      const legacy=state==='unmigrated';
      const usage={quota_bytes:legacy?5_000_000_000:1_000_000_000,used_bytes:state==='over'?1_000_000_001:32,
        reserved_bytes:state==='over'?8:0,available_bytes:state==='over'?0:(legacy?5_000_000_000:1_000_000_000)-32,
        ready:true,frozen:false,...(legacy?{}:{policy_version:2,retention_seconds:259200,over_quota:state==='over'})};
      await page.route('**/api/v1/**',async route=>{
        const url=new URL(route.request().url());const p=url.pathname;
        const body=p.endsWith('/auth/me')?{user:{id:owner,username:'synthetic'},csrf_token:'synthetic-only'}:
          p.endsWith('/projects')?{projects:[{id:project,name:'P07 合成项目',created_at:'2026-09-26T00:00:00Z'}]}:
          p.endsWith('/storage/usage')?usage:p.endsWith('/assets')?{assets:[asset]}:
          p.endsWith('/delete-impact')?{active_jobs:[]}:
          p.endsWith('/capabilities')?{task_operations:['archive_zip','storage_check'],storage_operations:[]}:
          p.endsWith('/jobs')?{jobs:[]}:{};
        await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
      });
      await page.goto(origin+'/server/');
      try { await page.getByRole('button',{name:/P07 合成项目/}).click({timeout:10000}); }
      catch(error) {
        await fs.writeFile(path.join(out,state+'-failure.txt'),JSON.stringify({errors,body:await page.locator('body').ariaSnapshot()},null,2));
        throw error;
      }
      const panel=page.locator('.project-storage');
      await panel.locator('.storage-meter').waitFor();
      assert.match(await panel.innerText(),/1,000,000,000 字节/);
      assert.match(await panel.innerText(),/259,200 秒/);
      assert.equal(await panel.locator('input[type=file]').isDisabled(),state!=='ready');
      assert.equal(await panel.getByRole('button',{name:'运行存储流程检查'}).isDisabled(),state!=='ready');
      assert.equal(await panel.getByRole('button',{name:'展开 ZIP'}).isDisabled(),state!=='ready');
      assert.equal(await panel.getByRole('link',{name:'下载',exact:true}).count(),1);
      assert.equal(await panel.getByRole('button',{name:'删除',exact:true}).isEnabled(),true);
      if(legacy) assert.match(await panel.innerText(),/5.00 GB[\s\S]*尚未确认启用/);
      if(state==='over') assert.match(await panel.innerText(),/超过额度/);
      await panel.screenshot({path:path.join(out,state+'-light.png'),animations:'disabled'});
      await page.getByLabel('配色主题').selectOption('dark');
      await page.setViewportSize({width:390,height:844});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
      await panel.screenshot({path:path.join(out,state+'-dark-narrow.png'),animations:'disabled'});
      await panel.getByRole('button',{name:'删除',exact:true}).click();
      await page.getByRole('dialog').waitFor();
      assert.equal(await page.getByRole('button',{name:'确认删除',exact:true}).isEnabled(),true);
      await page.getByRole('button',{name:'保留文件',exact:true}).click();
      checks.push({state,upload_and_generation_gated:true,download_and_delete_available:true,no_horizontal_overflow:true});
      await context.close();
    }
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(out,'checks.json'),JSON.stringify({checks,errors},null,2));
    console.log(JSON.stringify({checks,errors}));
  } finally {
    if(browser) await browser.close();
    await new Promise(resolve=>server.httpServer.close(resolve));
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
