// Independent Chrome only. Never use or close a Codex in-app browser tab.
// Credentials arrive on stdin and are never written to reports or screenshots.
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
let redactions = [];

(async () => {
  let input = '';
  for await (const block of process.stdin) input += block;
  const config = JSON.parse(input);
  redactions = config.people.map(person => person.password);
  const { chromium } = require(config.playwright);
  const browser = await chromium.launch({executablePath:config.browser,headless:true});
  const errors = [];
  const filename = 'P07 浏览器 ɑ.bin';
  const data = Buffer.alloc(256*1024+37, 71);
  let context;
  try {
    context = await browser.newContext({viewport:{width:1280,height:920}});
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(config.origin+'/server/');
    await fs.writeFile(path.join(config.output,'real-login-snapshot.txt'),await page.locator('body').ariaSnapshot());
    const login = async (person) => {
      await page.getByLabel('登录名',{exact:true}).fill(person.username);
      await page.getByLabel('密码',{exact:true}).fill(person.password);
      await page.getByRole('button',{name:'登录',exact:true}).click();
      await page.getByRole('heading',{name:'你的研究项目',exact:true}).waitFor();
    };
    await login(config.people[0]);
    await fs.writeFile(path.join(config.output,'real-project-snapshot.txt'),await page.locator('body').ariaSnapshot());
    await page.getByRole('button',{name:/^P07 真实存储联合检查 /}).click();
    await page.getByRole('heading',{name:'项目文件',exact:true}).waitFor();
    await page.getByLabel('选择上传文件',{exact:true}).setInputFiles({name:filename,mimeType:'application/octet-stream',buffer:data});
    await page.getByRole('button',{name:'上传文件',exact:true}).click();
    await page.getByText('上传完成。文件保留 7 天，下载不会延长到期时间。',{exact:true}).waitFor();
    const row = page.locator('.storage-files li').filter({has:page.getByText(filename,{exact:true})});
    const download = row.getByRole('link',{name:'下载',exact:true});
    const href = await download.getAttribute('href');
    assert(href.includes('expected_account='+config.people[0].owner));
    const response = await context.request.get(config.origin+href);
    assert.equal(response.status(),200);
    assert.deepEqual(await response.body(),data);
    assert(response.headers()['content-disposition'].startsWith('attachment;'));
    await fs.writeFile(path.join(config.output,'real-storage-snapshot.txt'),await page.locator('body').ariaSnapshot());
    await page.locator('.storage-sort select').selectOption('size');
    await page.locator('.storage-files li').first().getByText(filename,{exact:true}).waitFor();
    await page.getByRole('heading',{name:'项目文件',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(config.output,'real-storage-light.png'),fullPage:true});
    await page.getByLabel('配色主题').selectOption('dark');
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(config.output,'real-storage-dark-narrow.png'),fullPage:true});
    const layout = await page.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth}));
    assert.equal(layout.document,layout.viewport);
    let joint = null;
    if(config.joint) {
      const waitJob = async id => {
        for(let i=0;i<150;i++) {
          const result = await context.request.get(config.origin+'/api/v1/jobs/'+id);
          assert.equal(result.status(),200);
          const job=await result.json();
          if(['succeeded','failed','cancelled','interrupted'].includes(job.state))return job;
          await new Promise(resolve=>setTimeout(resolve,100));
        }
        throw Error('File worker did not finish within the bounded UI check');
      };
      const clickJob = async button => {
        const pending=page.waitForResponse(r=>r.url()===config.origin+'/api/v1/jobs' && r.request().method()==='POST');
        await button.click();
        const result=await pending;assert.equal(result.status(),201);
        const job=await waitJob((await result.json()).id);
        assert.equal(job.state,'succeeded',job.error_code);
        await page.getByRole('button',{name:'刷新空间',exact:true}).click();
        return job;
      };
      await page.getByRole('checkbox',{name:'选择 '+filename,exact:true}).check();
      const archive=await clickJob(page.getByRole('button',{name:'打包所选文件（1）',exact:true}));
      const zipRow=page.locator('.storage-files li').filter({has:page.getByText('研究文件.zip',{exact:true})});
      await zipRow.getByRole('button',{name:'展开 ZIP',exact:true}).waitFor();
      const extract=await clickJob(zipRow.getByRole('button',{name:'展开 ZIP',exact:true}));
      assert.equal(extract.result_manifest.files[0].sha256,require('node:crypto').createHash('sha256').update(data).digest('hex'));
      const generated=await clickJob(page.getByRole('button',{name:'运行存储流程检查',exact:true}));
      assert.equal(generated.result_manifest.files.length,2);
      await page.getByText('存储流程检查-1.bin',{exact:true}).waitFor();
      await fs.writeFile(path.join(config.output,'joint-storage-snapshot.txt'),await page.locator('body').ariaSnapshot());
      await page.screenshot({path:path.join(config.output,'joint-storage-dark-narrow.png'),fullPage:true,animations:'disabled'});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
      await page.getByLabel('配色主题').selectOption('light');await page.setViewportSize({width:1280,height:920});
      await page.screenshot({path:path.join(config.output,'joint-storage-light.png'),fullPage:true,animations:'disabled'});
      await page.getByLabel('配色主题').selectOption('dark');await page.setViewportSize({width:390,height:844});

      // Concurrent mutations originate in two real tabs sharing the account.
      const tab=await context.newPage();await tab.goto(config.origin+'/server/');
      const session=await (await context.request.get(config.origin+'/api/v1/auth/me')).json();
      const assets=await (await context.request.get(config.origin+'/api/v1/assets?project_id='+config.people[0].project)).json();
      const raceSource=assets.assets.find(a=>a.name==='合成 ɑ.bin');assert(raceSource);
      const mutation=async (target,url,body,method)=>target.evaluate(async args=>{
        const response=await fetch(args.url,{method:args.method,headers:{'Content-Type':'application/json','X-CSRF-Token':args.csrf,'X-PTB-Account':args.owner},body:args.body?JSON.stringify(args.body):undefined});
        return {status:response.status,body:await response.json()};
      },{url,body,method,csrf:session.csrf_token,owner:config.people[0].owner});
      const race=await Promise.all([
        mutation(page,'/api/v1/jobs',{project_id:config.people[0].project,operation:'archive_zip',idempotency_key:require('node:crypto').randomUUID(),config:{inputs:[raceSource.id]}},'POST'),
        mutation(tab,'/api/v1/assets/'+raceSource.id,null,'DELETE')
      ]);
      assert.equal(race[1].status,200);assert.equal(race[1].body.state,'deleted');
      let raceState='input_rejected';
      if(race[0].status===201) {
        const done=await waitJob(race[0].body.id);raceState=done.state;
        assert(['succeeded','cancelled','failed'].includes(done.state));
        assert.equal(Boolean(done.result_manifest),done.state==='succeeded');
      } else assert.equal(race[0].status,410);
      if(!await page.getByRole('heading',{name:'项目文件',exact:true}).count()) {
        await page.getByRole('button',{name:/^P07 真实存储联合检查 /}).click();
      }
      joint={archive:true,extract_exact_hash:true,two_results:true,two_tab_race:raceState};
    }
    await row.getByRole('button',{name:'删除',exact:true}).click();
    await page.getByRole('dialog').waitFor();
    await page.screenshot({path:path.join(config.output,'real-storage-delete-narrow.png'),fullPage:true});
    await page.getByRole('button',{name:'保留文件',exact:true}).click();
    assert.equal((await context.request.get(config.origin+href)).status(),200);
    await row.getByRole('button',{name:'删除',exact:true}).click();
    await page.getByRole('button',{name:'确认删除',exact:true}).click();
    await page.getByText('文件已删除，占用空间已释放。',{exact:true}).waitFor();
    await row.waitFor({state:'detached'});
    assert.equal(await row.count(),0);
    assert.equal((await context.request.get(config.origin+href)).status(),410);
    await page.getByRole('button',{name:'退出登录',exact:true}).click();
    await page.getByRole('heading',{name:'登录研究工作台',exact:true}).waitFor();
    await login(config.people[1]);
    const other = await context.request.get(config.origin+href);
    assert.equal(other.status(),409); // stale link expected-account guard
    assert.equal((await context.request.get(config.origin+href.split('?')[0])).status(),404);
    assert.equal(await page.getByText(filename,{exact:true}).count(),0);
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(config.output,'real-ui-validation.json'),JSON.stringify({
      scope:'Real PG and marked storage root; independent Chrome. Download link validated via browser HTTP client, not OS save dialog.',
      upload_bytes:data.length,download_bytes_verified:true,cancel_delete_preserves:true,confirmed_delete_unavailable:true,
      stale_account_link_rejected:true,light_dark:true,narrow:layout,page_errors:errors,joint
    },null,2));
    console.log('P07 real storage browser checks passed.');
  } finally {
    if(context) await context.close();
    await browser.close();
    console.log('Owned independent Chrome stopped.');
  }
})().catch(error=>{
  let message = String(error.stack || error);
  for (const secret of redactions) message = message.split(secret).join('[redacted]');
  console.error(message);
  process.exitCode=1;
});
