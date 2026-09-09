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
    const row = page.locator('.storage-files li').filter({hasText:filename});
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
      stale_account_link_rejected:true,light_dark:true,narrow:layout,page_errors:errors
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
