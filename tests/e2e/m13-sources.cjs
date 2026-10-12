// M13-ACADEMIC-SOURCES. Launched within verification_artifacts.py ownership.
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../..');
const [out, scratch] = process.argv.slice(2);

(async () => {
  assert(out && scratch, 'Use scripts/verify_m13_sources.py for owned scratch');
  const {createServer} = await import(pathToFileURL(path.join(root, 'frontend/node_modules/vite/dist/node/index.js')));
  const {chromium} = require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const sources = JSON.parse(await fs.readFile(path.join(root, 'frontend/src/generated/sources.json'), 'utf8'));
  const expected = sources.filter(s => s.modules.includes('M13') && s.acknowledgement_group === 'academic');
  const unt = expected.find(s => s.id === 'REF-M13-UNTPHESOCA-2019');
  assert.equal(expected.length, 10);
  const report = {success: false, cases: [], errors: [], externalRequests: []};
  const server = await createServer({root: path.join(root, 'frontend'), cacheDir: path.join(scratch, 'vite-cache'),
    server: {host: '127.0.0.1', port: 0, watch: null}, optimizeDeps: {noDiscovery: true, include: ['vue']}});
  let browser;
  try {
    await server.listen();
    browser = await chromium.launchPersistentContext(path.join(scratch, 'chrome-profile'), {
      executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
      viewport: {width: 1440, height: 900},
    });
    await browser.addInitScript(() => {
      window.qaCopied = '';
      Object.defineProperty(navigator, 'clipboard', {configurable: true,
        value: {writeText: async text => {window.qaCopied = text;}}});
    });
    const page = await browser.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('request', request => {
      if (/^https?:/.test(request.url()) && !request.url().startsWith(server.resolvedUrls.local[0])) report.externalRequests.push(request.url());
    });
    await page.goto(server.resolvedUrls.local[0]);
    await page.locator('.home-page').waitFor();
    const close = () => page.locator('dialog[open]').getByRole('button', {name: '关闭对话框', exact: true}).click();
    const rows = page.locator('dialog[open] .reference-row');
    const academic = () => page.getByRole('tab', {name: /^语音学与语言学学术来源/}).click();
    const search = page.getByRole('textbox', {name: '搜索来源'});
    for (const mode of ['light', 'dark']) {
      await page.locator('.utility-grid').getByRole('button', {name: '设置', exact: true}).click();
      await page.getByRole('button', {name: mode === 'light' ? '浅色' : '深色', exact: true}).click();
      for (const viewport of [{width: 1440, height: 900}, {width: 900, height: 700}]) {
        await page.setViewportSize(viewport);
        await page.locator('.nav-item').and(page.getByRole('button', {name: '汉字转国际音标', exact: true})).click();
        await page.locator('.mandarin-ipa-page').getByRole('button', {name: '方法与引用', exact: true}).click();
        await academic();
        assert.equal(await rows.count(), 10);
        for (const source of expected) {
          const row = rows.filter({has: page.getByRole('heading', {name: source.title, exact: true})});
          assert.equal(await row.count(), 1, source.id);
          assert((await row.innerText()).includes(source.citation), source.id + ' citation');
          assert((await row.innerText()).includes(source.reference_note), source.id + ' note');
        }
        const dictionary=rows.filter({has:page.getByRole('heading',{name:'重編國語辭典修訂本（2021年版）及異體字字典',exact:true})});
        assert.deepEqual(await dictionary.getByRole('link').allTextContents(),['修訂本','異體字字典']);
        const geometry = await page.locator('dialog[open]').evaluate(dialog => {
          const body = dialog.querySelector('.dialog-body'), rect = dialog.getBoundingClientRect();
          return {left: rect.left, right: rect.right, viewport: innerWidth,
            overflow: body.scrollWidth - body.clientWidth};
        });
        assert(geometry.left >= -1 && geometry.right <= geometry.viewport + 1 && geometry.overflow <= 1, JSON.stringify(geometry));
        await page.locator('dialog[open]').screenshot({path: path.join(out, `m13-sources-${mode}-${viewport.width}.png`)});
        await search.fill('UntPhesoca');
        assert.equal(await rows.count(), 1);
        assert.equal(await rows.getByRole('link', {name: '知乎原文', exact: true}).getAttribute('href'), unt.urls['知乎原文']);
        assert.equal(await rows.getByRole('link', {name: '作者网站', exact: true}).getAttribute('href'), unt.urls['作者网站']);
        await rows.getByRole('button', {name: '复制引用', exact: true}).click();
        assert.equal(await page.evaluate(() => window.qaCopied), unt.citation);
        await rows.screenshot({path: path.join(out, `unt-citation-${mode}-${viewport.width}.png`)});
        await search.fill('吴宗济');
        assert.equal(await rows.count(), 1);
        assert((await rows.innerText()).includes('尚未确认具体依据'));
        await search.fill('');
        await page.getByRole('tab', {name: /^软件与代码来源/}).click();
        assert.equal(await rows.count(), sources.filter(s => s.modules.includes('M13') && s.acknowledgement_group === 'software').length);
        await close();
        await page.locator('.utility-grid').getByRole('button', {name: '关于', exact: true}).click();
        await page.locator('dialog[open]').getByRole('button', {name: '开源与学术致谢', exact: true}).click();
        await page.getByRole('combobox', {name: '按模块查看致谢'}).selectOption('M13');
        await academic();
        assert.equal(await rows.count(), 10);
        await search.fill('UntPhesoca');
        assert.equal(await rows.count(), 1);
        await rows.getByRole('button', {name: '复制引用', exact: true}).click();
        assert.equal(await page.evaluate(() => window.qaCopied), unt.citation);
        await search.fill('');
        await page.getByRole('combobox', {name: '按模块查看致谢'}).selectOption('M10');
        assert.equal(await rows.count(), sources.filter(s => s.modules.includes('M10') && s.acknowledgement_group === 'academic').length);
        await close();
        report.cases.push({mode, viewport, academicSources: 10, moduleAndAboutMatch: true, untLinksAndCopy: true, geometry});
      }
    }
    assert.deepEqual(report.errors, []);
    assert.deepEqual(report.externalRequests, []);
    report.success = true;
  } finally {
    if (browser) await browser.close();
    await server.close();
    await fs.writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  }
  console.log(JSON.stringify(report, null, 2));
})().catch(error => {console.error(error); process.exitCode = 1;});
