"""Verify the shared manual in built Vue UI with an owned Chrome profile."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]
CHECK = r"""
const path=require('node:path'),fs=require('node:fs/promises'),os=require('node:os'),assert=require('node:assert/strict');
const preferences=require(path.join(process.cwd(),'tests/e2e/preferences-fixture.cjs')).createPreferencesFixture();
(async()=>{
 const root=process.cwd(),out=process.argv[2],scratch=process.argv[3],captureMode=process.argv[4]==='capture';
 const {preview}=await import(require('node:url').pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')).href);
 const server=await preview({root:path.join(root,'frontend'),build:{outDir:path.join(scratch,'frontend')},logLevel:'error',preview:{host:'127.0.0.1',port:0}});
 const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context,signed=!captureMode;const errors=[],checks=[],captures=[];
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{headless:false,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1707,height:1067},deviceScaleFactor:1.5,colorScheme:'light'});
  await context.addInitScript(()=>{localStorage.setItem('ptb.v3.account-theme','light');localStorage.setItem('ptb.v3.theme',JSON.stringify('light'));localStorage.setItem('ptb.v3.palette',JSON.stringify('absolutely'));localStorage.setItem('ptb.v3.pageScale','100');});
  await context.route('**/server/assets/**',route=>route.continue({url:route.request().url().replace('/server/assets/','/assets/')}));
  await context.route('**/server/manual/**',route=>route.continue({url:route.request().url().replace('/server/manual/','/manual/')}));
  const owner='00000000-0000-4000-8000-000000000001',project='00000000-0000-4000-8000-000000000010';
  preferences.seed(owner,'APP',{theme:'light',palette:'absolutely',pageScale:100});
  await context.route('**/api/v1/**',async route=>{
   const p=new URL(route.request().url()).pathname,now=Date.now()/1000;let body={},status=200;
   const session={user:{id:owner,username:'synthetic',role:'user',email_verified:true},csrf_token:'synthetic-only'};
   const prefs=preferences.reply(route.request(),owner);
   if(prefs){status=prefs.status;body=prefs.body;}
   else if(p.endsWith('/auth/options'))body={registration:true,password_recovery:true,ephemeral_workspaces:true};
   else if(p.endsWith('/auth/me')){body=signed?session:{detail:'authentication_required'};status=signed?200:401;}
   else if(p.endsWith('/auth/challenge'))body={csrf_token:'synthetic-only'};
   else if(p.endsWith('/auth/login')){signed=true;body=session;}
   else if(p.endsWith('/workspaces/default-project'))body={id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'};
   else if(p.endsWith('/workspaces/presence'))body={online_accounts:1,window_seconds:90,measured_at:now};
   else if(p.endsWith('/workspaces/capacity-request'))body=null;
   else if(p.includes('/workspaces/'))body={id:project,generation:1,last_seen:now,expires_at:now+1800,grace_seconds:1800};
   else if(p.endsWith('/projects'))body={projects:[{id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'}]};
   else if(p.endsWith('/storage/usage'))body={quota_bytes:1000000000,used_bytes:0,reserved_bytes:0,available_bytes:1000000000,ready:true,frozen:false,policy_version:2,workspace_policy_version:1,retention_seconds:1800,over_quota:false};
   else if(p.endsWith('/assets'))body={assets:[]};
   else if(p.endsWith('/jobs'))body={jobs:[]};
   else if(p.endsWith('/capabilities'))body={task_operations:['lpc_spectrum'],storage_operations:['upload','download'],algorithms:[]};
   await route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin+'/server/');
  async function capture(name,caption){
   await page.evaluate(async()=>{await document.fonts.ready;await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
   const metrics=await page.evaluate(()=>({dpr:devicePixelRatio,theme:document.documentElement.dataset.theme,palette:document.documentElement.dataset.palette,pageZoom:getComputedStyle(document.documentElement).zoom,viewport:[innerWidth,innerHeight]}));
   assert.equal(metrics.dpr,1.5);assert.equal(metrics.theme,'light');assert.equal(Number(metrics.pageZoom),1);
   const file=path.join(scratch,name+'.png');await page.screenshot({path:file});captures.push({name,file,caption,...metrics});
  }
  if(captureMode){
   await page.getByRole('button',{name:'注册账号',exact:true}).waitFor();
   await capture('login','在登录页输入登录名和密码。首次使用可注册账号，忘记密码时从这里找回。');
   await page.getByRole('button',{name:'注册账号',exact:true}).click();
   await page.getByRole('heading',{name:'创建研究者账号',exact:true}).waitFor();
   await capture('register','填写登录名、邮箱和密码后发送验证邮件，完成邮箱验证再返回登录。');
   await page.getByRole('button',{name:'返回登录',exact:true}).click();
   await page.getByLabel('登录名',{exact:true}).fill('manual-demo');await page.getByLabel('密码',{exact:true}).fill('manual-demo-only');
   await page.getByRole('button',{name:'登录',exact:true}).click();
   await page.getByRole('button',{name:'文件与空间',exact:true}).click();
   await page.getByRole('heading',{name:'项目文件',exact:true}).waitFor();
   await page.getByRole('progressbar',{name:'账号文件空间'}).waitFor();
   await capture('files','文件与空间集中管理上传材料和生成结果，也可查看用量、下载文件或退出登录。');
   await page.getByRole('button',{name:'任务记录',exact:true}).click();
   await page.getByText('暂无任务',{exact:true}).waitFor();
   await capture('tasks','任务记录按当前项目显示处理进度与任务记录，新项目的列表为空。');
   await page.getByRole('button',{name:'关闭对话框',exact:true}).click();
   await page.getByRole('button',{name:'容量与扩容',exact:true}).click();
   await page.getByLabel('申请总额度（MB）',{exact:true}).waitFor();
   await capture('capacity','查看个人可用空间，填写申请总额度、期限和用途，提交后等待管理员审核。');
   await page.getByRole('button',{name:'关闭对话框',exact:true}).click();
   await page.getByRole('button',{name:'设置',exact:true}).click();
   await page.getByRole('button',{name:'浅色',exact:true}).click();
   await capture('settings','从左下角设置打开外观页，调整配色、浅深模式与页面缩放，当前选择即时生效。');
   await fs.writeFile(path.join(out,'captures.json'),JSON.stringify({scope:'Actual built Vue UI; synthetic account, empty files and tasks; Windows AppliedDPI 144; Chrome effective DPR 1.5; page zoom 100%',captures},null,2));
  }
  await page.getByRole('button',{name:'使用说明',exact:true}).click();
  await page.getByRole('heading',{name:/网页版使用入门/}).waitFor();
  await page.getByText('换浏览器或换设备登录后，工作台会先恢复已保存的设置',{exact:false}).waitFor();
  const chapters=page.locator('.manual-toc button[data-manual-chapter]');assert.equal(await chapters.count(),21);
  assert.equal(await chapters.first().getAttribute('data-manual-chapter'),'web-introduction');
  assert.equal(await page.locator('.manual-toc button[data-manual-chapter^="m"]').count(),18);
  checks.push('global-help-opens-web-introduction-with-all-18-modules');
  for(const theme of ['light','dark']){
   await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   await page.screenshot({path:path.join(out,'introduction-'+theme+'.png')});
  }
  const search=page.getByRole('searchbox',{name:'搜索说明书'});await search.fill('扩容');
  await page.locator('.manual-search-results button').first().click();
  await page.getByRole('heading',{name:/网页版使用入门/}).waitFor();checks.push('web-introduction-is-searchable');
  await search.fill('');await page.locator('.manual-toc button[data-manual-chapter="m11"]').click();
  await page.getByRole('heading',{name:/MFA 自动标注/}).waitFor();checks.push('desktop-only-module-chapter-remains-readable');
  await page.locator('.manual-toc button[data-manual-chapter="getting-started"]').click();
  await page.getByRole('heading',{name:/开始使用与共同操作/}).waitFor();checks.push('shared-common-chapter-opens');
  await page.getByRole('button',{name:'使用说明',exact:true}).click();
  await page.getByRole('heading',{name:/网页版使用入门/}).waitFor();checks.push('reopening-general-help-restores-web-entry');
  const media=page.locator('.manual-document img');
  if(!captureMode)assert.equal(await media.count(),6);
  for(const img of await media.all()){await img.scrollIntoViewIfNeeded();await img.evaluate(async node=>{if(!node.complete)await node.decode();});assert.deepEqual(await img.evaluate(node=>[node.naturalWidth,node.naturalHeight]),[2560,1600]);}
  if(await media.count())checks.push('all-introduction-screenshots-decode-in-reader');
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'result.json'),JSON.stringify({scope:'Windows built Vue UI; independent visible Chrome; synthetic account HTTP only',browser:context.browser().version(),checks,errors},null,2));
  console.log(JSON.stringify({out,checks,errors}));
 }catch(error){const page=context?.pages().at(-1);if(page){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());}throw error;}
 finally{await context?.close();await new Promise(resolve=>server.httpServer.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
"""


def register_captures(out: Path):
    """Register the requested light screenshots without resizing UI pixels."""
    from PIL import Image
    report = json.loads((out / 'captures.json').read_text('utf-8'))
    project_path = ROOT / 'manual/project.json'
    chapter_path = ROOT / 'manual/chapters/web-introduction.json'
    project = json.loads(project_path.read_text('utf-8'))
    chapter = json.loads(chapter_path.read_text('utf-8'))
    images = {}
    for capture in report['captures']:
        name = capture['name']
        identity = 'web-introduction-' + name + '-light'
        relative = 'assets/software-only/web-introduction/' + identity + '.png'
        target = ROOT / 'manual' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(capture['file']) as screenshot:
            raw_size = screenshot.size
            assert raw_size in ((2560, 1600), (2561, 1601)), raw_size
            screenshot.crop((0, 0, 2560, 1600)).save(target)
        capture.update(assetId=identity, path=relative, raw_image=list(raw_size), image=[2560, 1600],
                       sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        asset = dict(id=identity, path=relative, kind='image', mime='image/png',
                     sha256=capture['sha256'], width=2560, height=1600,
                     caption=capture['caption'], git=False, distribution='software-only',
                     sourceType='实际应用截图',
                     source='Windows当前源码独立Chrome网页，浅色Absolutely，DPR 1.5、页面100%；隔离演示账号和空文件/任务列表。',
                     capture=dict(report=(out / 'captures.json').relative_to(ROOT).as_posix(),
                                  dpr=1.5, pageZoom=1, rawImage=list(raw_size), image=[2560, 1600],
                                  crop='Only right/bottom rounding pixels; no resampling'))
        previous = next((i for i, value in enumerate(project['assets']) if value['id'] == identity), None)
        if previous is None:
            project['assets'].append(asset)
        else:
            project['assets'][previous] = asset
        images[name] = dict(type='image', attrs=dict(id='web-introduction-figure-' + name + '-light',
                           assetId=identity, caption=capture['caption'], alt=capture['caption'],
                           width='100%', align='center'))
    content = []
    section, paragraphs = '', 0
    for node in chapter['body']['content']:
        if node['type'] == 'image' and node.get('attrs', {}).get('assetId', '').startswith('web-introduction-'):
            continue
        content.append(node)
        if node['type'] == 'heading':
            section = node['attrs']['id']
            paragraphs = 0
            if section == 'web-introduction-account':
                content.append(images['login'])
        elif section == 'web-introduction-account' and node['type'] == 'orderedList':
            content.append(images['register'])
        elif node['type'] == 'paragraph':
            paragraphs += 1
            if section == 'web-introduction-files-jobs' and paragraphs in (1, 2):
                content.append(images['files' if paragraphs == 1 else 'tasks'])
            elif section == 'web-introduction-capacity' and paragraphs == 2:
                content.append(images['capacity'])
            elif section == 'web-introduction-settings-manual' and paragraphs == 1:
                content.append(images['settings'])
    chapter['body']['content'] = content
    for path, value in ((project_path, project), (chapter_path, chapter), (out / 'captures.json', report)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'screenshots': len(images), 'image': [2560, 1600], 'dpr': 1.5}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    with verification_run('web-manual', 'browser', 'Shared built manual: web introduction, search, original chapters, themes') as (out, scratch):
        with (out / 'build.log').open('w', encoding='utf-8') as log:
            subprocess.run(['node', str(ROOT / 'frontend/node_modules/vite/bin/vite.js'), 'build', '--outDir', str(scratch / 'frontend')], cwd=ROOT / 'frontend', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['node', '-', str(out), str(scratch), 'capture' if args.capture else 'verify'], input=CHECK, text=True, encoding='utf-8', cwd=ROOT, check=True)
        if args.capture:
            register_captures(out)
