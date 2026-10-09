import test, { after } from 'node:test';
import { createArtifacts } from './artifacts.mjs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { ProjectStore } from '../server/project-store.mjs';
import { launchStudio, sessionFile, openBrowser } from '../server/launcher.mjs';

const execute = promisify(execFile);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = await createArtifacts(root, 'launcher');
after(() => artifacts.finish());

test('Windows PowerShell parses the actual double-click launcher script', { skip: process.platform !== 'win32' }, async () => {
  const script = [
    '$tokens = $null; $parseErrors = $null',
    '$null = [System.Management.Automation.Language.Parser]::ParseFile($env:PTB_MANUAL_LAUNCHER_TEST, [ref]$tokens, [ref]$parseErrors)',
    'if ($parseErrors.Count) { $parseErrors | ForEach-Object { Write-Output ($_.ErrorId + " at line " + $_.Extent.StartLineNumber) }; exit 1 }',
  ].join('\n');
  const { stdout } = await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')], {
    windowsHide: true,
    env: { ...process.env, PTB_MANUAL_LAUNCHER_TEST: path.join(root, 'Start-Manual-Studio.ps1') },
  });
  assert.equal(stdout.trim(), '');
});

test('native Windows launcher opens a project with Chinese characters and spaces', { skip: process.platform !== 'win32', timeout: 20000 }, async () => {
  const output = path.join(artifacts.scratch, 'launcher-' + randomUUID());
  const directory = path.join(output, '作者工程 中文 空格');
  const store = new ProjectStore();
  await store.create(directory, '启动器验证工程');
  await store.release();
  const projectBefore = await fs.readFile(path.join(directory, 'project.json'));
  const infoPath = path.join(output, 'server-info.json');
  const errorPath = path.join(output, 'server-error.log');
  const pidPath = path.join(output, 'server-pid.txt');
  // Redirect only the process created by this launcher. --test-info suppresses
  // opening the user's browser and reports the local test session to a file.
  const script = [
    "$ErrorActionPreference = 'Stop'",
    'function Start-Process {',
    '  param([string]$FilePath, [string[]]$ArgumentList, [string]$WorkingDirectory, [string]$WindowStyle, [string]$RedirectStandardOutput, [string]$RedirectStandardError, [switch]$PassThru)',
    '  $process = Microsoft.PowerShell.Management\\Start-Process -FilePath $FilePath -ArgumentList ($ArgumentList + "--test-info") -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -RedirectStandardOutput $env:PTB_MANUAL_TEST_INFO -RedirectStandardError $env:PTB_MANUAL_TEST_ERROR -PassThru',
    '  [IO.File]::WriteAllText($env:PTB_MANUAL_TEST_PID, [string]$process.Id)',
    '  return $process',
    '}',
    '& $env:PTB_MANUAL_LAUNCHER_TEST -Project $env:PTB_MANUAL_TEST_PROJECT',
  ].join('\n');
  const launcher = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')], {
    windowsHide: true,
    stdio: 'ignore',
    env: { ...process.env, PTB_MANUAL_LAUNCHER_TEST: path.join(root, 'Start-Manual-Studio.ps1'), PTB_MANUAL_TEST_PROJECT: directory, PTB_MANUAL_TEST_INFO: infoPath, PTB_MANUAL_TEST_ERROR: errorPath, PTB_MANUAL_TEST_PID: pidPath },
  });
  await new Promise((resolve, reject) => { launcher.once('error', reject); launcher.once('exit', code => code === 0 ? resolve() : reject(new Error('Launcher exit code: ' + code))); });
  const childPid = Number((await fs.readFile(pidPath, 'utf8')).trim());
  assert.ok(Number.isSafeInteger(childPid) && childPid > 0, 'launcher returned its child process ID');
  let info;
  try {
    const deadline = Date.now() + 10000;
    while (!info && Date.now() < deadline) {
      try { info = JSON.parse((await fs.readFile(infoPath, 'utf8')).trim()); } catch {}
      if (!info) await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.ok(info, 'launcher started the editor service');
    const headers = { Authorization: 'Bearer ' + info.capability };
    const response = await fetch(info.origin + '/api/session', { headers });
    assert.equal(response.status, 200);
    const session = await response.json();
    assert.equal(session.open, true);
    assert.equal(session.snapshot.project.title, '启动器验证工程');
    const page = await fetch(info.origin + '/');
    assert.equal(page.status, 200);
    assert.match(await page.text(), /<div id="app"><\/div>/);
    assert.deepEqual(await fs.readFile(path.join(directory, 'project.json')), projectBefore);
  } finally {
    if (info) await fetch(info.origin + '/api/close', { method: 'POST', headers: { Authorization: 'Bearer ' + info.capability, Origin: info.origin } });
    const deadline = Date.now() + 5000;
    let alive = true;
    while (alive && Date.now() < deadline) {
      try { process.kill(childPid, 0); await new Promise(resolve => setTimeout(resolve, 100)); } catch { alive = false; }
    }
    if (alive) process.kill(childPid);
    assert.equal(alive, false, 'editor shutdown released its own child process');
  }
  assert.equal(JSON.parse(await fs.readFile(path.join(directory, '.studio/lock.json'), 'utf8')).released, true);
});

async function project(t) {
  const directory = path.join(artifacts.scratch, '启动会话-' + randomUUID());
  const store = new ProjectStore();
  await store.create(directory, '启动会话测试');
  t.after(() => store.release());
  return { directory, store };
}
test('repeated launch reopens exactly the authenticated owner without a second writer', async t => {
  const { directory, store } = await project(t);
  await store.release();
  const opened = [];
  const first = await launchStudio({ studioRoot: root, projectRoot: directory, openPage: async url => opened.push(url) });
  t.after(() => first.close());
  const lock = await fs.readFile(path.join(directory, '.studio/lock.json'));
  const second = await launchStudio({ studioRoot: root, projectRoot: directory, openPage: async url => opened.push(url) });
  assert.equal(second.reused, true);
  assert.equal(second.pid, first.pid);
  assert.equal(second.origin, first.origin);
  assert.equal(second.capability, first.capability);
  assert.deepEqual(opened, [first.url, first.url]);
  assert.deepEqual(await fs.readFile(path.join(directory, '.studio/lock.json')), lock);
});
test('a stale session entry starts a new server after the original lock is released', async t => {
  const { directory, store } = await project(t); await store.release();
  const first = await launchStudio({ studioRoot: root, projectRoot: directory, openPage: async () => {} });
  await first.close();
  const second = await launchStudio({ studioRoot: root, projectRoot: directory, openPage: async () => {} });
  t.after(() => second.close());
  assert.equal(second.reused, false);
  assert.notEqual(second.capability, first.capability);
});
test('legacy or tampered session records never stop or adopt another writer', async t => {
  const { directory, store } = await project(t);
  const lock = await fs.readFile(path.join(directory, '.studio/lock.json'));
  let opened = false;
  await assert.rejects(() => launchStudio({ studioRoot: root, projectRoot: directory, openPage: async () => { opened = true; } }), /先前的编辑器占用/);
  const file = await sessionFile(root, directory);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, JSON.stringify({ active: true, studioRoot: root, projectRoot: directory, pid: process.pid, lockId: store.lockId, origin: 'https://example.org', capability: 'a'.repeat(64) }));
  await assert.rejects(() => launchStudio({ studioRoot: root, projectRoot: directory, openPage: async () => { opened = true; } }), /先前的编辑器占用/);
  assert.equal(opened, false);
  assert.deepEqual(await fs.readFile(path.join(directory, '.studio/lock.json')), lock);
  assert.ok((await store.load()).project);
});
test('browser-open failure is reported and releases only the newly created server', async t => {
  const { directory, store } = await project(t); await store.release();
  await assert.rejects(() => launchStudio({ studioRoot: root, projectRoot: directory, openPage: async () => { throw new Error('浏览器启动失败'); } }), /浏览器启动失败/);
  assert.equal(JSON.parse(await fs.readFile(path.join(directory, '.studio/lock.json'), 'utf8')).released, true);
  await assert.rejects(() => openBrowser('https://example.org'), /浏览器地址无效/);
});
test('native PowerShell surfaces a locked-project failure instead of silently succeeding', { skip: process.platform !== 'win32', timeout: 15000 }, async t => {
  const { directory } = await project(t);
  const lock = await fs.readFile(path.join(directory, '.studio/lock.json'));
  await assert.rejects(() => execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', path.join(root, 'Start-Manual-Studio.ps1'), '-Project', directory], { windowsHide: true }), error => {
    assert.equal(error.code, 1);
    assert.match(error.stdout, /工程仍由先前的编辑器占用/);
    return true;
  });
  assert.deepEqual(await fs.readFile(path.join(directory, '.studio/lock.json')), lock);
});
