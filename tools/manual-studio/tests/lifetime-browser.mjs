import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import { ProjectStore } from '../server/project-store.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'test-output', 'lifetime-browser-' + new Date().toISOString().replace(/[:.]/g, '-'));
const directory = path.join(output, '关闭标签页验证工程');
const store = new ProjectStore(), snapshot = await store.create(directory, '标签页关闭验证');
const chapter = { schemaVersion: 'ptb-manual-chapter/1', id: 'test-chapter', title: '编辑器会话测试', body: { type: 'doc', content: [{ type: 'paragraph', content: [{ type: 'text', text: '中文说明书编辑与关闭测试。' }] }] } };
snapshot.project.chapters.push({ id: chapter.id, title: chapter.title, path: 'chapters/test-chapter.json', status: 'draft' });
snapshot.chapters.push(chapter);
await store.save({ project: snapshot.project, chapters: snapshot.chapters, baseRevision: snapshot.revision });
await store.release();
const runtime = process.env.PTB_PLAYWRIGHT ?? path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const { chromium } = await import(pathToFileURL(runtime));
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const checks = [], errors = [], children = [];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function start() {
  const child = spawn(process.execPath, [path.join(root, 'server/start.mjs'), '--project', directory, '--test-info'], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  children.push(child);
  let stdout = '', stderr = '';
  child.stderr.on('data', bytes => { stderr += bytes.toString(); });
  const info = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Test launcher startup timed out')), 15000);
    child.stdout.on('data', bytes => { stdout += bytes.toString(); if (stdout.includes('\n')) { clearTimeout(timer); resolve(JSON.parse(stdout.trim())); } });
    child.once('error', reject);
    child.once('exit', code => { if (!stdout) { clearTimeout(timer); reject(new Error('Test launcher failed (' + code + '): ' + stderr)); } });
  });
  return { child, info };
}
async function open(info) {
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(info.url);
  await page.locator('.tiptap').waitFor();
  return page;
}
async function owner(info) {
  const response = await fetch(info.origin + '/api/launcher', { headers: { Authorization: 'Bearer ' + info.capability } });
  assert.equal(response.status, 200);
  return response.json();
}
async function exited(child) {
  const deadline = Date.now() + 10000;
  while (child.exitCode === null && Date.now() < deadline) await pause(100);
  assert.equal(child.exitCode, 0, 'owned server exits normally');
  assert.equal(JSON.parse(await fs.readFile(path.join(directory, '.studio/lock.json'), 'utf8')).released, true);
}
try {
  const first = await start();
  const page = await open(first.info);
  const originalLock = await fs.readFile(path.join(directory, '.studio/lock.json'));
  await page.reload(); await page.locator('.tiptap').waitFor();
  await pause(3500);
  assert.equal((await owner(first.info)).pages, 1);
  assert.deepEqual(await fs.readFile(path.join(directory, '.studio/lock.json')), originalLock);
  checks.push('refresh preserves the same live server and lock');

  await page.getByLabel('自动保存到磁盘', { exact: true }).uncheck();
  await page.locator('.tiptap').click(); await page.keyboard.press('Control+End');
  await page.keyboard.type('待保存测试');
  let prompted = false;
  page.once('dialog', async dialog => { assert.equal(dialog.type(), 'beforeunload'); prompted = true; await dialog.dismiss(); });
  await page.close({ runBeforeUnload: true });
  await pause(3500);
  assert.equal(prompted, true);
  assert.equal(page.isClosed(), false);
  assert.equal((await owner(first.info)).pages, 1);
  checks.push('cancelled unsaved-content prompt keeps the tab and server alive');
  await page.getByRole('button', { name: '保存 Ctrl+S', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.save-state')?.textContent?.startsWith('已保存到磁盘'));
  assert.ok((await fs.readFile(path.join(directory, 'chapters/test-chapter.json'), 'utf8')).includes('待保存测试'));

  const reopened = await start();
  assert.equal(reopened.info.reused, true);
  assert.equal(reopened.info.pid, first.child.pid);
  await exitedReuse(reopened.child);
  const secondPage = await open(reopened.info);
  assert.equal((await owner(first.info)).pages, 2);
  await page.close(); await pause(3500);
  assert.equal((await owner(first.info)).pages, 1);
  assert.equal(first.child.exitCode, null);
  checks.push('repeat launcher reuses the process; closing one of two tabs retains the session');
  await secondPage.close(); await exited(first.child);
  checks.push('closing the last real browser tab exits Node and releases the lock');

  const restarted = await start(), thirdPage = await open(restarted.info);
  assert.equal(restarted.info.reused, false);
  assert.ok(await thirdPage.locator('.tiptap').innerText().then(text => text.includes('待保存测试')));
  await thirdPage.getByRole('button', { name: '退出工具', exact: true }).click();
  await thirdPage.getByText('作者工具已退出，可以关闭此页面。', { exact: true }).waitFor();
  await exited(restarted.child);
  assert.equal(await thirdPage.locator('.error-banner').count(), 0);
  await thirdPage.close();
  checks.push('fresh launch retains saved content; explicit exit still closes cleanly');

  const crashed = await start(); await open(crashed.info);
  await browser.close(); await exited(crashed.child);
  checks.push('closing the entire browser also exits its owned server');
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, checks, errors }, null, 2));
  console.log(JSON.stringify({ passed: true, checks: checks.length, output }));
} finally {
  await browser.close();
  for (const child of children) if (child.exitCode === null) child.kill();
}
async function exitedReuse(child) {
  const deadline = Date.now() + 5000;
  while (child.exitCode === null && Date.now() < deadline) await pause(50);
  assert.equal(child.exitCode, 0, 'duplicate launcher exits after reopening the existing server');
}
