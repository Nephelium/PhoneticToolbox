import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { startServer } from '../server/http-server.mjs';
import { sha } from '../server/project-store.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(root, '../../manual');
const output = path.join(root, 'test-output', 'compatibility-' + new Date().toISOString().replace(/[:.]/g, '-'));
const directory = path.join(output, '说明书验证副本');
await fs.mkdir(path.join(directory, 'chapters'), { recursive: true });
const projectBytes = await fs.readFile(path.join(source, 'project.json'));
const project = JSON.parse(projectBytes);
await fs.writeFile(path.join(directory, 'project.json'), projectBytes);
const original = new Map();
for (const descriptor of project.chapters) {
  const bytes = await fs.readFile(path.join(source, descriptor.path));
  original.set(descriptor.id, JSON.parse(bytes));
  await fs.writeFile(path.join(directory, descriptor.path), bytes);
}
// Only the independent replica is opened for writing. Referenced assets are
// copied for the real preview; the user's recovery/history/lock are not copied.
for (const asset of project.assets) {
  const target = path.join(directory, asset.path);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(path.join(source, asset.path), target);
}
const runtime = process.env.PTB_PLAYWRIGHT ?? path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const { chromium } = await import(pathToFileURL(runtime));
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errors = [], checks = [], affected = ['getting-started', 'm04', 'm08', 'm11'];
const markers = new Map(), changedLabels = new Map();
let instance;
page.on('pageerror', error => errors.push(error.message));
const text = node => (node.type === 'text' ? node.text ?? '' : (node.content ?? []).map(text).join(''));
const references = node => [
  ...(['crossReference', 'citation'].includes(node.type) ? [{ type: node.type, chapterId: node.attrs?.chapterId, targetId: node.attrs?.targetId, referenceId: node.attrs?.referenceId, label: node.attrs?.label }] : []),
  ...(node.content ?? []).flatMap(references),
];
const anchoredNodes = node => [...(node.attrs?.id ? [node] : []), ...(node.content ?? []).flatMap(anchoredNodes)];
const nodeTypes = node => [node.type, ...(node.content ?? []).flatMap(nodeTypes)];
async function select(id) {
  const index = project.chapters.findIndex(chapter => chapter.id === id);
  await page.locator('.chapter-list button').nth(index).click();
  await page.locator('.tiptap').waitFor();
  assert.equal(await page.locator('.unknown-content').count(), 0);
  assert.equal(await page.locator('.chapter-heading > input').inputValue(), original.get(id).title);
}
async function save() {
  await page.getByRole('button', { name: '保存 Ctrl+S', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.save-state')?.textContent?.startsWith('已保存到磁盘'));
  assert.equal(await page.locator('.error-banner').count(), 0);
}
async function verifyLayout(name) {
  const bounds = await page.evaluate(() => {
    const scrolling = document.scrollingElement;
    const workspace = document.querySelector('.workspace').getBoundingClientRect();
    const editor = document.querySelector('.editing-area').getBoundingClientRect();
    return { width: innerWidth, height: innerHeight, scrollWidth: scrolling.scrollWidth, scrollHeight: scrolling.scrollHeight, workspaceBottom: workspace.bottom, editorHeight: editor.height, editorBottom: editor.bottom };
  });
  assert.ok(bounds.scrollHeight <= bounds.height + 1, name + ': no outer vertical overflow');
  assert.ok(bounds.scrollWidth <= bounds.width + 1, name + ': no outer horizontal overflow');
  assert.ok(Math.abs(bounds.workspaceBottom - bounds.height) <= 1, name + ': workspace fills the available height');
  assert.ok(bounds.editorHeight > 20 && bounds.editorBottom <= bounds.height + 1, name + ': editor remains inside the viewport');
  await page.evaluate(() => window.scrollTo(100000, 100000));
  assert.deepEqual(await page.evaluate(() => [scrollX, scrollY]), [0, 0]);
  checks.push({ name: 'bounded layout', mode: name, ...bounds });
}
try {
  instance = await startServer({ studioRoot: root, projectRoot: directory });
  await page.goto(instance.url);
  await page.locator('.tiptap').waitFor();
  await page.getByLabel('自动保存到磁盘', { exact: true }).uncheck();
  assert.equal(await page.locator('.inspector-pane').count(), 0);
  for (const descriptor of project.chapters) {
    await select(descriptor.id);
    checks.push({ name: 'chapter opens for visual editing', id: descriptor.id });
  }
  await select('getting-started');
  for (const size of [{ width: 2560, height: 1440 }, { width: 1920, height: 1080 }, { width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 1024, height: 768 }, { width: 900, height: 600 }, { width: 640, height: 700 }, { width: 1920, height: 540 }]) {
    await page.setViewportSize(size);
    const prefix = `${size.width}x${size.height}`;
    assert.equal(await page.locator('.inspector-pane').count(), 0);
    await verifyLayout(prefix + ': default tools hidden, preview visible');
    const editBounds = await page.locator('.editing-area').boundingBox();
    await page.mouse.move(editBounds.x + editBounds.width / 2, editBounds.y + editBounds.height / 2);
    await page.mouse.wheel(0, 100000);
    await page.waitForFunction(() => document.querySelector('.editing-area').scrollTop > 0);
    assert.equal(await page.evaluate(() => scrollY), 0);
    await page.getByRole('button', { name: '排版、素材与文献', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: '排版、素材与文献', exact: true }).getAttribute('aria-expanded'), 'true');
    await verifyLayout(prefix + ': tools and preview visible');
    await page.getByRole('button', { name: '收起预览', exact: true }).click();
    await verifyLayout(prefix + ': tools visible, preview hidden');
    await page.getByRole('button', { name: '收起编辑工具', exact: true }).click();
    await verifyLayout(prefix + ': tools and preview hidden');
    await page.getByRole('button', { name: '正式预览', exact: true }).click();
  }
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.getByRole('button', { name: '排版、素材与文献', exact: true }).focus();
  await page.keyboard.press('Enter');
  for (const name of ['素材', '文献', '排版']) {
    await page.locator('.inspector-tabs').getByRole('button', { name, exact: true }).click();
    assert.equal(await page.locator('.inspector-pane .properties').count(), 1);
  }
  await page.getByRole('button', { name: '收起编辑工具', exact: true }).click();
  checks.push({ name: 'tools open with keyboard, all three tabs remain accessible and panel closes' });
  for (const id of affected) {
    await select(id);
    const marker = `[可视化保存验证-${id}]`;
    markers.set(id, marker);
    await page.locator('.tiptap > p').first().click();
    await page.keyboard.press('Home');
    await page.keyboard.type(marker);
    assert.ok((await page.locator('.tiptap').innerText()).includes(marker));
    if (id === 'getting-started' || id === 'm08') {
      const type = id === 'getting-started' ? 'crossReference' : 'citation';
      const attribute = type === 'crossReference' ? 'data-reference' : 'data-citation';
      await page.locator(`.tiptap > div[${attribute}]`).first().click();
      if (!await page.locator('.inspector-pane').count()) await page.getByRole('button', { name: '排版、素材与文献', exact: true }).click();
      await page.locator('.inspector-tabs').getByRole('button', { name: '排版', exact: true }).click();
      const label = type === 'crossReference' ? '显示文字' : '引用文字';
      const field = page.getByLabel(label, { exact: true });
      await field.waitFor();
      const oldLabel = await field.inputValue(), nextLabel = oldLabel + '（属性验证）';
      await field.fill(nextLabel);
      changedLabels.set(id, { type, oldLabel, nextLabel });
      checks.push({ name: 'standalone reference selected and label edited through inspector', id, type });
    }
    await save();
    const descriptor = project.chapters.find(chapter => chapter.id === id);
    const saved = JSON.parse(await fs.readFile(path.join(directory, descriptor.path), 'utf8'));
    assert.equal(text(saved.body).replace(marker, ''), text(original.get(id).body));
    const expectedReferences = references(original.get(id).body);
    const change = changedLabels.get(id);
    if (change) expectedReferences.find(reference => reference.type === change.type && reference.label === change.oldLabel).label = change.nextLabel;
    assert.deepEqual(references(saved.body), expectedReferences);
    const savedAnchors = new Map(anchoredNodes(saved.body).map(node => [node.attrs.id, node]));
    for (const node of anchoredNodes(original.get(id).body)) {
      const retained = savedAnchors.get(node.attrs.id);
      assert.ok(retained, 'original stable anchor retained: ' + node.attrs.id);
      assert.equal(retained.type, node.type);
    }
    assert.ok(!nodeTypes(saved.body).includes('blockCrossReference'));
    assert.ok(!nodeTypes(saved.body).includes('blockCitation'));
    await page.reload();
    await page.locator('.tiptap').waitFor();
    await page.getByLabel('自动保存到磁盘', { exact: true }).uncheck();
    assert.equal(await page.locator('.inspector-pane').count(), 0);
    await select(id);
    assert.ok((await page.locator('.tiptap').innerText()).includes(marker));
    if (change) assert.ok((await page.locator('.reading-preview').innerText()).includes(change.nextLabel));
    checks.push({ name: 'native typing, disk save, unchanged references/anchors and reload', id });
  }
  await select('getting-started');
  await page.screenshot({ path: path.join(output, 'visual-editing-light.png') });
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.evaluate(() => Promise.all(document.getAnimations().map(animation => animation.finished.catch(() => {}))));
  await page.screenshot({ path: path.join(output, 'visual-editing-dark.png') });
  await instance.close();
  instance = undefined;
  instance = await startServer({ studioRoot: root, projectRoot: directory });
  await page.goto(instance.url);
  await page.locator('.tiptap').waitFor();
  await page.getByLabel('自动保存到磁盘', { exact: true }).uncheck();
  for (const id of affected) {
    await select(id);
    assert.ok((await page.locator('.tiptap').innerText()).includes(markers.get(id)));
  }
  checks.push({ name: 'new service session reopens all four saved chapters' });
  const exported = await fetch(instance.origin + '/api/export', { headers: { Authorization: 'Bearer ' + instance.capability } });
  assert.equal(exported.status, 200);
  const { gunzipSync } = await import('node:zlib');
  const bundle = JSON.parse(gunzipSync(Buffer.from(await exported.arrayBuffer())));
  for (const file of bundle.files) assert.equal(sha(Buffer.from(file.data, 'base64')), file.sha256);
  checks.push({ name: 'complete project export preserves file hashes', files: bundle.files.length });
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ checks, errors, scope: 'Windows Chrome; independent replica; original manual read only' }, null, 2));
  console.log(JSON.stringify({ passed: checks.length, report: path.join(output, 'report.json'), screenshots: [path.join(output, 'visual-editing-light.png'), path.join(output, 'visual-editing-dark.png')] }));
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png') });
  await fs.writeFile(path.join(output, 'failure.json'), JSON.stringify({ error: error.stack, errors, checks }, null, 2));
  throw error;
} finally {
  if (instance) await instance.close();
  await browser.close();
}
