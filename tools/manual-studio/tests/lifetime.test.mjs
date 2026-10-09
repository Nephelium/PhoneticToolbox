import test, { after } from 'node:test';
import { createArtifacts } from './artifacts.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { ProjectStore } from '../server/project-store.mjs';
import { startServer } from '../server/http-server.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = await createArtifacts(root, 'lifetime');
after(() => artifacts.finish());
const sleep = delay => new Promise(resolve => setTimeout(resolve, delay));
async function fixture(t) {
  const directory = path.join(artifacts.scratch, '标签页会话-' + randomUUID());
  const store = new ProjectStore(); await store.create(directory, '页面生命周期'); await store.release();
  const instance = await startServer({ studioRoot: root, projectRoot: directory, pageCloseGraceMs: 200 });
  t.after(() => instance.close());
  return { ...instance, directory, headers: { Authorization: 'Bearer ' + instance.capability } };
}
async function connect(instance) {
  const id = randomUUID(), controller = new AbortController();
  const response = await fetch(instance.origin + '/api/page-live?id=' + id, { headers: instance.headers, signal: controller.signal });
  assert.equal(response.status, 200);
  return { id, controller, response };
}
async function beacon(instance, id, capability = instance.capability, origin = instance.origin) {
  return fetch(instance.origin + '/api/page-close', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify({ id, capability }) });
}
test('last tab beacon closes the service and releases its project lock', async t => {
  const instance = await fixture(t), page = await connect(instance);
  assert.equal((await beacon(instance, page.id)).status, 200);
  await sleep(400);
  assert.equal(instance.server.listening, false);
  assert.equal(JSON.parse(await fs.readFile(path.join(instance.directory, '.studio/lock.json'), 'utf8')).released, true);
});
test('stream disconnect also closes the service when no beacon is delivered', async t => {
  const instance = await fixture(t), page = await connect(instance);
  page.controller.abort();
  await sleep(500);
  assert.equal(instance.server.listening, false);
});
test('refresh reconnects during grace and closing one of several tabs keeps the writer alive', async t => {
  const instance = await fixture(t), first = await connect(instance);
  await beacon(instance, first.id);
  const refreshed = await connect(instance), other = await connect(instance);
  await sleep(400);
  assert.equal(instance.server.listening, true);
  await beacon(instance, refreshed.id);
  await sleep(400);
  assert.equal(instance.server.listening, true);
  await beacon(instance, other.id);
  await sleep(400);
  assert.equal(instance.server.listening, false);
});
test('foreign-origin and invalid-capability close notifications cannot end an editor session', async t => {
  const instance = await fixture(t), page = await connect(instance);
  assert.equal((await beacon(instance, page.id, instance.capability, 'https://example.org')).status, 403);
  assert.equal((await beacon(instance, page.id, 'invalid')).status, 401);
  assert.equal((await fetch(instance.origin + '/api/page-live?id=' + randomUUID())).status, 401);
  await sleep(400);
  assert.equal(instance.server.listening, true);
});
test('auto-close waits for an already queued save before releasing the project', async t => {
  const instance = await fixture(t), page = await connect(instance), snapshot = await instance.store.load();
  instance.store.serial(() => sleep(450));
  snapshot.project.title = '关闭前已提交的保存';
  const saved = instance.store.save({ project: snapshot.project, chapters: snapshot.chapters, baseRevision: snapshot.revision });
  await beacon(instance, page.id);
  await saved;
  await instance.close();
  assert.equal(JSON.parse(await fs.readFile(path.join(instance.directory, 'project.json'), 'utf8')).title, '关闭前已提交的保存');
  assert.equal(JSON.parse(await fs.readFile(path.join(instance.directory, '.studio/lock.json'), 'utf8')).released, true);
});
