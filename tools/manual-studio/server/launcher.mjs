import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { startServer } from './http-server.mjs';

const execute = promisify(execFile);
const samePath = (a, b) => typeof a === 'string' && typeof b === 'string' && (process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b);
async function writeState(file, state) {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = file + '.' + randomUUID() + '.tmp';
  await fs.writeFile(temporary, JSON.stringify(state), { mode: 0o600 });
  await fs.rename(temporary, file);
}
export async function sessionFile(studioRoot, projectRoot) {
  const canonical = await fs.realpath(projectRoot).catch(() => path.resolve(projectRoot));
  const key = createHash('sha256').update(process.platform === 'win32' ? canonical.toLowerCase() : canonical).digest('hex');
  return path.join(studioRoot, '.runtime', 'sessions', key + '.json');
}
async function existingSession(file, studioRoot, projectRoot) {
  try {
    const state = JSON.parse(await fs.readFile(file, 'utf8'));
    if (!state.active || !Number.isSafeInteger(state.pid) || !/^[a-f0-9]{64}$/.test(state.capability)) return;
    if (!samePath(state.studioRoot, studioRoot) || !samePath(state.projectRoot, projectRoot)) return;
    const url = new URL(state.origin);
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || state.origin !== url.origin) return;
    process.kill(state.pid, 0);
    const lock = JSON.parse(await fs.readFile(path.join(projectRoot, '.studio', 'lock.json'), 'utf8'));
    if (lock.released || lock.pid !== state.pid || lock.id !== state.lockId) return;
    const response = await fetch(state.origin + '/api/launcher', { headers: { Authorization: 'Bearer ' + state.capability }, signal: AbortSignal.timeout(1500) });
    if (!response.ok) return;
    const owner = await response.json();
    if (owner.pid !== state.pid || owner.lockId !== state.lockId || !samePath(owner.studioRoot, studioRoot) || !samePath(owner.projectRoot, projectRoot)) return;
    return state;
  } catch { /* A stale or unrelated entry never authorizes adopting a service. */ }
}
export async function openBrowser(url) {
  if (!/^http:\/\/127\.0\.0\.1:\d+\/#([a-f0-9]{64})$/.test(url)) throw new Error('编辑器浏览器地址无效');
  const command = "Start-Process -FilePath '" + url + "'";
  await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(command, 'utf16le').toString('base64')], { windowsHide: true, timeout: 15000 });
}
export async function launchStudio({ studioRoot, projectRoot, port = 0, dev = false, openPage = openBrowser }) {
  studioRoot = await fs.realpath(studioRoot);
  projectRoot = await fs.realpath(projectRoot).catch(() => path.resolve(projectRoot));
  const file = await sessionFile(studioRoot, projectRoot);
  async function reuse(state) {
    const response = await fetch(state.origin + '/api/launch', { method: 'POST', headers: { Authorization: 'Bearer ' + state.capability, Origin: state.origin, 'Content-Type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(1500) });
    if (!response.ok) throw new Error('已有编辑器会话无法重新打开，请在原页面退出工具后重试。');
    const url = state.origin + '/#' + state.capability;
    await openPage(url);
    return { ...state, url, reused: true };
  }
  const existing = await existingSession(file, studioRoot, projectRoot);
  if (existing) return reuse(existing);
  let openProject;
  try { await fs.access(path.join(projectRoot, 'project.json')); openProject = projectRoot; } catch {}
  let instance;
  try { instance = await startServer({ studioRoot, projectRoot: openProject, port, dev }); }
  catch (error) {
    if (error.status === 423) {
      // A simultaneous launcher may still be writing its private runtime entry.
      for (let attempt = 0; attempt < 10; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        const current = await existingSession(file, studioRoot, projectRoot);
        if (current) return reuse(current);
      }
      throw new Error('工程仍由先前的编辑器占用。请回到原标签页，保存并点击“退出工具”后再打开。若原标签页已关闭，请检查遗留的编辑器服务。');
    }
    throw error;
  }
  const state = { active: true, studioRoot, projectRoot, pid: process.pid, lockId: instance.store.lockId, origin: instance.origin, capability: instance.capability };
  try {
    // Private runtime metadata is ignored by Git and excluded from project exports.
    await writeState(file, state);
    instance.reservePage();
    await openPage(instance.url);
  } catch (error) { await instance.close(); throw error; }
  // Expired entries are harmless: every reuse checks the live PID, project lock
  // and authenticated server identity. Shutdown never overwrites a newer entry.
  return { ...instance, pid: process.pid, reused: false };
}
export async function reportReady(file, state) { if (file) await writeState(path.resolve(file), state); }
