import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchStudio, reportReady } from './launcher.mjs';

const studioRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), value = key => { const index = args.indexOf(key); return index < 0 ? undefined : args[index + 1]; };
const readyFile = value('--ready-file');
try {
  const instance = await launchStudio({ studioRoot, projectRoot: value('--project') ?? path.resolve(studioRoot, '../../manual'), port: Number(value('--port') ?? 0), dev: args.includes('--dev'), ...(args.includes('--test-info') ? { openPage: async () => {} } : {}) });
  await reportReady(readyFile, { ok: true, reused: instance.reused, origin: instance.origin, pid: instance.pid });
  if (args.includes('--test-info')) console.log(JSON.stringify({ url: instance.url, origin: instance.origin, capability: instance.capability, reused: instance.reused, pid: instance.pid }));
  else console.log(instance.reused ? '已重新打开现有说明书编辑器。' : '说明书编辑器已启动。关闭最后一个编辑器标签页会自动退出服务。');
  if (!instance.reused) {
    let stopping = false;
    const stop = async () => { if (stopping) return; stopping = true; await instance.close(); };
    process.once('SIGINT', () => void stop());
    process.once('SIGTERM', () => void stop());
  }
} catch (error) {
  const message = String(error.message || '编辑器启动失败');
  await reportReady(readyFile, { ok: false, error: message });
  console.error(message);
  process.exitCode = 1;
}
