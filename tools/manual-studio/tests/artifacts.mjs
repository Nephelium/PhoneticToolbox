import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';

// Only a new independent test run is owned. Formal manual projects never enter here.
export async function createArtifacts(studioRoot, prefix='browser') {
  if (!/^[a-z0-9-]+$/.test(prefix)) throw new Error('Invalid test run name');
  const out = path.join(studioRoot, 'test-output', `${prefix}-${randomUUID()}`);
  for (let p = path.dirname(out); ; p = path.dirname(p)) {
    const stat = await fs.lstat(p).catch(e => { if (e.code !== 'ENOENT') throw e; });
    if (stat?.isSymbolicLink()) throw new Error(`Linked test path: ${p}`);
    if (p === path.dirname(p)) break;
  }
  await fs.mkdir(out, { recursive: true });
  const scratch = path.join(out, 'scratch');
  await fs.mkdir(scratch);
  const write = (name, data) => fs.writeFile(path.join(out, name), JSON.stringify(data, null, 2) + '\n');
  const marker = { schema: 'ptb-test-run/1', owner_pid: process.pid, state: 'active',
    recipe: `Manual Studio ${prefix} tests: generated chapters, bytes and synthetic media; no original manual assets`,
    started_at: new Date().toISOString() };
  await write('run.json', marker);
  const finish = async (outcome='see-test-report') => {
    try {
      const files = [];
      async function visit(directory) {
        for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
          const file = path.join(directory, entry.name), stat = await fs.lstat(file);
          if (stat.isSymbolicLink()) throw new Error(`Scratch contains a link: ${file}`);
          if (stat.isDirectory()) await visit(file);
          else files.push({ path: path.relative(scratch, file), bytes: stat.size,
            sha256: createHash('sha256').update(await fs.readFile(file)).digest('hex') });
        }
      }
      await visit(scratch);
      await write('scratch-manifest.json', { schema: 'ptb-test-scratch-files/1', files });
      await write('run.json', { ...marker, state: 'ready', test_outcome: outcome });
      const script = path.resolve(studioRoot, '../../scripts/cleanup_test_scratch.ps1');
      if (process.platform !== 'win32') throw new Error('Automatic scratch cleanup currently supports Windows');
      const result = spawnSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script,
        '-RunDirectory', out], { windowsHide: true, encoding: 'utf8' });
      if (result.status !== 0) throw new Error(result.stderr || result.error?.message || 'Scratch retained; inspect cleanup.json');
    } catch (error) {
      await write('cleanup.json', { status: 'retained', path: scratch, reason: String(error) });
      console.error(`Test scratch retained: ${path.join(out, 'cleanup.json')}`);
    }
  };
  return { out, scratch, finish };
}

export async function withArtifacts(studioRoot, callback) {
  const run=await createArtifacts(studioRoot);
  let outcome='completed';
  try { return await callback(run); }
  catch(error) { outcome='failed'; throw error; }
  finally { await run.finish(outcome); }
}
