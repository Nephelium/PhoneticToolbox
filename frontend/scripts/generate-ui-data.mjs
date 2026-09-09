import { readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url);
const parameters = JSON.parse(readFileSync(new URL('docs/modules/all-80-acoustic-parameters.json', root), 'utf8'));
const registry = JSON.parse(readFileSync(new URL('third_party/source-registry.json', root), 'utf8'));
for (const s of registry.sources) {
  if (!['academic', 'software'].includes(s.acknowledgement_group)) throw Error('Missing acknowledgement group: ' + s.id);
}
const sources = registry.sources.map(s => ({ id:s.id, title:s.title, modules:s.modules || [], kind:s.kind,
  acknowledgement_group:s.acknowledgement_group,
  authors:s.authors || '作者待核实', version:s.actual_included_version || '实际版本待核实', license:s.license || '许可待核实',
  urls: Object.fromEntries(Object.entries(s.urls || {}).filter(([,v]) => typeof v === 'string' && /^https:\/\//.test(v))) }));
for (const [file, data] of [['parameters.json', parameters], ['sources.json', sources]]) {
  const target = new URL('frontend/src/generated/' + file, root);
  const text = JSON.stringify(data, null, 2) + '\n';
  if (process.argv.includes('--check')) { if (readFileSync(target,'utf8') !== text) throw Error('UI registry drift: '+file); }
  else writeFileSync(target, text);
}
console.log(`UI data: ${parameters.length} parameters, ${sources.length} source records`);
