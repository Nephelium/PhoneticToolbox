import { readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url);
const parameters = JSON.parse(readFileSync(new URL('docs/modules/all-80-acoustic-parameters.json', root), 'utf8'));
const registry = JSON.parse(readFileSync(new URL('third_party/source-registry.json', root), 'utf8'));
for (const s of registry.sources) {
  if (!['academic', 'software'].includes(s.acknowledgement_group)) throw Error('Missing acknowledgement group: ' + s.id);
}
const authorDisplay = value => (value || '').split(/[;；]/).map(part => part.trim())
  .filter(part => part && !/^(unknown|作者待核实)$/i.test(part) && !/作者待(?:核实|确认)|PhoneticToolbox (?:project|contributors)/i.test(part)).join('；');
const sources = registry.sources.filter(s => !s.retired && s.show_in_acknowledgements !== false).map(s => ({ id:s.id, title:s.title, modules:s.modules || [], kind:s.kind,
  acknowledgement_group:s.acknowledgement_group,
  authors:authorDisplay(s.authors),
  version:!s.actual_included_version || /^(unknown|实际版本待核实)$/i.test(s.actual_included_version.trim()) ? '' : s.actual_included_version,
  license:s.show_license_in_acknowledgements === false ? '' : s.license || '许可待核实',
  urls: Object.fromEntries(Object.entries(s.urls || {}).filter(([,v]) => typeof v === 'string' && /^https:\/\//.test(v))),
  ...(s.citation ? { citation:s.citation } : {}),
  ...(s.homepage_attribution ? { attribution:s.homepage_attribution, permission_date:s.author_permission?.date, adaptation_note:s.m07_adaptation_note, adaptation_caution:s.m07_adaptation_caution } : {}) }));
for (const [file, data] of [['parameters.json', parameters], ['sources.json', sources]]) {
  const target = new URL('frontend/src/generated/' + file, root);
  const text = JSON.stringify(data, null, 2) + '\n';
  if (process.argv.includes('--check')) { if (readFileSync(target,'utf8') !== text) throw Error('UI registry drift: '+file); }
  else writeFileSync(target, text);
}
console.log(`UI data: ${parameters.length} parameters, ${sources.length} source records`);
