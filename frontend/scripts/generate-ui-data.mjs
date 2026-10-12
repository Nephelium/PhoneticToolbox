import { readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url);
const parameters = JSON.parse(readFileSync(new URL('docs/modules/all-80-acoustic-parameters.json', root), 'utf8'));
const registry = JSON.parse(readFileSync(new URL('third_party/source-registry.json', root), 'utf8'));
for (const s of registry.sources) {
  if (!['academic', 'software'].includes(s.acknowledgement_group)) throw Error('Missing acknowledgement group: ' + s.id);
}
const authorDisplay = value => (value || '').split(/[;；]/).map(part => part.trim())
  .filter(part => part && !/^(unknown|作者待核实)$/i.test(part) && !/作者待(?:核实|确认)|PhoneticToolbox (?:project|contributors)/i.test(part)).join('；');
// Keep unresolved provenance and environment audits in the source registry.
const versionDisplay = value => {
  const text = (value || '').trim();
  if (!text || /^(unknown|实际版本待核实)$/i.test(text) || /reviewed .+commit unknown/i.test(text)) return '';
  const runtimeVersion = text.match(/^(\d[\d.]+) in /);
  return runtimeVersion ? runtimeVersion[1] : text.split('；历史引入版本')[0];
};
const licenseDisplay = source => source.show_license_in_acknowledgements === false ||
  /待确认|待核实|尚未找到|尚未取得|未取得|unknown/i.test(source.license || '') ? '' : source.license || '';
const sources = registry.sources.filter(s => !s.retired && s.show_in_acknowledgements !== false).map(s => ({ id:s.id, title:s.title, modules:s.modules || [], kind:s.kind,
  acknowledgement_group:s.acknowledgement_group,
  authors:authorDisplay(s.authors),
  version:versionDisplay(s.acknowledgement_version ?? s.actual_included_version),
  license:licenseDisplay(s),
  urls: Object.fromEntries(Object.entries(s.urls || {}).filter(([,v]) => typeof v === 'string' && /^https:\/\//.test(v))),
  ...(s.citation ? { citation:s.citation } : {}),
  ...(s.reference_note ? { reference_note:s.reference_note } : {}),
  ...(s.homepage_attribution ? { attribution:s.homepage_attribution, permission_date:s.author_permission?.date, adaptation_note:s.m07_adaptation_note, adaptation_caution:s.m07_adaptation_caution } : {}) }));
for (const [file, data] of [['parameters.json', parameters], ['sources.json', sources]]) {
  const target = new URL('frontend/src/generated/' + file, root);
  const text = JSON.stringify(data, null, 2) + '\n';
  if (process.argv.includes('--check')) { if (readFileSync(target,'utf8') !== text) throw Error('UI registry drift: '+file); }
  else writeFileSync(target, text);
}
console.log(`UI data: ${parameters.length} parameters, ${sources.length} source records`);
