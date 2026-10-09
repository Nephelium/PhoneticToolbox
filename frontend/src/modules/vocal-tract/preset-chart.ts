import type {Catalog} from '../ipa-plus/types.ts';

export function buildPresetChart(catalog:Catalog){
  const matrix=catalog.charts.ipa.find(section=>section.id==='extended')!;
  const vowels=catalog.charts.ipa.find(section=>section.id==='vowels')!;
  const base=catalog.entries.filter(entry=>entry.system==='ipa'&&!entry.isExample&&entry.insertionMode==='literal'&&entry.display!=='ʼ'&&
    ['pulmonic','vowels','nonpulmonic','other'].includes(entry.section));
  // Keep all literal combinations from the displayed M17 extended matrix.
  const ids=new Set([...base.map(entry=>entry.id),...matrix.rows!.flatMap(row=>row.cells.flatMap(cell=>cell.ids))]);
  return {
    version:catalog.version,
    entries:catalog.entries.filter(entry=>ids.has(entry.id)).map(({id,display,insertText,nameZh})=>({id,display,insertText,nameZh})),
    consonants:{columns:matrix.columns!,rows:matrix.rows!.map(row=>({label:row.label,cells:row.cells.map(cell=>cell.ids)}))},
    vowels:vowels.points!,
    extras:base.filter(entry=>['nonpulmonic','other'].includes(entry.section)).map(entry=>entry.id),
  };
}
