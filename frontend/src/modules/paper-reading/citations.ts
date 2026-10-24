import type {Paper} from '../../platform/papers.ts';
export const citationStyles=['GB/T 7714—2015','ASA（美国社会学会）','MLA 9','APA 7','BibTeX'] as const;
export type CitationStyle=typeof citationStyles[number];
function name(value:string){const parts=value.trim().split(/\s+/);return {family:parts.pop()??'',given:parts.join(' ')};}
function initials(value:string){return value.split(/[\s-]+/).filter(Boolean).map(x=>x[0]+'.').join(' ');}
/** Keep source author order and use journal publication dates when provided. */
export function citation(paper:Paper,style:CitationStyle,accessed:string):string{
 const authors=paper.authors.split(/\s*[·;,、]\s*|\s+and\s+/).filter(s=>s.trim()).map(name),year=paper.submittedAt.slice(0,4);
 const inverted=authors.map(a=>`${a.family}, ${a.given}`),normal=authors.map(a=>`${a.given} ${a.family}`.trim());
 const version=paper.version.replace(/^arXiv:/,'');
 const j=paper.journal;
 if(j){
  const y=j.publishedAt.slice(0,4),location=j.pages??j.articleNumber??'',volume=j.volume??'',issue=j.issue?`(${j.issue})`:'';
  const journalLocation=[volume+issue,location].filter(Boolean).join(': ');
  if(style==='GB/T 7714—2015')return `${authors.slice(0,3).map(a=>`${a.family.toUpperCase()} ${initials(a.given).replaceAll('.','')}`).join(', ')}${authors.length>3?', et al':''}. ${paper.title}[J/OL]. ${j.title}, ${y}${journalLocation?', '+journalLocation:''}[${accessed}]. ${paper.sourceUrl}.`;
  if(style==='ASA（美国社会学会）')return `${inverted[0]}${authors.length>1?', '+normal.slice(1,-1).map(x=>x+', ').join('')+'and '+normal.at(-1):''}. ${y}. “${paper.title}.” ${j.title}${journalLocation?' '+journalLocation:''}. ${paper.sourceUrl}.`;
  if(style==='MLA 9')return `${authors.length>2?inverted[0]+', et al':authors.length===2?inverted[0]+', and '+normal[1]:inverted[0]}. “${paper.title}.” ${j.title}${volume?', vol. '+volume:''}${j.issue?', no. '+j.issue:''}, ${y}${location?', '+(j.pages?'pp. ':'article ')+location:''}, ${paper.sourceUrl}.`;
  if(style==='APA 7'){const values=authors.map(a=>`${a.family}, ${initials(a.given)}`);return `${values.length>1?values.slice(0,-1).join(', ')+', & '+values.at(-1):values[0]} (${y}). ${paper.title}. ${j.title}${volume?', '+volume+issue:''}${location?', '+(j.pages?'':'Article ')+location:''}. ${paper.sourceUrl}`;}
  const safe=(x:string)=>x.replace(/[{}]/g,'').replaceAll('\\','');
  const fields=[['title',paper.title],['author',inverted.join(' and ')],['journal',j.title],['year',y],...(volume?[['volume',volume]]:[]),...(j.issue?[['number',j.issue]]:[]),...(j.pages?[['pages',j.pages]]:[]),...(j.articleNumber?[['eid',j.articleNumber]]:[]),['doi',j.doi],['url',paper.sourceUrl],['urldate',accessed]];
  return `@article{${safe(authors[0].family.toLowerCase())}${y}_${safe(j.doi).replace(/[^a-zA-Z0-9]/g,'')},\n${fields.map(([key,value])=>`  ${key} = {${safe(value)}}`).join(',\n')}\n}`;
 }
 if(style==='GB/T 7714—2015')return `${authors.slice(0,3).map(a=>`${a.family.toUpperCase()} ${initials(a.given).replaceAll('.','')}`).join(', ')}${authors.length>3?', et al':''}. ${paper.title}[EB/OL]. (${paper.submittedAt})[${accessed}]. ${paper.sourceUrl}.`;
 if(style==='ASA（美国社会学会）')return `${inverted[0]}${authors.length>1?', '+normal.slice(1,-1).map(x=>x+', ').join('')+'and '+normal.at(-1):''}. ${year}. “${paper.title}.” arXiv preprint ${version}. Retrieved ${new Date(accessed+'T12:00:00').toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'})} (${paper.sourceUrl}).`;
 if(style==='MLA 9')return `${authors.length>2?inverted[0]+', et al':authors.length===2?inverted[0]+', and '+normal[1]:inverted[0]}. “${paper.title}.” arXiv, ${new Date(paper.submittedAt+'T12:00:00').toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'})}, ${paper.sourceUrl}. Preprint.`;
 if(style==='APA 7'){const values=authors.map(a=>`${a.family}, ${initials(a.given)}`);return `${values.length>1?values.slice(0,-1).join(', ')+', & '+values.at(-1):values[0]} (${year}). ${paper.title} [Preprint]. arXiv. ${paper.sourceUrl}`;}
 const safe=(x:string)=>x.replace(/[{}]/g,'').replaceAll('\\','');
 return `@misc{${safe(authors[0].family.toLowerCase())}${year}_${version.replace(/[^a-zA-Z0-9]/g,'')},\n  title = {${safe(paper.title)}},\n  author = {${inverted.map(safe).join(' and ')}},\n  year = {${year}},\n  eprint = {${version}},\n  archivePrefix = {arXiv},\n  url = {${paper.sourceUrl}},\n  urldate = {${accessed}}\n}`;
}
