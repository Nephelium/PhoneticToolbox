import test from 'node:test';
import assert from 'node:assert/strict';
import {citation,citationStyles} from '../src/modules/paper-reading/citations.ts';
import type {Paper} from '../src/platform/papers.ts';
const paper={title:'A Speech Study',authors:'Jane Mary Smith · John Doe · Xin Wang',submittedAt:'2026-09-30',publishedAt:'2026-10-07',version:'arXiv:2610.12345v2',sourceUrl:'https://arxiv.org/abs/2610.12345v2'} as Paper;
test('M18 citations cite source date and fixed preprint version, never column date or invented journal',()=>{
 for(const style of citationStyles){const text=citation(paper,style,'2026-10-08');assert.ok(text.includes(paper.sourceUrl));assert.ok(text.includes('2026'));assert.ok(!text.includes('2026-10-07'));}
 assert.equal(citation(paper,'GB/T 7714—2015','2026-10-08'),'SMITH J M, DOE J, WANG X. A Speech Study[EB/OL]. (2026-09-30)[2026-10-08]. https://arxiv.org/abs/2610.12345v2.');
 assert.ok(citation(paper,'ASA（美国社会学会）','2026-10-08').startsWith('Smith, Jane Mary, John Doe, and Xin Wang. 2026.'));
 assert.ok(citation(paper,'MLA 9','2026-10-08').startsWith('Smith, Jane Mary, et al.'));
 assert.ok(citation(paper,'BibTeX','2026-10-08').includes('Smith, Jane Mary and Doe, John and Wang, Xin'));
});

test('journal citations use publication date, real venue and DOI in every style',()=>{
 const journal={...paper,submittedAt:'2025-09-12',publishedAt:'2026-10-21',version:'Version of Record, 2026-09-09',sourceUrl:'https://doi.org/10.3758/s13428-026-03174-y',journal:{title:'Behavior Research Methods',publishedAt:'2026-09-09',doi:'10.3758/s13428-026-03174-y',volume:'58',articleNumber:'289'}};
 for(const style of citationStyles){const text=citation(journal,style,'2026-10-21');assert.ok(text.includes('2026'));assert.ok(text.includes('Behavior Research Methods'));assert.ok(text.includes(journal.sourceUrl));assert.ok(!text.includes('2025'));assert.ok(!text.includes('arXiv'));assert.ok(!text.includes('Preprint'));}
 assert.equal(citation(journal,'APA 7','2026-10-21'),'Smith, J. M., Doe, J., & Wang, X. (2026). A Speech Study. Behavior Research Methods, 58, Article 289. https://doi.org/10.3758/s13428-026-03174-y');
 const bib=citation(journal,'BibTeX','2026-10-21');assert.ok(bib.startsWith('@article{'));assert.ok(bib.includes('eid = {289}'));assert.ok(bib.includes('doi = {10.3758/s13428-026-03174-y}'));assert.ok(!bib.includes('archivePrefix'));
});

test('journal citations support page ranges and avoid inventing missing issue/volume',()=>{
 const journal={...paper,journal:{title:'Journal of Phonetics',publishedAt:'2024-01-02',doi:'10.1016/example',pages:'1–12'},sourceUrl:'https://doi.org/10.1016/example'};
 assert.ok(citation(journal,'MLA 9','2026-10-21').includes('2024, pp. 1–12'));
 const bib=citation(journal,'BibTeX','2026-10-21');assert.ok(bib.includes('pages = {1–12}'));assert.ok(!bib.includes('volume ='));assert.ok(!bib.includes('number ='));
});
