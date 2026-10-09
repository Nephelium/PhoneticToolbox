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
