import test from 'node:test';
import assert from 'node:assert/strict';
import {citation} from '../src/modules/paper-reading/citations.ts';
import type {Paper} from '../src/platform/papers.ts';

const paper={authors:'Chaofei Fan、Jaimie M. Henderson、Chris Manning、Francis R. Willett',title:'Towards a Quantitative Analysis of Coarticulation with a Phoneme-to-Articulatory Model',submittedAt:'2024-08-10',version:'arXiv:2408.05641v1',sourceUrl:'https://arxiv.org/abs/2408.05641v1'} as Paper;
test('P2A GB/T keeps author order and abbreviates authors individually',()=>{
 assert.equal(citation(paper,'GB/T 7714—2015','2026-10-12'),'FAN C, HENDERSON J M, MANNING C, et al. Towards a Quantitative Analysis of Coarticulation with a Phoneme-to-Articulatory Model[EB/OL]. (2024-08-10)[2026-10-12]. https://arxiv.org/abs/2408.05641v1.');
});
test('P2A APA and BibTeX preserve all four distinct original authors',()=>{
 assert(citation(paper,'APA 7','2026-10-12').startsWith('Fan, C., Henderson, J. M., Manning, C., & Willett, F. R. (2024).'));
 assert(citation(paper,'BibTeX','2026-10-12').includes('author = {Fan, Chaofei and Henderson, Jaimie M. and Manning, Chris and Willett, Francis R.}'));
});
test('Hyphenated given names and existing middle-dot metadata remain usable',()=>{
 const value={...paper,authors:'Chin-Yun Yu · György Fazekas'};
 assert(citation(value,'GB/T 7714—2015','2026-10-12').startsWith('YU C Y, FAZEKAS G.'));
});
