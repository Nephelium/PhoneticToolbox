import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeBackground} from '../src/design/background.ts';
import {customPalette,palettes,paletteTokens,contrast} from '../src/design/themes.ts';
test('background preferences keep zero, clamp boundaries and reject invalid numbers',()=>{
 assert.deepEqual(normalizeBackground(null),{blur:8,opacity:25});
 assert.deepEqual(normalizeBackground({blur:0,opacity:0}),{blur:0,opacity:0});
 assert.deepEqual(normalizeBackground({blur:-4,opacity:500}),{blur:0,opacity:100});
 assert.deepEqual(normalizeBackground({blur:99,opacity:-5}),{blur:40,opacity:0});
 assert.deepEqual(normalizeBackground({blur:NaN,opacity:'80'}),{blur:8,opacity:25});
});
test('custom colour families retain readable light/dark tokens across extreme seed colours',()=>{
 const index=palettes.findIndex(p=>p.id==='custom'),original=palettes[index];
 try{for(const seed of ['#000000','#ffffff','#ff0000','#00ff00','#0000ff','#808080','#123456','#fff000','#ff00ff']){
  palettes[index]=customPalette(seed);
  for(const mode of ['light','dark'] as const){const t=paletteTokens('custom',mode);
   for(const bg of ['--app','--panel','--sidebar','--rail','--selected'])for(const fg of ['--text','--muted','--accent'])assert(contrast(t[bg],t[fg])>=4.5,`${seed}/${mode} ${bg}/${fg}`);
   assert(contrast(t['--on-accent'],t['--accent'])>=4.5);
  }
 }}finally{palettes[index]=original;}
});
