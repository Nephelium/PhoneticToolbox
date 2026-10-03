import {test} from 'node:test';
import assert from 'node:assert/strict';
import {palettes,paletteTokens,normalizeMode,normalizePalette,contrast} from '../src/design/themes.ts';
import {defaults,normalizeFonts} from '../src/design/fonts.ts';
test('P19 old default and malformed preferences recover to Everforest, preserving other choices',()=>{
 for(const value of [undefined,null,'ptb',{},42,'missing','dark;display:none']){assert.equal(normalizePalette(value),'everforest');assert.equal(normalizeMode(value),'system');}
 assert.equal(normalizeMode('dark'),'dark');assert.equal(normalizePalette('matrix'),'matrix');
 assert(!palettes.some(p=>p.id==='ptb'));
 for(const mode of ['light','dark'] as const)assert.deepEqual(paletteTokens('ptb',mode),paletteTokens('everforest',mode));
});
test('P19 every paired UI palette has readable text, controls and status colors',()=>{
 assert.equal(new Set(palettes.map(p=>p.id)).size,palettes.length);
 assert.equal(contrast('#000000','#ffffff'),21);
 for(const palette of palettes)for(const mode of ['light','dark'] as const){
  const c=paletteTokens(palette.id,mode);
  for(const background of ['--app','--panel','--sidebar','--selected'])for(const role of ['--text','--muted','--accent','--success','--warning','--danger'])assert(contrast(c[role],c[background])>=4.5,`${palette.id}/${mode} ${role} on ${background}`);
  assert(contrast(c['--on-accent'],c['--accent'])>=4.5);
  assert(contrast(c['--text'],c['--selected'])>=4.5,`${palette.id}/${mode} selection`);
 }
});
test('P19 requested defaults and saved user choices remain distinct',()=>{
 const d=defaults();assert.equal(d.zh,'SimSun');assert.equal(d.latin,'Times New Roman');assert.equal(d.mono,'JetBrains Mono');assert.equal(d.ipa,'Doulos SIL');
 const old=normalizeFonts({...d,zh:'KaiTi',latin:'Georgia',mono:'Consolas'});assert.equal(old.zh,'KaiTi');assert.equal(old.latin,'Georgia');assert.equal(old.mono,'Consolas');
});
