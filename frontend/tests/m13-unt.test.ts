import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {autoErhuaPairs,convertText,createDraft,moveErhuaPairs,plainOutput,restoreDraft,snapshotDraft,variantsFor} from '../src/modules/mandarin-ipa/state.ts';
import {untTranscription,untVersion} from '../src/modules/mandarin-ipa/unt.ts';
import {normalizeReading,readingRepairs} from '../src/modules/mandarin-ipa/reading-repairs.ts';

test('Unt revision and authorized reading repairs preserve all unaffected legacy cells and row order',()=>{
  const path=new URL('../../phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html',import.meta.url);
  const legacy=readFileSync(path,'utf8');
  assert.equal(createHash('sha256').update(legacy).digest('hex'),'11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451');
  const rows=JSON.parse(legacy.match(/const\s+ipaData\s*=\s*(\[[\s\S]*?\]);/)![1].replace(/([:,]\s*)NaN(?=\s*[,}\]])/g,'$1null'));
  const current=JSON.parse(readFileSync(new URL('../src/modules/mandarin-ipa/ipa-data.json',import.meta.url),'utf8'));
  assert.equal(current.rows.length,rows.length);
  let corrected=0,restored=0;
  for(let i=0;i<rows.length;i++){
    const before=Object.values(rows[i]) as (string|number|null)[],after=current.rows[i];
    const reading=normalizeReading(String(before[2]??''),Number(before[1]),String(before[0]));
    const repair=readingRepairs.find(r=>r.characters.includes(String(before[0]))&&r.legacy_pinyin===before[2]&&r.legacy_tone===before[1]);
    assert.equal(after[0],before[0],`character row ${i}`);
    assert.deepEqual(after.slice(1,3),[reading.tone,reading.pinyin||before[2]],`reading row ${i}`);
    if(repair){if(!repair.overwrite_existing_mismatched_reading)assert(before.slice(3,11).every(v=>v===null));assert(after.slice(3,11).every((v:unknown)=>typeof v==='string'&&v));corrected++;}
    else{assert.deepEqual(after.slice(3,11),before.slice(3,11),`mapping row ${i}`);if(before[2]===null&&reading.pinyin==='nan')restored++;}
  }
  assert.equal(corrected,10);assert.equal(restored,29);
  assert.deepEqual(current.unt_revision,untVersion);
  assert.equal(untVersion.revisionHistoryThrough,'2024-10-11');
  assert.equal(untVersion.articleTextSha256.length,64);
  const missing=current.rows.filter((row:any[])=>!row[11]||!row[12]);
  assert.deepEqual(missing.map((row:any[])=>[row[0],row[2]]),[['萡','be']]);
});

test('confirmed reading repairs are consistent in pinyin and every IPA standard',()=>{
  const current=JSON.parse(readFileSync(new URL('../src/modules/mandarin-ipa/ipa-data.json',import.meta.url),'utf8'));
  const diu=['tiou˥','tio̟u˥','tiou˥','tiou˥','tiɤᵘ˥','tioʊ˥','d̥iou˥','tioʊ˥','tjoʊ˥','d̥j͗ȏ̜ʊ˥'];
  for(const char of [...'丢丟铥銩']){
    const row=current.rows.find((r:any[])=>r[0]===char);assert.deepEqual(row.slice(1),[1,'diu',...diu]);assert.equal(plainOutput(convertText(char,'汉语拼音')),'diū');
  }
  for(const [char,p,t] of [['呗','bei',0],['唄','bei',0],['葛','ge',2],['数','shu',4],['额','e',2],['距','ju',4]] as const){
    const rows=current.rows.filter((r:any[])=>r[0]===char&&r[2]===p&&r[1]===t);
    const donor=current.rows.find((r:any[])=>r[0]!==char&&r[2]===p&&r[1]===t);
    assert(rows.length);for(const row of rows)assert.deepEqual(row.slice(3),donor.slice(3));
  }
  assert.equal(plainOutput(convertText('囡男南楠','汉语拼音')),'nānnánnánnán');
  assert.equal(plainOutput(convertText('额距','汉语拼音')),'éjù');
  assert.equal(plainOutput(convertText('萡','Standard Chinese (Beijing)')),'?');
});

test('Unt current body corrects segment conditions, er and the middle third tone',()=>{
  const cases:[string,number,string,string][]=[
    ['zi',1,'tsɹ̩˥','d̥͡z̟̊ɹ̟̍˥'], ['zhi',1,'tʂɻ̍˥','d̥͡ʐ̺̊ɻ̺̍˥'],
    ['chun',1,'tʂʰwən˥','t͡ʂ̺ʰuɘ̹̑˗n̚˥'], ['qu',4,'tɕʰy˥˩','t͡ɕʰy˥˩'],
    ['er',2,'ɚ˧˥','ə̠ɻ˕˧˥'], ['er',4,'ɚ˥˩','ɐɻ˕˥˩'],
    ['ma',3,'ma˨˩˦','mä˨˩˧'], ['dong',1,'tʊŋ˥','d̥ʷʊŋʷ̚˥'],
    ['xiong',1,'ɕjʊŋ˥','ɕʷɥ͑ʊ̟ŋʷ̚˥'], ['yi',1,'i˥','iᶨ˥'],
    ['wu',3,'u˨˩˦','uᵝ˨˩˧'], ['yu',2,'y˧˥','yᶨ˧˥'],
    ['wei',1,'weɪ˥','we̠ɪ̯˥'], ['gui',1,'kweɪ˥','ɡ̊ʷwe̠ɪ̯˥'],
    ['dui',4,'tweɪ˥˩','d̥uɘ̹̑ɪ̯˥˩'], ['fu',1,'fʋ̩˥','fʋ̩ˠ˥'],
    ['liu',2,'ljoʊ˧˥','l̠ʲj͗ȏ̜ʊ˧˥'], ['diu',1,'tjoʊ˥','d̥j͗ȏ̜ʊ˥'],
  ];
  for(const [p,t,b,n]of cases){assert.equal(untTranscription(p,t,false),b,p);assert.equal(untTranscription(p,t,true),n,p);}
  assert.equal(plainOutput(convertText('的么','UntPhesoca宽')),'təmə');
  assert.equal(plainOutput(convertText('的么','UntPhesoca严')),'d̥əmə');
  assert.equal(untTranscription('de',2,true,'德'),'d̥ɤ̽˧˥');
  assert.equal(plainOutput(convertText('男','UntPhesoca严')),'na̠n̚˧˥');
  assert.equal(plainOutput(convertText('丢','UntPhesoca严')),'d̥j͗ȏ̜ʊ˥');
  assert.equal(plainOutput(convertText('呗','UntPhesoca宽')),'peɪ');
});

test('Unt erhua replaces whole finals with the authored mergers, nasalization and tone',()=>{
  const cases:[string,number,string,string][]=[
    ['hua',1,'xwɐɚ˥','x̞ʷwɐɻ˕˥'], ['wan',2,'wɐɚ˧˥','wɐɻ˕˧˥'],
    ['dian',3,'tjɐɚ˨˩˦','d̥jɐ̟ɻ˕˨˩˧'], ['xian',4,'ɕjɐɚ˥˩','ɕjɐ̟ɻ˕˥˩'],
    ['quan',1,'tɕʰɥɐɚ˥','t͡ɕʰʷɥʏ̯̈ɐ̹ɻ˕˥'], ['men',2,'mɚ˧˥','mə̠ɻ˕˧˥'],
    ['jin',1,'tɕiɚ˥','d̥͡ʑ̥ji̞ə̯ɻ˕˥'], ['gun',3,'kwɚ˨˩˦','ɡ̊ʷwə̠ɻ˕˨˩˧'],
    ['dun',4,'twɚ˥˩','d̥uɘ̹̑˗ɻ˕˥˩'], ['yun',2,'yɚ˧˥','ɥʏɘ̹̑ɻ˕˧˥'],
    ['zi',3,'tsɚ˨˩˦','d̥͡z̟̊ə̠ɻ˕˨˩˧'], ['zhi',3,'tʂɚ˨˩˦','d̥͡ʐ̺̊ə̠ɻ˕˨˩˧'],
    ['hao',3,'xɑ˞ʊ˞˨˩˦','x̞ɑ̹̽˞ʊ̯˞˨˩˧'], ['dou',4,'to˞ʊ˞˥˩','d̥ʷo̜̽˞ʊ̯˞˥˩'],
    ['gang',1,'kɑ̃˞˥','ɡ̊ɑ̽̃˞˥'], ['ding',1,'tiɘ̃˞˥','d̥iɘ̯̃˞˥'],
    ['dong',1,'tʊ̃˞˥','d̥ʊ̃˞˥'], ['xiong',1,'ɕjʊ̃˞˥','ɕʷɥ͑ʊ̟̃˞˥'],
    ['fu',2,'fʊ˞ᶹ˧˥','fʊ̜˞ᶹ˧˥'], ['jie',1,'tɕjɛ˞˥','d̥͡ʑ̥jɚ˥'],
  ];
  for(const [p,t,b,n]of cases){assert.equal(untTranscription(p,t,false,'',true),b,p);assert.equal(untTranscription(p,t,true,'',true),n,p);}
  for(const p of ['er','m','n','ng','yo'])assert.equal(untTranscription(p,2,true,'',true),null,p);
});

test('Erhua decisions are reversible per position and leave the other standards intact',()=>{
  const text='花儿女儿花 儿花，儿花\n儿兒子';
  const pairs=autoErhuaPairs(text,'UntPhesoca宽');
  assert.deepEqual(pairs,{'花_0':true,'女_2':true});
  let tokens=convertText(text,'UntPhesoca宽',{},true,pairs);
  assert.equal(tokens.filter(t=>t.kind==='mapped'&&t.erhua).length,2);
  const flower=tokens[0];assert.equal(flower.kind,'mapped');if(flower.kind!=='mapped')return;
  assert.equal(flower.char,'花儿');assert.equal(flower.value,'xwɐɚ˥');
  delete pairs['女_2'];tokens=convertText(text,'UntPhesoca宽',{},true,pairs);
  assert.equal(tokens.filter(t=>t.kind==='mapped'&&t.erhua).length,1);
  assert.equal(plainOutput(convertText('花儿','UntPhesoca严',{},false,{'花_0':true})),'x̞ʷwɐɻ˕');
  for(const standard of ['Standard Chinese (Beijing)','汉语拼音'] as const)assert.deepEqual(convertText(text,standard,{},true,pairs),convertText(text,standard));
  assert.equal(Object.keys(autoErhuaPairs('儿儿儿，儿子','UntPhesoca严')).length,0);
});

test('Erhua preserves selected base readings, edits and compatible v1 drafts',()=>{
  const text='行儿',choices={'行_0':1},pairs=autoErhuaPairs(text,'UntPhesoca严',choices);
  const token=convertText(text,'UntPhesoca严',choices,true,pairs)[0];assert.equal(token.kind,'mapped');if(token.kind!=='mapped')return;
  assert.equal(token.selectedVariant,1);assert.equal(token.value,'x̞ɑ̽̃˞˧˥');
  assert.equal(variantsFor(token,'UntPhesoca严')[1].value,token.value);
  assert.deepEqual(moveErhuaPairs('花儿女儿','😀花儿女儿',{'花_0':true,'女_2':true}),{'花_1':true,'女_3':true});
  assert.deepEqual(moveErhuaPairs('花儿','花 儿',{'花_0':true}),{});
  assert.deepEqual(moveErhuaPairs('花儿','花兒',{'花_0':true}),{});
  const saved={...createDraft(),text,standard:'UntPhesoca严' as const,selectedVariants:choices,erhuaPairs:pairs};
  assert.deepEqual(restoreDraft(snapshotDraft(saved)),saved);
  const snapshot=snapshotDraft(saved);delete saved.erhuaPairs['行_0'];assert.equal(snapshot.erhuaPairs['行_0'],true);
  assert.deepEqual(restoreDraft({version:1,text:'花儿'}).erhuaPairs,{});
  assert.deepEqual(restoreDraft({...saved,erhuaPairs:{'行_0':true,'行_999':true,'bad':true}}).erhuaPairs,{'行_0':true});
});
