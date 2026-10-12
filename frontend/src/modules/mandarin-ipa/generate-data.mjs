import {createHash} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {untTranscription,untVersion} from './unt.ts';
import {normalizeReading,readingRepairs,readingRepairVersion} from './reading-repairs.ts';

const sourceUrl=new URL('../../../../phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html',import.meta.url);
const outputUrl=new URL('./ipa-data.json',import.meta.url);
const expectedSourceSha256='11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451';
const columns=['汉字','声调','拼音','Standard Chinese (Beijing)','Standard Chinese (Beijing)严','胡裕树《现代汉语》','黄伯荣、廖序东《现代汉语》','钱乃荣《现代汉语》','吴宗济','赵元任《汉语口语语法》','《汉语方音字汇》','UntPhesoca宽','UntPhesoca严'];

const source=await readFile(sourceUrl,'utf8');
const sourceSha256=createHash('sha256').update(source).digest('hex');
if(sourceSha256!==expectedSourceSha256)throw Error(`IPA source hash changed: ${sourceSha256}`);
const match=/const\s+ipaData\s*=\s*(\[[\s\S]*?\]);/.exec(source);
if(!match)throw Error('ipaData was not found in the legacy HTML source.');
// Python's historical json.dumps call emitted bare NaN for three missing cells.
// JavaScript accepted those values; normalize only bare value tokens to null so
// the generated resource is strict JSON while preserving the visible `?` path.
const strictJson=match[1].replace(/([:,]\s*)NaN(?=\s*[,}\]])/g,'$1null');
const records=JSON.parse(strictJson);
if(records.length!==21572)throw Error(`Unexpected IPA row count: ${records.length}`);
for(const [index,record] of records.entries()){
  const actual=Object.keys(record);
  if(actual.length!==columns.length||actual.some((name,i)=>name!==columns[i]))throw Error(`Unexpected columns at row ${index}.`);
}
const rows=records.map(record=>columns.map(column=>record[column]??null));
// Confirmed reading-key errors are repaired across standards. Valid legacy
// mappings remain frozen; donors must agree on each of the eight older columns.
function existingValues(pinyin,tone){
  const donors=rows.filter(row=>row[2]===pinyin&&row[1]===tone&&row.slice(3,11).every(value=>typeof value==='string'));
  if(!donors.length)throw Error(`No complete donor for ${pinyin}${tone}`);
  const values=donors[0].slice(3,11);
  if(donors.some(row=>JSON.stringify(row.slice(3,11))!==JSON.stringify(values)))throw Error(`Conflicting donors for ${pinyin}${tone}`);
  return values;
}
const repairedValues=new Map(readingRepairs.map(repair=>{
  if(repair.pinyin!=='diu')return [repair,existingValues(repair.pinyin,repair.tone)];
  const onsets=existingValues('di',1),rimes=existingValues('liu',1);
  return [repair,onsets.map((di,i)=>{
    if(!di.endsWith('i˥')||!rimes[i].startsWith('l'))throw Error('Invalid d/iu donor');
    return di.slice(0,-2)+rimes[i].slice(1);
  })];
}));
let repairedRows=0,restoredNanRows=0;
for(const row of rows){
  const char=String(row[0]),pinyin=String(row[2]??''),tone=Number(row[1]);
  const repair=readingRepairs.find(r=>r.characters.includes(char)&&r.legacy_pinyin===pinyin&&r.legacy_tone===tone);
  const corrected=normalizeReading(pinyin,tone,char);
  if(repair){
    if(row.slice(3,11).some(value=>value!==null)){
      if(!repair.overwrite_existing_mismatched_reading||JSON.stringify(row.slice(3,11))!==JSON.stringify(existingValues(pinyin,tone)))throw Error(`Repair would overwrite an unconfirmed mapping: ${char}`);
    }
    row.splice(3,8,...repairedValues.get(repair));repairedRows++;
  }
  if(row[2]===null&&corrected.pinyin==='nan'){
    if(!row.slice(3,11).every(value=>typeof value==='string'&&value.startsWith('n')))throw Error(`Unconfirmed nan restoration: ${char}`);
    restoredNanRows++;
  }
  row[1]=corrected.tone;row[2]=corrected.pinyin||row[2];
}
if(repairedRows!==10||restoredNanRows!==29)throw Error(`Unexpected reading repairs: ${repairedRows}/${restoredNanRows}`);
// The author revision is confined to the two Unt columns.
for(const row of rows){
  row[11]=untTranscription(String(row[2]??''),Number(row[1]),false,String(row[0]));
  row[12]=untTranscription(String(row[2]??''),Number(row[1]),true,String(row[0]));
}
const uniqueCharacters=new Set(rows.map(row=>row[0])).size;
const counts=new Map();for(const row of rows)counts.set(row[0],(counts.get(row[0])??0)+1);
const duplicateCharacters=[...counts.values()].filter(count=>count>1).length;
const maxVariants=Math.max(...counts.values());
if(uniqueCharacters!==20771||duplicateCharacters!==681||maxVariants!==7)throw Error('IPA data inventory changed.');
const header={schema_version:'m13-ipa-data/1',source_path:'phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html',source_sha256:sourceSha256,rows:rows.length,unique_characters:uniqueCharacters,duplicate_characters:duplicateCharacters,max_variants:7,columns};
const body=`{\n  "schema_version": ${JSON.stringify(header.schema_version)},\n  "source_path": ${JSON.stringify(header.source_path)},\n  "source_sha256": ${JSON.stringify(header.source_sha256)},\n  "unt_revision": ${JSON.stringify(untVersion)},\n  "reading_revision": ${JSON.stringify({version:readingRepairVersion,repaired_rows:repairedRows,restored_nan_rows:restoredNanRows})},\n  "row_count": ${header.rows},\n  "unique_character_count": ${header.unique_characters},\n  "duplicate_character_count": ${header.duplicate_characters},\n  "max_variants": ${header.max_variants},\n  "columns": ${JSON.stringify(columns)},\n  "rows": [\n${rows.map(row=>'    '+JSON.stringify(row)).join(',\n')}\n  ]\n}\n`;
if(process.argv.includes('--check')){
  if(await readFile(outputUrl,'utf8')!==body)throw Error('M13 mapping must be regenerated.');
}else await writeFile(outputUrl,body,'utf8');
console.log(JSON.stringify({...header,unt_revision:untVersion,output:fileURLToPath(outputUrl)}));
