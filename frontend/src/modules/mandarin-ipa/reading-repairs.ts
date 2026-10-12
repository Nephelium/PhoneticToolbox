import rules from './reading-repairs.json' with {type:'json'};

export const readingRepairVersion=rules.version;
export const readingRepairs=rules.repairs;
const nanCharacters=new Set([...rules.missing_nan_characters]);

export function normalizeReading(pinyin:string,tone:number,char=''){
  const repair=readingRepairs.find(row=>char&&row.characters.includes(char)&&row.legacy_pinyin===pinyin&&row.legacy_tone===tone);
  return repair?{pinyin:repair.pinyin,tone:repair.tone}:{pinyin:!pinyin&&nanCharacters.has(char)?'nan':pinyin,tone};
}
