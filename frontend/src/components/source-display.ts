import {tr} from '../i18n/index.ts';
import original from '../i18n/locales/zh-CN/source-notes.json' with {type:'json'};

const notes=original as Record<string,string>;
export type SourceDisplayField='reference_note'|'license'|'attribution'|'adaptation_note'|'adaptation_caution';
/** Product explanations only; citations, authors, titles and identifiers stay literal. */
export function sourceText(source:{id:string}&Partial<Record<SourceDisplayField,string>>,field:SourceDisplayField):string{
 const value=source[field]??'',key='sources.metadata.'+source.id+'.'+field;
 return notes[key]===value?tr(key):value;
}
