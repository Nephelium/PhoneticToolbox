import type {ManualTarget} from './types.ts';

/** Stable module identifiers survive display-name changes. */
export function moduleHelpTarget(moduleId?:string,targetId?:string,introduction='getting-started'):ManualTarget {
  return {chapterId:moduleId&&/^M(?:0[1-9]|1[0-9])$/.test(moduleId)?moduleId.toLowerCase():introduction,...(targetId?{targetId}:{})};
}

const normalizeHelpLabel=(label:string)=>label.normalize('NFKC').replace(/\s+/g,'').toLowerCase();
// Compatibility for previously rendered pages. New controls use the stable
// data-action="module-help" identity; translated text is never a module ID.
const legacyHelpLabels=new Set([
  '帮助','使用说明','帮助与来源','帮助与说明',
  'Help','User manual','Help and sources','Help and instructions','Help & sources',
  'ヘルプ','マニュアル','ユーザーマニュアル','使い方','ヘルプと出典','ヘルプと説明',
].map(normalizeHelpLabel));
export function isModuleHelpLabel(label:string):boolean {return legacyHelpLabels.has(normalizeHelpLabel(label));}
