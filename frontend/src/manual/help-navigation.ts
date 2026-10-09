import type {ManualTarget} from './types.ts';

/** Stable module identifiers survive display-name changes. */
export function moduleHelpTarget(moduleId?:string,targetId?:string):ManualTarget {
  return {chapterId:moduleId&&/^M(?:0[1-9]|1[0-8])$/.test(moduleId)?moduleId.toLowerCase():'getting-started',...(targetId?{targetId}:{})};
}

export function isModuleHelpLabel(label:string):boolean {
  return ['帮助','使用说明','帮助与来源','帮助与说明'].includes(label.replace(/\s+/g,'').trim());
}
