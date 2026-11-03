import {uiMessage} from '../module-localization.ts';
const kinds:Record<string,string>={'manual-pause':'manual','module-hidden':'moduleHidden','page-freeze':'pageFreeze','page-resume':'pageResume','window-blur':'windowBlur','document-hidden':'documentHidden','pagehide':'pageHidden','devicechange':'deviceChanged','fullscreen-exit':'fullscreenExit','scheduler-gap':'schedulerGap','partial-export':'partialExport','leave-runner':'leaveRunner'};
/** Event codes/details remain research records; only the current UI status localizes. */
export function interruptionDisplay(kind:string,detail=''){
 const key=kinds[kind]??(kind.startsWith('audio-context-')?'audioContext':undefined);
 return key?uiMessage('modules.perception.interruption.'+key):detail||kind;
}
