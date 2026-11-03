import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
/** Persist the validated document before changing the active source settings. */
export async function commitSequenceImport(post,incoming,previous,setPerturbation,previousPerturbation){
  await post('keyframes',incoming);
  try{await setPerturbation(incoming.perturbation??null);}
  catch(error){
    const failures=[];
    try{await post('keyframes',previous);}catch(e){failures.push(e);}
    try{await setPerturbation(previousPerturbation);}catch(e){failures.push(e);}
    if(failures.length){const failed=uiError(join(()=>msg("modules.vocal-tract-frame.text.restoration_incomplete_reopen_and_inspect_saved_records_302"),()=>error.message));failed.sequenceRestored=false;throw failed;}
    throw error;
  }
}
