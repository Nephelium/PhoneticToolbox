/** Persist the validated document before changing the active source settings. */
export async function commitSequenceImport(post,incoming,previous,setPerturbation,previousPerturbation){
  await post('keyframes',incoming);
  try{await setPerturbation(incoming.perturbation??null);}
  catch(error){
    const failures=[];
    try{await post('keyframes',previous);}catch(e){failures.push(e);}
    try{await setPerturbation(previousPerturbation);}catch(e){failures.push(e);}
    if(failures.length){const failed=Error('恢复未完成，请重新打开工作区核对保存记录。'+error.message);failed.sequenceRestored=false;throw failed;}
    throw error;
  }
}
