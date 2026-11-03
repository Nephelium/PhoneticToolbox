import {tr} from '../i18n/index.ts';
import katex from 'katex';

/** Only KaTeX's bounded compiler produces markup. HTML extensions stay disabled. */
export function renderFormula(latex:unknown,display:unknown){
  const source=typeof latex==='string'?latex:'';
  if(!source||source.length>12000)return {source,html:'',error:tr('manual.reader.formulaSize')};
  try{
    return {source,html:katex.renderToString(source,{displayMode:display===true,throwOnError:true,
      trust:false,strict:'ignore',maxExpand:1000,maxSize:20,output:'htmlAndMathml'}),error:''};
  }catch{return {source,html:'',error:tr('manual.reader.formulaError')};}
}
