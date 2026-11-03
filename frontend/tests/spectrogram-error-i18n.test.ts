import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import * as vue from 'vue';
import * as i18n from '../src/i18n/index.ts';
import * as display from '../src/modules/module-localization.ts';

const source=readFileSync(new URL('../src/components/SpectrogramViewport.vue',import.meta.url),'utf8');
const script=compileScript(parse(source).descriptor,{id:'spectrogram-error-test',inlineTemplate:false});
const javascript=ts.transpileModule(script.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;

function preview(failure:Error){
 const cleanup:(()=>void)[]=[],scope=vue.effectScope(),module={exports:{}as any};let calls=0;
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onUnmounted:(fn:()=>void)=>cleanup.push(fn)};
  if(name==='../i18n/index.ts')return i18n;
  if(name==='../modules/module-localization.ts')return display;
  throw Error('Unexpected runtime dependency: '+name);
 };
 new Function('require','module','exports',javascript)(require,module,module.exports);
 const props=vue.reactive({loader:async()=>{calls++;throw failure;},start:0,end:1,channel:0,width:640});
 const instance=scope.run(()=>module.exports.default.setup(props,{emit:()=>{},expose:()=>{}}));
 return {instance,calls:()=>calls,close(){for(const fn of cleanup)fn();scope.stop();}};
}

test('actual spectrogram SFC retains a single failed preview across language changes',async()=>{
 await i18n.setLocale('en');
 const fixtures=[
  [new Error('preview_timeout'),'components.spectrogram_computation_timed_out_choose_a_shorter_time_window'],
  [new Error('preview_memory_exceeded'),'components.the_spectrogram_exceeds_the_preview_memory_budget_choose_shorter'],
  [new Error('preview_runtime_unavailable'),'components.the_praat_scientific_runtime_is_unavailable_in_this_environment'],
  [new Error('invalid_spectrogram_input'),'components.a_spectrogram_cannot_be_generated_for_this_audio_or'],
  [display.uiError(display.uiMessage('components.the_spectrogram_pixel_count_is_inconsistent')),'components.the_spectrogram_pixel_count_is_inconsistent'],
  [new Error('Raw diagnostic: 样本-{V}.wav'),null],
 ]as const;
 try{
  for(const [failure,key]of fixtures){
   const fixture=preview(failure);
   try{
    await fixture.instance.run();
    for(const language of ['en','ja','zh-CN']as const){
     await i18n.setLocale(language);await vue.nextTick();
     assert.equal(fixture.instance.error.value,key?i18n.tr(key):failure.message);
     assert.equal(fixture.calls(),1,'changing display language must not retry the preview');
    }
   }finally{fixture.close();}
  }
 }finally{await i18n.setLocale('zh-CN');}
});
