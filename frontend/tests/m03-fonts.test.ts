import {test} from 'node:test';
import assert from 'node:assert/strict';
import {preflightExportFonts} from '../src/modules/egg-analysis/fonts.ts';
import {defaults,taskConfig} from '../src/modules/egg-analysis/state.ts';
const font={schema_version:'font/1' as const,zh:'SimSun',latin:'Arial',ipa:'Doulos SIL' as const,size_px:12};
test('data-only modes do not depend on worker fonts',async()=>{for(const mode of ['preview','inverse','batch'] as const){const config=taskConfig(defaults(),mode);config.generate_images=false;const value=await preflightExportFonts(config,font,async()=>{throw Error('must not run');});assert.deepEqual(value.font,font);}});
test('missing fonts name role and preserve IPA requirement',async()=>{await assert.rejects(preflightExportFonts(taskConfig(defaults(),'single'),font,async()=>({available:false,fonts:[{role:'latin',requested:'MissingFont',available:false,family:null,sha256:null}]})),/英文与数字 MissingFont.*Doulos SIL/);});
test('font snapshot remains the checked snapshot after preferences change',async()=>{const selected={...font};const result=await preflightExportFonts(taskConfig(defaults(),'single'),selected,async(snapshot)=>{selected.latin='Changed';assert.equal(snapshot.latin,'Arial');return {available:true,fonts:[]};});assert.equal(result.font.latin,'Arial');});
test('missing capability and failed preflight block image submission',async()=>{const config=taskConfig(defaults(),'single');await assert.rejects(preflightExportFonts(config,font,undefined),/最新工作台/);await assert.rejects(preflightExportFonts(config,font,async()=>{throw Error('503');}),/暂时无法检查/);});
