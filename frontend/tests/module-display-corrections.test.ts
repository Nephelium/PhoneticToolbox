import test from 'node:test';
import assert from 'node:assert/strict';
import VectorSource from 'ol/source/Vector.js';
import {setLocale,messageRef,missingMessages} from '../src/i18n/index.ts';
import {uiErrorMessage} from '../src/modules/module-localization.ts';
import {beamChanged,defaults as mfaDefaults} from '../src/modules/mfa/state.ts';
import {plotGroupTitle} from '../src/modules/parameter-display/state.ts';
import {emptyProject,type MapPoint} from '../src/modules/geolinguistics/model.ts';
import {syncMapPointFeatures} from '../src/modules/geolinguistics/point-features.ts';
import {catalog} from '../src/modules/ipa-plus/catalog.ts';
import {localizedSourceLocator} from '../src/modules/ipa-plus/source-localization.ts';
import {ipaTextFilename} from '../src/modules/ipa-plus/filename.ts';

test('existing UI validation errors and default display captions follow language without changing user data',async()=>{
 await setLocale('zh-CN');const error=messageRef();
 try{beamChanged(mfaDefaults(),0);}catch(value){error.value=uiErrorMessage(value);}
 const original={id:1,title:'图窗 1',parameters:['F0']},custom={id:2,title:'井井的F0',parameters:['F0']};
 const fixed=JSON.stringify([original,custom]),errors=new Set<string>(),titles=new Set<string>(),newNames=new Set<string>();
 const firstProject=emptyProject(),firstName=firstProject.name;
 for(const language of ['zh-CN','en','ja'] as const){await setLocale(language);errors.add(error.value);titles.add(plotGroupTitle(original));newNames.add(emptyProject().name);assert.equal(plotGroupTitle(custom),custom.title);assert.equal(firstProject.name,firstName);assert.equal(ipaTextFilename(new Date('2026-10-30T00:00:00Z')),'国际音标-2026-10-30.txt');}
 assert.equal(errors.size,3);assert.equal(titles.size,3);assert.equal(newNames.size,3);assert.equal(JSON.stringify([original,custom]),fixed);
 await setLocale('zh-CN');
});

test('all visible Chinese source locators localize with stable numeric locations and no catalogue mutation',async()=>{
 const before=JSON.stringify(catalog),digits=(text:string)=>text.match(/\d+(?:\.\d+)*/g)??[];
 for(const language of ['zh-CN','en','ja'] as const){await setLocale(language);for(const entry of catalog.entries)for(const source of entry.sourceRefs){if(['ipa-chart-zh-2007','ipa-handbook-jiang-2008'].includes(source.sourceId)||!/[\u3400-\u9fff]/u.test(source.locator))continue;const display=localizedSourceLocator(entry.id,source);assert.deepEqual(digits(display).sort(),digits(source.locator).sort(),entry.id);assert.ok(!display.startsWith('modules.'),entry.id);if(language==='en')assert.ok(!/[\u3400-\u9fff]/u.test(display),entry.id);}}
 assert.equal(JSON.stringify(catalog),before);assert.deepEqual(missingMessages().filter(key=>key.includes('modules.ipa-plus.locator.')),[]);await setLocale('zh-CN');
});

test('localized map point labels reuse existing Feature and geometry; only coordinate edits reproject',()=>{
 const source=new VectorSource(),point:MapPoint={id:'stable-site',name:'珠海',lon:113.58,lat:22.27,color:'#123456',label:'原值待核'};
 assert.deepEqual(syncMapPointFeatures(source,[point]),{created:1,projected:1});
 const feature=source.getFeatureById(point.id)!,geometry=feature.getGeometry();
 for(const label of ['Original value pending verification','元の値は未確認','原值待核']){assert.deepEqual(syncMapPointFeatures(source,[{...point,label}]),{created:0,projected:0});assert.equal(source.getFeatureById(point.id),feature);assert.equal(feature.getGeometry(),geometry);assert.equal(feature.get('datum').label,label);}
 assert.deepEqual(syncMapPointFeatures(source,[{...point,lon:114}]),{created:0,projected:1});assert.equal(source.getFeatureById(point.id),feature);assert.notEqual(feature.getGeometry(),geometry);
 syncMapPointFeatures(source,[]);assert.equal(source.getFeatures().length,0);source.dispose();
});
