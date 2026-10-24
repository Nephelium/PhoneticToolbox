import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyProject,defaultSettings,inRegion,regionSelection,validateProject,type Admin} from '../src/modules/geolinguistics/model.ts';

const province={id:'province:500000',parent:'100000',name:'重庆市',province:'province:500000',level:'province'} as Admin;
const district={id:'district:500157',parent:province.id,name:'两江新区',province:province.id,level:'district'} as Admin;
const index=new Map([[province.id,province],[district.id,district]]);

test('a valid selection keeps its existing region filter',()=>{
 assert.deepEqual(regionSelection(province.id,index),{effective:province.id,missing:false});
 assert.equal(inRegion(district.id,regionSelection(province.id,index).effective,index),true);
 assert.equal(inRegion(null,regionSelection(province.id,index).effective,index),false);
});

test('an unavailable historical selection shows all sites without rewriting the saved project',()=>{
 const original=emptyProject();original.settings={...defaultSettings(),region:'district:500112'};
 original.edits['S1:F1_P']='原记录';original.customSites=[{id:'custom:1',name:'原点',lon:106.5,lat:29.6,value:'甲',note:''}];
 const project=validateProject(original),selection=regionSelection(project.settings!.region,index);
 assert.deepEqual(selection,{effective:'',missing:true});
 assert.equal(inRegion(null,selection.effective,index),true);
 assert.equal(project.settings!.region,'district:500112');
 assert.deepEqual(project.edits,original.edits);assert.deepEqual(project.customSites,original.customSites);
 assert.equal(regionSelection(project.settings!.region,new Map([...index,['district:500112',{...district,id:'district:500112'}]])).missing,false);
});

test('a deliberate nationwide selection needs no recovery notice',()=>{
 assert.deepEqual(regionSelection('',index),{effective:'',missing:false});
});
