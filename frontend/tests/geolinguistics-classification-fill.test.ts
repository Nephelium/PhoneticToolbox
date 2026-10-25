import test from 'node:test';
import assert from 'node:assert/strict';
import {classificationPolygonColors,emptyProject,categoryColor,type Admin,type ClassificationRow} from '../src/modules/geolinguistics/model.ts';
const node=(id:string,parent:string,level:Admin['level']):Admin=>({id,parent,level,name:id,code:id,province:'p',center:[110,30],extent:[109,29,111,31],source:'test'});
const row=(id:string,admin:string,path:string[]):ClassificationRow=>({id,admin,sheet:'combined',row:1,values:['','','','','','',...path],original:[],match:'test'});
test('regional faces include descendant-only cities and keep mixed classifications instead of arbitrarily choosing the city row',()=>{
 const index=new Map([node('p','','province'),node('c1','p','city'),node('c2','p','city'),node('d1','c1','district'),node('d2','c2','district'),node('empty','p','city')].map(a=>[a.id,a]));
 const rows=[row('own','c1',['甲','a','同名']),row('child','d1',['乙','b','同名']),row('only-child','d2',['丙','c','同名'])],before=structuredClone(rows),project=emptyProject();
 const colors=classificationPolygonColors(rows,project,index,3);assert.equal(colors.c1,'#647d8b');assert.equal(colors.p,'#647d8b');assert.equal(colors.c2,categoryColor('丙 / c / 同名'));assert.equal(colors.d2,colors.c2);assert.equal(colors.empty,undefined);assert.deepEqual(rows,before);
 const filtered=classificationPolygonColors([rows[1]],project,index,3);assert.equal(filtered.c1,categoryColor('乙 / b / 同名'));assert.equal(filtered.p,filtered.c1);assert.equal(filtered.c2,undefined);
 project.classEdits.child=[...rows[1].values.slice(0,6),'甲','a','同名'];assert.equal(classificationPolygonColors(rows,project,index,3).c1,categoryColor('甲 / a / 同名'));assert.deepEqual(rows,before);
});
