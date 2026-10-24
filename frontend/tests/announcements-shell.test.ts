import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse} from '@vue/compiler-sfc';
import ts from 'typescript';
// Execute the actual SFC guard, including tasks running while their tab is hidden.
test('actual AppShell home guard defers notices for capture, running experiments, saves, updates and hidden documents',()=>{
 const raw=readFileSync(new URL('../src/app/AppShell.vue',import.meta.url),'utf8'),script=parse(raw).descriptor.scriptSetup!.content;
 const file=ts.createSourceFile('AppShell.vue',script,ts.ScriptTarget.Latest,true,ts.ScriptKind.TS),guard=file.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text==='announcementTaskSafe');assert(guard);
 const names=['active','modal','closing','closingBusy','updateClosing','m16Recording','m15Focus'];
 const env:any=Object.fromEntries(names.map(name=>[name,{value:name==='active'?'home':['closingBusy','m16Recording','m15Focus'].includes(name)?false:''}]));
 let capture:string|undefined,experiment=false;env.captureOwner=()=>capture;env.m15Page={value:{updateBlocked:()=>experiment}};env.document={hidden:false};
 const compiled=ts.transpileModule(guard.getText(file)+'\nreturn announcementTaskSafe;',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText;
 const safe=new Function(...Object.keys(env),compiled)(...Object.values(env));assert.equal(safe(),true);
 for(const name of names){const previous=env[name].value;env[name].value=name==='active'?'M15':typeof previous==='boolean'?true:'owned-busy';assert.equal(safe(),false,name);env[name].value=previous;}
 capture='M16';assert.equal(safe(),false);capture='M05';assert.equal(safe(),false);capture=undefined;
 experiment=true;assert.equal(safe(),false,'experiment still running after returning home');experiment=false;env.document.hidden=true;assert.equal(safe(),false);env.document.hidden=false;assert.equal(safe(),true);
});
