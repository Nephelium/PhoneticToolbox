import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import {createSSRApp,h} from 'vue';
import {renderToString} from '@vue/server-renderer';

const require=createRequire(import.meta.url);
function component(name:string){
 const source=readFileSync(new URL(`../src/components/${name}.vue`,import.meta.url),'utf8');
 const {descriptor}=parse(source),compiled=compileScript(descriptor,{id:name,inlineTemplate:true});
 const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const module={exports:{} as any};new Function('require','module','exports',js)(require,module,module.exports);return module.exports.default;
}
const Frame=component('ModuleFrame'),Toolbar=component('ModuleToolbar'),Section=component('ModuleSection'),Status=component('ModuleStatus');
const render=(type:any,props:any,slots:any={})=>renderToString(createSSRApp({render:()=>h(type,props,slots)}));

test('P04 frame keeps toolbar, recoverable error and scientific content together while busy',async()=>{
 const html=await render(Frame,{label:'研究工作区','aria-busy':true,'data-owner':'M12'},{toolbar:()=>h('button','保存'),status:()=>h(Status,{kind:'error',message:'保存失败'}),default:()=>h('input',{value:'井井 ɑ̃˥'})});
 assert.match(html,/aria-label="研究工作区"/);assert.match(html,/aria-busy="true"/);assert.match(html,/data-owner="M12"/);assert.match(html,/value="井井 ɑ̃˥"/);assert(html.indexOf('保存</button>')<html.indexOf('保存失败'));assert(html.indexOf('保存失败')<html.indexOf('<input'));
});
test('P04 actions retain native button semantics and source order without nested toolbar roles',async()=>{
 const html=await render(Toolbar,{label:'文件操作'},{default:()=>[h('button',{disabled:true},'打开'),h('button','刷新')],actions:()=>h('button','方法与引用')});
 assert.match(html,/role="group" aria-label="文件操作"/);assert.doesNotMatch(html,/role="toolbar"|tabindex="-1"/);assert.match(html,/<button disabled>打开/);assert(html.indexOf('刷新')<html.indexOf('方法与引用'));
});
test('P04 status announces failure separately from loading and escapes file/error text',async()=>{
 const error=await render(Status,{kind:'error',message:'<img src=x> 文件已失效'},{default:()=>h('button','重新选择')});
 assert.match(error,/role="alert"/);assert.match(error,/&lt;img src=x&gt;/);assert.match(error,/>重新选择<\/button>/);
 assert.match(await render(Status,{kind:'loading',message:'读取中'}),/role="status" aria-busy="true"/);
 assert.doesNotMatch(await render(Status,{kind:'empty',message:'请选择文件'}),/role="alert"|role="status"/);
});
test('P04 section retains accessible name when its visual title is omitted',async()=>{
 const html=await render(Section,{label:'科学绘图区'},{default:()=>h('svg',{'aria-label':'真实波形'})});
 assert.match(html,/aria-label="科学绘图区"/);assert.doesNotMatch(html,/<h[12]/);assert.match(html,/aria-label="真实波形"/);
 const titled=await render(Section,{label:'参数区',title:'分析参数'},{actions:()=>h('button','保存参数草稿')});assert.match(titled,/<h2>分析参数<\/h2>/);assert.match(titled,/保存参数草稿/);
});
