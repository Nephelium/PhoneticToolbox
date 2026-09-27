import {test} from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
const css=readFileSync(new URL('../src/design/tokens.css',import.meta.url),'utf8');
function luminance(hex:string){const rgb=hex.length===4?[...hex.slice(1)].map(x=>x+x).join(''):hex.slice(1);return [0,2,4].map((p)=>parseInt(rgb.slice(p,p+2),16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);}
test('P04 text/token pairs meet AA contrast in both themes',()=>{
 const blocks=[css.match(/:root\{([^}]+)/)![1],css.match(/:root\[data-theme=dark\]\{([^}]+)/)![1]];
 for(const [index,block] of blocks.entries()){
  const tokens=Object.fromEntries([...block.matchAll(/--([\w-]+):(#[0-9a-f]+)/g)].map(m=>[m[1],m[2]]));
  for(const [fg,bg] of [['text','panel'],['text','selected'],['muted','panel'],['muted','sidebar'],['accent','selected'],['on-accent','accent'],['teal','teal-bg'],['violet','violet-bg'],['warning','app'],['danger','panel']]){
   const a=luminance(tokens[fg]),b=luminance(tokens[bg]);const ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);assert.ok(ratio>=4.5,`${index}:${fg}/${bg} = ${ratio}`);
  }
 }
});
test('P04 native theme popup has explicit opaque surfaces and readable option text',()=>{
 assert.match(css,/select option,select optgroup\{background-color:var\(--panel\);color:var\(--text\)\}/);
 assert.match(css,/select option:checked\{background-color:var\(--selected\);color:var\(--text\)\}/);
 assert.match(css,/input,select\{[^}]*background:var\(--panel\)/);
});
