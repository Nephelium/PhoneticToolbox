import { Node, Mark, Extension, mergeAttributes, getSchema, type JSONContent } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { TextStyleKit } from '@tiptap/extension-text-style';
import TextAlign from '@tiptap/extension-text-align';
import { TableKit } from '@tiptap/extension-table';
import Subscript from '@tiptap/extension-subscript';
import Superscript from '@tiptap/extension-superscript';
import Highlight from '@tiptap/extension-highlight';
import { Plugin } from '@tiptap/pm/state';
import { toEditorDocument, sourceNodeType } from './document-codec.ts';

const values = (defaults: Record<string, unknown>) => Object.fromEntries(Object.entries(defaults).map(([key,value])=>[key,{default:value,parseHTML:(el:HTMLElement)=>el.getAttribute('data-'+key) ?? value,renderHTML:(attrs:Record<string,unknown>)=>attrs[key]==null?{}:{['data-'+key]:attrs[key]}}]));
export const EditorialAttributes = Extension.create({
  name:'editorialAttributes',
  addGlobalAttributes(){return [{types:['paragraph','heading','table','tableCell','tableHeader','codeBlock','blockquote'],attributes:{
    ...values({id:null,caption:null,note:null,color:null,textAlign:null,colorDark:null,backgroundColorDark:null}),
    lineHeight:{default:null,parseHTML:e=>e.style.lineHeight || null,renderHTML:a=>a.lineHeight?{style:`line-height:${a.lineHeight}`} :{}},
    marginTop:{default:null,parseHTML:e=>e.style.marginTop || null,renderHTML:a=>a.marginTop?{style:`margin-top:${a.marginTop}`} :{}},
    marginBottom:{default:null,parseHTML:e=>e.style.marginBottom || null,renderHTML:a=>a.marginBottom?{style:`margin-bottom:${a.marginBottom}`} :{}},
    indent:{default:0,renderHTML:a=>a.indent?{style:`margin-left:${Number(a.indent)*2}em`}:{}},
    backgroundColor:{default:null,parseHTML:e=>e.style.backgroundColor || null,renderHTML:a=>a.backgroundColor?{style:`background-color:${a.backgroundColor}`} :{}},
  }}];},
});
const media = (name:'image'|'audio'|'video') => Node.create({
  name,group:'block',atom:true,draggable:true,selectable:true,
  addAttributes(){return values({id:null,assetId:'',caption:'',alt:'',width:'100%',align:'center',crop:null,title:''});},
  parseHTML(){return [{tag:`figure[data-manual-type="${name}"]`}];},
  renderHTML({HTMLAttributes}){return ['figure',mergeAttributes(HTMLAttributes,{'data-manual-type':name}),['div',{'class':'media-editor-preview'},`${name==='image'?'图片':name==='audio'?'音频':'视频'} · ${HTMLAttributes['data-assetId'] ?? ''}`],['figcaption',{},String(HTMLAttributes['data-caption'] || '选中此块，在右栏修改图注、尺寸与素材')]];},
  addNodeView(){return ({node,editor,getPos})=>{
    const figure=document.createElement('figure');figure.className='editable-media';figure.dataset.manualType=name;
    const paint=()=>{
      figure.replaceChildren(); const id=node.attrs.assetId,registry=(editor.storage as any).manualAssets;
      const asset=registry?.assets?.find((a:any)=>a.id===id),url=asset && registry?.resolve?.(asset);
      if(url){const mediaElement=document.createElement(name==='image'?'img':name);mediaElement.setAttribute('src',url);if(name!=='image'){mediaElement.setAttribute('controls','');mediaElement.setAttribute('preload','metadata');}else mediaElement.setAttribute('alt',node.attrs.alt||asset.alt||'');mediaElement.style.maxWidth='100%';mediaElement.style.width=node.attrs.width||'100%';figure.append(mediaElement);}else{const placeholder=document.createElement('div');placeholder.className='media-editor-preview';placeholder.textContent=`${name} · ${id} · 素材缺失`;figure.append(placeholder);}
      const caption=document.createElement('figcaption');caption.textContent=node.attrs.caption||asset?.caption||'点击选中，在右栏编辑图注';figure.append(caption);figure.style.textAlign=node.attrs.align||'center';
    };paint();figure.addEventListener('click',()=>{const pos=getPos();if(pos!=null)editor.commands.setNodeSelection(pos);});
    return {dom:figure,update(updated){if(updated.type.name!==name)return false;node=updated;paint();return true;},selectNode(){figure.classList.add('ProseMirror-selectednode');},deselectNode(){figure.classList.remove('ProseMirror-selectednode');},ignoreMutation(){return true;}};
  };},
});
export const Formula=Node.create({name:'formula',group:'block',atom:true,draggable:true,addAttributes(){return values({id:null,latex:'',display:true,caption:''});},parseHTML(){return[{tag:'div[data-formula]'}];},renderHTML({node,HTMLAttributes}){return['div',mergeAttributes(HTMLAttributes,{'data-formula':'','class':'formula-editor'}),String(node.attrs.latex||'点击公式块，在右栏编辑 LaTeX')];}});
export const CrossReference=Node.create({name:'crossReference',group:'inline',inline:true,atom:true,addAttributes(){return values({id:null,chapterId:'',targetId:'',label:'交叉引用'});},parseHTML(){return[{tag:'span[data-reference]'}];},renderHTML({node,HTMLAttributes}){return['span',mergeAttributes(HTMLAttributes,{'data-reference':'','class':'reference-editor'}),String(node.attrs.label||node.attrs.targetId)];}});
export const Citation=Node.create({name:'citation',group:'inline',inline:true,atom:true,addAttributes(){return values({id:null,referenceId:'',label:'参考文献'});},parseHTML(){return[{tag:'span[data-citation]'}];},renderHTML({node,HTMLAttributes}){return['span',mergeAttributes(HTMLAttributes,{'data-citation':'','class':'reference-editor'}),String(node.attrs.label)];}});
export const BlockCrossReference=CrossReference.extend({name:'blockCrossReference',group:'block',inline:false,draggable:true,parseHTML(){return[{tag:'div[data-reference]'}];},renderHTML({node,HTMLAttributes}){return['div',mergeAttributes(HTMLAttributes,{'data-reference':'','class':'reference-editor'}),String(node.attrs.label||node.attrs.targetId)];}});
export const BlockCitation=Citation.extend({name:'blockCitation',group:'block',inline:false,draggable:true,parseHTML(){return[{tag:'div[data-citation]'}];},renderHTML({node,HTMLAttributes}){return['div',mergeAttributes(HTMLAttributes,{'data-citation':'','class':'reference-editor'}),String(node.attrs.label)];}});
export const Footnote=Node.create({name:'footnote',group:'block',content:'block+',defining:true,addAttributes(){return values({id:null,label:'注'});},parseHTML(){return[{tag:'aside[data-footnote]'}];},renderHTML({HTMLAttributes}){return['aside',mergeAttributes(HTMLAttributes,{'data-footnote':'','class':'footnote-editor'}),0];}});
export const Columns=Node.create({name:'columns',group:'block',content:'column column',defining:true,draggable:true,addAttributes(){return values({id:null,ratio:'1:1'});},parseHTML(){return[{tag:'div[data-columns]'}];},renderHTML({HTMLAttributes}){return['div',mergeAttributes(HTMLAttributes,{'data-columns':'','class':'columns-editor'}),0];}});
export const Column=Node.create({name:'column',content:'block+',defining:true,addAttributes(){return values({id:null});},parseHTML(){return[{tag:'div[data-column]'}];},renderHTML(){return['div',{'data-column':'','class':'column-editor'},0];}});
export const Admonition=Node.create({name:'admonition',group:'block',content:'block+',defining:true,addAttributes(){return values({id:null,kind:'tip',title:'操作提示'});},parseHTML(){return[{tag:'aside[data-admonition]'}];},renderHTML({HTMLAttributes}){return['aside',mergeAttributes(HTMLAttributes,{'data-admonition':'','class':'admonition-editor'}),0];}});
export const Ipa=Mark.create({name:'ipa',parseHTML(){return[{tag:'span[data-ipa]'}];},renderHTML(){return['span',{'data-ipa':'','style':'font-family: "Doulos SIL", serif'},0];}});
export const ManualAssets=Extension.create({name:'manualAssets',addOptions(){return {assets:[] as unknown[],resolve:(_asset:unknown):string|null=>null};},addStorage(){return {assets:this.options.assets,resolve:this.options.resolve};}});
export const StableAnchors=Extension.create({name:'stableAnchors',addProseMirrorPlugins(){return [new Plugin({appendTransaction(transactions,_oldState,state){if(!transactions.some(t=>t.docChanged))return;const tr=state.tr,seen=new Set<string>();state.doc.descendants((node,pos)=>{if(!('id' in node.attrs))return;const id=node.attrs.id;if(id&&!seen.has(id)){seen.add(id);return;}const next=sourceNodeType(node.type.name)+'-'+crypto.randomUUID();seen.add(next);tr.setNodeMarkup(pos,undefined,{...node.attrs,id:next});});return tr.docChanged?tr:undefined;}})];}});
export const DarkColorAttrs=Extension.create({name:'darkColorAttrs',addGlobalAttributes(){return [{types:['textStyle','highlight'],attributes:values({colorDark:null,backgroundColorDark:null})},{types:['orderedList','bulletList'],attributes:values({id:null})}];}});
export function extensions(assets:unknown[]=[],resolve:(_asset:any)=>string|null=()=>null){return [StarterKit.configure({heading:{levels:[1,2,3,4,5,6]},link:{openOnClick:false}}),TextStyleKit,TextAlign.configure({types:['heading','paragraph']}),TableKit.configure({table:{resizable:true}}),Subscript,Superscript,Highlight.configure({multicolor:true}),EditorialAttributes,DarkColorAttrs,StableAnchors,ManualAssets.configure({assets,resolve}),media('image'),media('audio'),media('video'),Formula,CrossReference,Citation,BlockCrossReference,BlockCitation,Footnote,Columns,Column,Admonition,Ipa];}
export function unknownTypes(body:JSONContent):string[]{
  const schema=getSchema(extensions()),unknown=new Set<string>();
  const inspectAttrs=(attrs:Record<string,any>|undefined,spec:any,label:string)=>{for(const [key,value] of Object.entries(attrs??{}))if(value!=null&&!(key in (spec?.attrs??{})))unknown.add(`${label}.${key}`);};
  const walk=(node:JSONContent)=>{const spec=node.type?schema.nodes[node.type]:undefined;if(!spec||sourceNodeType(node.type??'')!==node.type)unknown.add(node.type??'无类型节点');inspectAttrs(node.attrs,spec,node.type??'node');for(const mark of node.marks??[]){const markSpec=schema.marks[mark.type];if(!markSpec)unknown.add('mark:'+mark.type);inspectAttrs(mark.attrs,markSpec,'mark:'+mark.type);}for(const child of node.content??[])walk(child);};walk(body);
  if(!unknown.size)try{schema.nodeFromJSON(toEditorDocument(body.type==='doc'&&!body.content?.length?{type:'doc',content:[{type:'paragraph'}]}:body)).check();}catch(e){unknown.add('结构校验：'+(e instanceof Error?e.message:String(e)));}
  return [...unknown];
}
