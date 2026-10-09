import {defineComponent,h,type PropType,type VNodeChild} from 'vue';
import type {ManualNode,ManualTarget} from './types.ts';
import {safeLink,validStableId} from './content.ts';
import {textStyle} from './styles.ts';

export default defineComponent({
  name:'ManualTextNode',
  props:{node:{type:Object as PropType<ManualNode>,required:true},chapterId:{type:String,required:true}},
  emits:{navigate:(_target:ManualTarget)=>true},
  setup(props,{emit}){return ()=>{
    let value:VNodeChild=props.node.text??'';
    for(const mark of props.node.marks??[]){
      const tag=({bold:'strong',italic:'em',underline:'u',strike:'s',code:'code',subscript:'sub',superscript:'sup',ipa:'span'} as Record<string,string>)[mark.type];
      if(tag)value=h(tag,mark.type==='ipa'?{class:'manual-ipa'}:null,[value]);
      else if(mark.type==='textStyle')value=h('span',{class:'manual-custom-color',style:textStyle(mark.attrs)},[value]);
      else if(mark.type==='highlight')value=h('mark',{class:'manual-custom-color',style:textStyle({backgroundColor:mark.attrs?.color,backgroundColorDark:mark.attrs?.colorDark})},[value]);
      else if(mark.type==='link'){
        const href=safeLink(mark.attrs?.href);
        if(href?.startsWith('#')&&validStableId(href.slice(1)))value=h('a',{href:'#'+href.slice(1),onClick:(event:MouseEvent)=>{event.preventDefault();emit('navigate',{chapterId:props.chapterId,targetId:href.slice(1)});}},[value]);
        else if(href)value=h('a',{href,target:/^https?:/i.test(href)?'_blank':undefined,rel:'noopener noreferrer'},[value]);
        else value=h('span',{class:'manual-invalid-link',title:'此链接无效，文字已保留。'},[value]);
      }else value=h('span',{class:'manual-unsupported-mark',title:'尚未支持的文字格式：'+mark.type},[value]);
    }return value;
  };}
});
