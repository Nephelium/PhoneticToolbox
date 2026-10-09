<script setup lang="ts">
import {computed} from 'vue';
import type {AssetResolver,ManualAsset,ManualNode,ManualReference,ManualTarget} from './types.ts';
import {manualAnchor,safeLink,validStableId,nodeText,unnumberedCaption} from './content.ts';
import {boundedSpan,contentStyle,textStyle} from './styles.ts';
import TextNode from './TextNode.ts';
import ManualMedia from './ManualMedia.vue';
import ManualFormula from './ManualFormula.vue';
defineOptions({name:'ManualNode'});
const props=withDefaults(defineProps<{node:ManualNode;chapterId:string;numbering?:Record<string,string>;assets:ManualAsset[];references?:ManualReference[];assetBaseUrl?:string;assetResolver?:AssetResolver;playbackAllowed?:boolean}>(),{numbering:()=>({}),references:()=>[],assetBaseUrl:'./manual/',playbackAllowed:true});
const emit=defineEmits<{navigate:[target:ManualTarget]}>();
const attrs=computed(()=>props.node.attrs??{});
const id=computed(()=>validStableId(attrs.value.id)?manualAnchor(props.chapterId,attrs.value.id):undefined);
const tags:Record<string,string>={doc:'div',paragraph:'p',bulletList:'ul',orderedList:'ol',listItem:'li',blockquote:'blockquote',tableRow:'tr',tableCell:'td',tableHeader:'th',column:'div',columns:'div',admonition:'aside',footnote:'aside'};
const tag=computed(()=>props.node.type==='heading'?'h'+Math.max(1,Math.min(6,Number(attrs.value.level)||2)):tags[props.node.type]);
const reference=computed(()=>props.references.find(item=>item.id===attrs.value.referenceId));
const referenceUrl=computed(()=>safeLink(reference.value?.url));
const target=computed(()=>({chapterId:validStableId(attrs.value.chapterId)?attrs.value.chapterId:props.chapterId,targetId:validStableId(attrs.value.targetId)?attrs.value.targetId:undefined}));
const admonitionTitle=computed(()=>({tip:'操作提示',condition:'使用条件',result:'结果说明',limitation:'适用范围'} as Record<string,string>)[String(attrs.value.kind)]||'说明');
const displayNumber=computed(()=>validStableId(attrs.value.id)?props.numbering[attrs.value.id]:undefined);
const caption=computed(()=>typeof attrs.value.caption==='string'?unnumberedCaption(attrs.value.caption):undefined);
</script>
<template>
  <TextNode v-if="node.type==='text'" :node="node" :chapter-id="chapterId" @navigate="emit('navigate',$event)"/>
  <br v-else-if="node.type==='hardBreak'"/>
  <hr v-else-if="node.type==='horizontalRule'" :id="id"/>
  <span v-else-if="node.type==='crossReference'" :id="id" class="manual-cross-reference"><button v-if="target.targetId||validStableId(attrs.chapterId)" type="button" @click="emit('navigate',target)">{{attrs.label||'查看相关内容'}}</button><span v-else class="manual-invalid-link">{{attrs.label||'交叉引用目标缺失'}}</span></span>
  <span v-else-if="node.type==='citation'" :id="id" class="manual-citation"><a v-if="referenceUrl" :href="referenceUrl" target="_blank" rel="noopener noreferrer">{{attrs.label||reference?.label||'参考文献'}}</a><span v-else :title="reference?'参考文献未提供网址':'参考文献标识未找到'">{{attrs.label||reference?.label||'参考文献未找到'}}</span></span>
  <ManualMedia v-else-if="node.type==='image'||node.type==='audio'||node.type==='video'" :id="id" :kind="node.type" :asset-id="String(attrs.assetId||'')" :assets="assets" :asset-base-url="assetBaseUrl" :asset-resolver="assetResolver" :caption="caption" :display-number="displayNumber" :alt="typeof attrs.alt==='string'?attrs.alt:undefined" :width="attrs.width" :align="attrs.align" :playback-allowed="playbackAllowed"/>
  <ManualFormula v-else-if="node.type==='formula'" :id="id" :latex="attrs.latex||nodeText(node)" :display="attrs.display" :caption="attrs.caption"/>
  <pre v-else-if="node.type==='codeBlock'" :id="id" class="manual-code" :data-language="attrs.language"><code>{{nodeText(node)}}</code></pre>
  <figure v-else-if="node.type==='table'" :id="id" class="manual-table-figure"><figcaption v-if="caption"><span v-if="displayNumber">表 {{displayNumber}}：</span>{{caption}}</figcaption><div class="manual-table-scroll" tabindex="0" :aria-label="String(attrs.caption||'说明书表格')"><table><tbody><ManualNode v-for="(child,index) in node.content" :key="index" :node="child" :chapter-id="chapterId" :numbering="numbering" :assets="assets" :references="references" :asset-base-url="assetBaseUrl" :asset-resolver="assetResolver" :playback-allowed="playbackAllowed" @navigate="emit('navigate',$event)"/></tbody></table></div><p v-if="attrs.note" class="manual-table-note">{{attrs.note}}</p></figure>
  <component :is="tag" v-else-if="tag" :id="id" :class="['manual-node','manual-'+node.type,{'manual-custom-color':node.type==='tableCell'||node.type==='tableHeader'}]" :style="{...contentStyle(attrs),...(node.type==='tableCell'||node.type==='tableHeader'?textStyle(attrs):{})}" :colspan="node.type==='tableCell'||node.type==='tableHeader'?boundedSpan(attrs.colspan):undefined" :rowspan="node.type==='tableCell'||node.type==='tableHeader'?boundedSpan(attrs.rowspan):undefined" :start="node.type==='orderedList'?boundedSpan(attrs.start):undefined">
    <span v-if="node.type==='heading'&&displayNumber" class="manual-section-number">{{displayNumber}} </span>
    <strong v-if="node.type==='admonition'" class="manual-admonition-title">{{admonitionTitle}}</strong>
    <small v-if="node.type==='footnote'">注 {{attrs.label||attrs.id||''}}</small>
    <ManualNode v-for="(child,index) in node.content" :key="index" :node="child" :chapter-id="chapterId" :numbering="numbering" :assets="assets" :references="references" :asset-base-url="assetBaseUrl" :asset-resolver="assetResolver" :playback-allowed="playbackAllowed" @navigate="emit('navigate',$event)"/>
  </component>
  <aside v-else :id="id" class="manual-unsupported" role="status"><strong>当前阅读器尚未支持 {{node.type}} 内容。</strong><p v-if="nodeText(node)">{{nodeText(node)}}</p><span>原始节点保留在说明书工程中，请在作者工具中修订。</span></aside>
</template>
