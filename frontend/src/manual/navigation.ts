import type {ManualChapter,ManualNode,ManualSection} from './types.ts';
import {sectionNumbers} from './content.ts';

export interface ManualTocSection extends ManualSection {number:string;children:ManualTocSection[]}
export interface ManualReadingPath {sectionId?:string;subsectionId?:string}

/** Only the active level-two group reveals its level-three entries in the reader. */
export function manualTocSections(sections:ManualSection[],chapterNumber:number):ManualTocSection[] {
  const numbers=sectionNumbers(sections,chapterNumber),result:ManualTocSection[]=[];
  let parent:ManualTocSection|undefined;
  for(const section of sections){
    const item={...section,number:numbers[section.id],children:[]};
    if(section.level===2){parent=item;result.push(item);}
    else if(section.level===3&&parent)parent.children.push(item);
    else if(section.level<2)parent=undefined;
  }
  return result;
}

/** Search and cross-reference anchors can point at paragraphs, tables or deeper headings. */
export function manualReadingPath(chapter:ManualChapter,targetId?:string):ManualReadingPath {
  if(!targetId)return {};
  let path:ManualReadingPath={},found:ManualReadingPath|undefined;
  const visit=(node:ManualNode)=>{
    if(found)return;
    if(node.type==='heading'){
      const level=Number(node.attrs?.level)||2,id=typeof node.attrs?.id==='string'?node.attrs.id:undefined;
      if(level<2)path={};
      else if(level===2)path={sectionId:id};
      else if(level===3)path={sectionId:path.sectionId,subsectionId:id};
    }
    if(node.attrs?.id===targetId){found={...path};return;}
    for(const child of node.content??[])visit(child);
  };
  visit(chapter.body);
  return found??{};
}

/** Heading tops are relative to the reading viewport, in CSS pixels. */
export function headingAtReadingLine(headings:{id:string;top:number}[],viewport:{viewportHeight:number;scrollTop:number;scrollHeight:number}):string|undefined {
  if(viewport.scrollHeight>viewport.viewportHeight+1&&viewport.scrollTop+viewport.viewportHeight>=viewport.scrollHeight-1)return headings.at(-1)?.id;
  const line=Math.min(80,viewport.viewportHeight*.2);
  let selected:string|undefined;
  for(const heading of headings){if(heading.top<=line)selected=heading.id;else break;}
  return selected;
}
