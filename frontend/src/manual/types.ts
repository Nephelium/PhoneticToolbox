/** Versioned, editor-independent reading contract. Node JSON follows Tiptap. */
export const MANUAL_SCHEMA = 'ptb-manual/1' as const;
export const CHAPTER_SCHEMA = 'ptb-manual-chapter/1' as const;

export interface ManualMark {type:string; attrs?:Record<string,unknown>}
export interface ManualNode {
  type:string;
  attrs?:Record<string,unknown>;
  text?:string;
  marks?:ManualMark[];
  content?:ManualNode[];
}
export interface ManualSection {id:string; title:string; level:number}
export interface ManualChapterDescriptor {
  id:string; title:string; path:string; moduleId?:string;
  status?:string; summary?:string; sections?:ManualSection[];
}
export interface ManualAsset {
  id:string; path:string; kind:'image'|'audio'|'video'|'example';
  distribution:'public'|'software-only';
  mime?:string; sha256?:string; caption?:string; alt?:string; title?:string;
  source?:string|Record<string,unknown>; sourceType?:string; git?:boolean;
  width?:number; height?:number; duration?:number; sampleRate?:number; channels?:number;
}
export interface ManualReference {id:string; label:string; url?:string; [key:string]:unknown}
export interface ManualTarget {chapterId:string; targetId?:string}
export interface ManualLocation extends ManualTarget {scrollTop:number}
export interface ManualSearchEntry extends ManualTarget {text:string; title?:string}
export interface ManualSearchHit extends ManualSearchEntry {excerpt:string}
export interface ManualProject {
  schemaVersion:typeof MANUAL_SCHEMA; id:string; title:string; language?:string;
  version?:string; softwareVersion?:string;
  chapters:ManualChapterDescriptor[]; assets:ManualAsset[];
  references?:ManualReference[]; searchIndex?:ManualSearchEntry[];
}
export interface ManualChapter {
  schemaVersion:typeof CHAPTER_SCHEMA; id:string; title:string;
  moduleId?:string; summary?:string; body:ManualNode;
}
export type ChapterLoader = (chapter:ManualChapterDescriptor, signal:AbortSignal) => Promise<ManualChapter>;
export type AssetResolver = (asset:ManualAsset) => string|null;
