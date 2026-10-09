import {parseManualChapter} from './content.ts';
import type {ChapterLoader,ManualChapter,ManualChapterDescriptor} from './types.ts';

/** Bounded cache contains JSON only. Media elements exist only in the open chapter. */
export class ManualChapterRepository {
  private cache=new Map<string,ManualChapter>();
  private loader:ChapterLoader;
  private capacity:number;
  constructor(loader:ChapterLoader,capacity=3){this.loader=loader;this.capacity=capacity;}
  clear():void {this.cache.clear();}
  get(id:string):ManualChapter|undefined {return this.cache.get(id);}
  values():ManualChapter[] {return [...this.cache.values()];}
  async load(descriptor:ManualChapterDescriptor,signal:AbortSignal):Promise<ManualChapter> {
    if(signal.aborted)throw new DOMException('章节请求已取消。','AbortError');
    const prior=this.cache.get(descriptor.id);
    if(prior){this.cache.delete(descriptor.id);this.cache.set(descriptor.id,prior);return prior;}
    const chapter=parseManualChapter(await this.loader(descriptor,signal),descriptor.id);
    if(signal.aborted)throw new DOMException('章节请求已取消。','AbortError');
    this.cache.set(descriptor.id,chapter);
    while(this.cache.size>Math.max(1,this.capacity))this.cache.delete(this.cache.keys().next().value!);
    return chapter;
  }
}
