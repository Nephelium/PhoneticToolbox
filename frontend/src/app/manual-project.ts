import type {ManualProject} from '../manual/types.ts';
import type {HostCapabilities} from '../platform/types.ts';

export const webIntroductionChapter='web-introduction';

export function manualIntroduction(kind:HostCapabilities['kind']):string {
  return kind==='browser'?webIntroductionChapter:'getting-started';
}

/** Both hosts read the same chapters; only the browser adds its introduction. */
export function manualForHost(project:ManualProject,kind:HostCapabilities['kind']):ManualProject {
  if(kind==='browser')return project;
  return {...project,chapters:project.chapters.filter(chapter=>chapter.id!==webIntroductionChapter),
    ...(project.searchIndex?{searchIndex:project.searchIndex.filter(entry=>entry.chapterId!==webIntroductionChapter)}:{})};
}
