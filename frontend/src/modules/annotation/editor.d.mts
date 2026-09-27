export interface Interval {xmin:number;xmax:number;text:string}
export interface Point {number:number;mark:string}
export interface AnnotationTier {name:string;intervals?:Interval[];points?:Point[];xmin?:number;xmax?:number}
export interface Grid {xmin:number;xmax:number;tiers:AnnotationTier[]}
export interface IntervalTier extends AnnotationTier {intervals:Interval[]}
export interface Hit {tier:string;index:number;edge:'start'|'end'|null;time:number}
export interface Selection {tier:string;index:number}
export interface EditorState {
 audioBuffer:{duration:number;sampleRate:number;getChannelData:(index:number)=>Float32Array}|null;
 textgrid:Grid|null;visibleStart:number;visibleDuration:number;selected:Selection|null;
 selectedBoundary:{tier:string;time:number}|null;selectedIndices:number[];
 drag:Record<string,any>|null;lastMouseTime:number;dirty:boolean;undoStack:{tiers:AnnotationTier[];sequenceIndex:number;sequenceStart:number|null;labSequence:string[]}[];
 sequenceIndex:number;sequenceStart:number|null;
 referenceTextGrid:Grid|null;phoneDict:Map<string,string[]>|null;searchResults:number[];searchIndex:number;
 labSequence:string[];labWords:Set<string>;copiedWord:string;copiedLabIndex:number|null;wordTierName:string;phoneTierName:string;
}
export interface Editor {
 state:EditorState;controls:Record<string,{value:string}>;setCanvas(canvas:HTMLCanvasElement):void;
 normalizeTextGrid(grid:Grid):void;saveUndoState():void;undo():void;editText(text:string,manualLabels?:string[]):void;clearText():void;copy():void;
 copyAnnotation(cut?:boolean):void;deleteAnnotation():void;pasteAnnotation(time?:number):void;
 tierByName(name:string):IntervalTier|null;wordTier():IntervalTier|null;phoneTier():IntervalTier|null;
 fillGaps(intervals:Interval[],xmax:number):Interval[];parseDictText(text:string):Map<string,string[]>;
 onGridMouseDown(event:MouseEvent,forceRange?:boolean):void;onGridMouseMove(event:MouseEvent):void;onGridMouseUp():void;hitTest(event:MouseEvent):Hit|null;moveSelected(delta:number):void;
 moveBoundary(hit:Hit,time:number):void;deleteSelectedBoundary():void;splitPhoneAt(time:number):void;
 boundaryAt(time:number,tolerance:number):Hit|null;beginBoundaryDrag(hit:Hit,event:MouseEvent):void;dragBoundaryTo(time:number,clientX:number):void;
 autoPhonesForSelection():void;ensurePhonesForWord(word:Interval):void;pasteCopiedWord():void;
 doSearch(query:string):void;findNext():void;findPrev():void;replaceCurrent(text:string):void;replaceAll(text:string):void;
 applyReferenceSplice():Promise<void>;fitIntensityRange():void;localIntensityEnvelope(start:number,end:number):{times:number[];db:number[];hopSec:number}|null;
 pinyinToPhones(label:string):string[];replacementWindows(mode:string,start:number,end:number,duration:number):number[][];
 spliceTier(current:IntervalTier,reference:IntervalTier|null,windows:number[][],duration:number):IntervalTier;
}
export function createEditor(options?:{changed?:()=>void;message?:(message:string)=>void}):Editor;
