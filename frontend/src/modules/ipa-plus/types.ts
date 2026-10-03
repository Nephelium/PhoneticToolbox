export type ChartSystem = 'ipa' | 'extipa' | 'voqs';
export type InsertionMode = 'literal' | 'combining' | 'paired-span' | 'bridge';
export interface SymbolEntry {
  id: string; system: ChartSystem; section: string; display: string; insertText: string;
  insertionMode: InsertionMode; prefix?: string; suffix?: string; codePoints: string[];
  nameZh: string; nameEn: string; descriptionZh: string; usageZh: string; contrastZh: string;
  aliases: string[]; examples: {text: string; noteZh: string}[];
  sourceRefs: {sourceId: string; locator: string}[];
  isExample: boolean; representation?: string;
  notationStatus?: 'combination' | 'historical';
}
export interface ChartCell { ids: string[]; shaded?: boolean; rightHalfShaded?:boolean; span?: number; }
export interface ChartRow { label: string; cells: ChartCell[]; }
export interface ChartSection {
  id: string; title: string; subtitle: string; kind: 'matrix' | 'vowels' | 'list';
  columns?: string[]; rows?: ChartRow[]; ids: string[];
  points?: {ids: string[]; x: number; y: number}[];
  groups?: {label: string; hint: string; ids: string[]}[];
}
export interface Catalog { version: string; entries: SymbolEntry[]; charts: Record<ChartSystem, ChartSection[]>; }
export interface EditorSnapshot { text: string; start: number; end: number; }
export interface Draft extends EditorSnapshot {
  version: 1; catalogVersion: string; system: ChartSystem; introductions: boolean;
  textSize: number; textSizeVersion?: 1; editorHeight: number; revision: number; writer: string;
}
