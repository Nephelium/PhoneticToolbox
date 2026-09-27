import type {FigureFontSnapshot} from '../../platform/research.ts';
export interface Row {character:string;ipa:string;note:string;initial:string;final:string;tone_value:string}
export interface Settings {tone_map:Record<string,string>;tone_order:string[];initial_order:string[];final_order:string[];initial_map:Record<string,string>;final_map:Record<string,string>}
export interface Options {skip_first_row:boolean;consonant_only_as_zero_initial:boolean}
export interface Source {asset_id:string;sha256:string;name:string}
export interface Preview {schema_version:'m14/1';input_sha256:string;analysis:{rows:Row[];unique_initials:string[];unique_finals:string[];unique_tones:string[];unique_ipa:string[]};diagnostics:{total_rows:number;accepted_rows:number;skipped:{row:number;reason:string}[];duplicate_rows:number};config:Settings;single_consonants:{character:string;ipa:string}[]}
export interface Result {job:string;files:{id:string;name:string;sha256:string;size_bytes:number}[]}
export interface M14Port {
 import(file:File,options:Options,signal:AbortSignal):Promise<{source:Source;preview:Preview}>;
 generate(source:Source,options:Options,settings:Settings,font:FigureFontSnapshot,signal:AbortSignal):Promise<Result>;
 save(result:Result):Promise<boolean>;
 download(result:Result,id:string):Promise<void>;
}
