import type {components} from '../../../../contracts/generated/api';
import type {JobView} from '../../platform/research.ts';
export type CorpusItem=components['schemas']['M11CorpusItem'];
export type MfaConfig=components['schemas']['M11Config'];
export type Request=components['schemas']['M11Request'];
export interface Grant {id:string;label:string;purpose:string}
export interface Catalog {
 schema_version:string;download_available:boolean;download_reason:string;execution_available:boolean;waiting_reason:string|null;
 runtimes:{id:string;version:string;platform:string;arch:string;validated:boolean;fingerprint:string;versions?:Record<string,string>;installed_bytes?:number;download_bytes?:number|null;source?:string;archive_sha256?:string}[];
 models:{id:string;name:string;model_sha256:string;dictionary_sha256:string;validated_runtime:string;model_bytes?:number;dictionary_bytes?:number}[];
}
export interface M11Port {
 local:boolean;
 log?(id:string):Promise<{text:string;truncated:boolean}>;
 catalog():Promise<Catalog>;
 history():Promise<JobView[]>;
 import(files:File[],signal:AbortSignal):Promise<CorpusItem[]>;
 dictionary(file:File,signal:AbortSignal):Promise<{asset_id:string;sha256:string}>;
 create(body:Omit<Request,'project_id'|'idempotency_key'|'schema_version'>):Promise<JobView>;
 job(id:string):Promise<JobView>;
 cancel(id:string):Promise<unknown>;
 events(id:string):Promise<{events:{sequence:number;code:string;created_at:number}[]}>;
 download(job:JobView,id:string):Promise<void>;
 save?(job:JobView,grant:Grant):Promise<{count:number;directory:string}>;
 chooseOutput?():Promise<Grant|null>;
 pick?(purpose:string):Promise<Grant|null>;
 corpus?(grant:Grant):Promise<CorpusItem[]>;
 component?(body:Record<string,string>):Promise<unknown>;
}
