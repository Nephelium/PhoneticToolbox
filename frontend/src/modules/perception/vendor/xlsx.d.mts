export const version:string;
export const utils:{aoa_to_sheet:(rows:unknown[][])=>any;sheet_to_json:(sheet:any,options:any)=>any[];book_new:()=>any;book_append_sheet:(book:any,sheet:any,name:string)=>void;decode_range:(range:string)=>{s:{r:number;c:number};e:{r:number;c:number}}};
export function read(data:ArrayBuffer|Uint8Array,options:Record<string,unknown>):any;
export function write(book:any,options:Record<string,unknown>):ArrayBuffer;
