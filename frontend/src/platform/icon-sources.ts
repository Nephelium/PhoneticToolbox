/** Original imported images stay in the local profile, outside native icon payloads. */
import {openImageDatabase} from './image-storage.ts';
async function sourceStore(id:string,blob?:Blob|null):Promise<Blob|undefined>{
 const db=await openImageDatabase('ptb-icon-sources','images');
 try{return await new Promise((resolve,reject)=>{const transaction=db.transaction('images',blob===undefined?'readonly':'readwrite'),store=transaction.objectStore('images'),request=blob===undefined?store.get(id):blob===null?store.delete(id):store.put(blob,id);transaction.oncomplete=()=>resolve(blob===undefined?request.result:undefined);transaction.onabort=()=>reject(transaction.error);transaction.onerror=()=>reject(transaction.error);});}finally{db.close();}
}
export const saveIconSource=(id:string,blob:Blob)=>sourceStore(id,blob);
export const removeIconSource=(id:string)=>sourceStore(id,null);
export async function originalIconSource(id:string,dataUrl:string){
 try{const source=await sourceStore(id);if(source instanceof Blob)return source;}catch{/* Existing icons can still be cropped from their saved image. */}
 const binary=atob(dataUrl.split(',')[1]!),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));return new Blob([bytes],{type:'image/png'});
}
