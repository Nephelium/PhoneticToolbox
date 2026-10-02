import type {ParameterTable,ResearchFile,ResearchFiles} from '../../platform/research.ts';
import {lipTrack,portableAnnotation} from '../../platform/annotation.ts';

/** Reuse the M12 inert reader and audio-relative time contract, including V2 companions. */
export async function lipParameterTable(files:ResearchFiles,file:ResearchFile):Promise<ParameterTable>{
 const read=await (files.annotation??portableAnnotation(files)).lip(file),track=lipTrack(read.wire);
 const vectors=[track.area,track.width,track.open,track.circularity];
 return {schema_version:'m02/1',sha256:read.sha256,columns:['Time_s','LipArea','LipWidth','LipOpen','LipCirc'],
  kinds:['number','number','number','number','number'],rows:track.times.map((time,i)=>[time+track.offset,...vectors.map(v=>v[i]??null)])};
}
