import {tr} from './index.ts';
const labels:Record<string,string>={LipArea:'parameters.lipArea',LipWidth:'parameters.lipWidth',LipOpen:'parameters.lipAperture',LipCirc:'parameters.lipCircularity'};
/** Presentation only: original parameter keys, column names and data stay intact. */
export function parameterLabel(key:string,fallback:string):string{return labels[key]?tr(labels[key]!):fallback;}
