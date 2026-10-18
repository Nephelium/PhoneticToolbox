// An explicit preference in an owned QA profile; no production bypass flag.
import fs from 'node:fs/promises';
import path from 'node:path';
export async function seedPortableProfile(profile){
  const directory=path.resolve(profile,'PhoneticToolbox-cache'),config=path.join(profile,'PhoneticToolbox/v3/portable-cache.json');
  try{const value=JSON.parse(await fs.readFile(config,'utf8'));if(value.schema!=='ptb-portable-cache/1'||path.resolve(value.directory)!==directory)throw Error('Unowned QA cache preference');return directory;}catch(error){if(error.code!=='ENOENT')throw error;}
  await fs.mkdir(profile,{recursive:true});await fs.mkdir(directory,{recursive:false});await fs.writeFile(path.join(directory,'cache-directory.json'),'{"schema":"ptb-portable-cache/1"}',{flag:'wx'});
  await fs.mkdir(path.dirname(config),{recursive:true});await fs.writeFile(config,JSON.stringify({schema:'ptb-portable-cache/1',directory,previousDirectories:[]}),{flag:'wx'});
  return directory;
}
