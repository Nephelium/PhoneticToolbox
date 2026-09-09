import { readFile, writeFile, mkdir } from 'node:fs/promises';
import openapiTS, { astToString } from 'openapi-typescript';

const schema = new URL('../../contracts/openapi.json', import.meta.url);
const output = new URL('../../contracts/generated/api.ts', import.meta.url);
const content = '/** Generated from contracts/openapi.json. Do not edit. */\n' + astToString(await openapiTS(schema));
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== content) throw new Error('Generated TypeScript drift');
  console.log('Generated TypeScript matches the OpenAPI snapshot.');
} else {
  await mkdir(new URL('../../contracts/generated/', import.meta.url), { recursive: true });
  await writeFile(output, content);
}
