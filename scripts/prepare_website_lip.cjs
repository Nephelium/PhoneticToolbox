// Burn a fully opaque decorative privacy sticker into the user-selected screenshot.
// The original is never copied to the static site; all pixels outside ROI stay exact.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{createHash}=require('node:crypto');
const sharp=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'));
async function main(){
 const input=process.argv[2];assert(input,'Provide the exact user-selected screenshot path');
 const root=path.resolve(__dirname,'..'),assets=path.join(root,'temporary-site/assets');
 const source=await fs.readFile(input),meta=await sharp(source).metadata();assert.equal(meta.width,2559);assert.equal(meta.height,1527);
 const roi={left:1891,top:333,width:601,height:340};
 const logo=(await fs.readFile(path.join(assets,'brand.png'))).toString('base64');
 const sticker=Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="601" height="340" viewBox="0 0 601 340">
 <rect width="601" height="340" fill="#e2dcf5"/><path d="M0 58Q110 94 226 37T601 58V340H0Z" fill="#fff9e9"/>
 <path d="M0 285Q140 225 298 284T601 280V340H0Z" fill="#ffea78"/>
 <g transform="translate(195 63) rotate(-7 105 105)"><rect x="4" y="10" width="210" height="210" rx="46" fill="#685482" opacity=".12"/><image href="data:image/png;base64,${logo}" width="210" height="210"/></g>
 <g fill="#ad99db"><path d="m115 87 9 25 26 10-26 9-9 26-9-26-26-9 26-10z"/><path d="m465 189 7 18 20 8-20 7-7 20-8-20-18-7 18-8z"/></g>
 <g fill="none" stroke="#d4c7ed" stroke-width="9" stroke-linecap="round"><path d="m447 81 8-14m17 30 16-3M117 233l-16 5"/></g>
 <circle cx="460" cy="260" r="6" fill="#f79e90"/><circle cx="157" cy="51" r="5" fill="#f79e90"/>
 <g transform="rotate(-4 301 301)"><rect x="233" y="277" width="142" height="53" rx="24" fill="#604265" opacity=".15"/><rect x="229" y="272" width="142" height="53" rx="24" fill="#ffb6a7" stroke="#fffaf0" stroke-width="4"/><text x="300" y="312" text-anchor="middle" font-family="Microsoft YaHei" font-size="39" font-weight="900" fill="#49314d" stroke="#49314d" stroke-width="1" stroke-linejoin="round">保密</text></g>
 </svg>`);
 const overlay=await sharp(sticker).ensureAlpha().raw().toBuffer();assert.equal(overlay.length,roi.width*roi.height*4);
 for(let i=3;i<overlay.length;i+=4)assert.equal(overlay[i],255,'Every sticker pixel must be opaque');
 const result=await sharp(source).composite([{input:sticker,left:roi.left,top:roi.top}]).png().toBuffer();
 const before=await sharp(source).ensureAlpha().raw().toBuffer(),after=await sharp(result).ensureAlpha().raw().toBuffer();
 for(let y=0;y<meta.height;y++)for(let x=0;x<meta.width;x++){const i=(y*meta.width+x)*4;if(x>=roi.left&&x<roi.left+roi.width&&y>=roi.top&&y<roi.top+roi.height){const j=((y-roi.top)*roi.width+x-roi.left)*4;assert(after.subarray(i,i+4).equals(overlay.subarray(j,j+4)));}else assert(before.subarray(i,i+4).equals(after.subarray(i,i+4)));}
 const file='lip-nord-dark.png';await fs.writeFile(path.join(assets,'screens',file),result);
 const record={file,module:'M05',palette:'nord',mode:'dark',width:meta.width,height:meta.height,source:'User-selected complete screenshot; only video rectangle replaced with opaque decorative sticker',source_sha256:createHash('sha256').update(source).digest('hex'),sha256:createHash('sha256').update(result).digest('hex'),privacy:{rectangle:roi,all_pixels_opaque:true,outside_rectangle_pixel_identical:true,original_in_public_assets:false}};
 const p=path.join(assets,'screens/provenance.json'),data=JSON.parse(await fs.readFile(p,'utf8'));data.captures=data.captures.filter(x=>x.file!==file);data.captures.push(record);await fs.writeFile(p,JSON.stringify(data,null,2)+'\n');console.log(JSON.stringify(record));
}
main().catch(e=>{console.error(e);process.exitCode=1});
