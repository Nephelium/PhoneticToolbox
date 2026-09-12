// Read dimensions before asking the renderer to decode a user-supplied image.
// The scientific child independently decodes and bounds the actual image again.
export function imageDimensions(raw:ArrayBuffer):{width:number;height:number} {
 const b=new Uint8Array(raw),v=new DataView(raw);let width=0,height=0;
 if(b.length>=24&&[137,80,78,71,13,10,26,10].every((n,i)=>b[i]===n)&&v.getUint32(12)===0x49484452){width=v.getUint32(16);height=v.getUint32(20);}
 else if(b.length>=26&&b[0]===66&&b[1]===77){const dib=v.getUint32(14,true);if(dib===12){width=v.getUint16(18,true);height=v.getUint16(20,true);}else if(dib>=40&&b.length>=54){width=v.getInt32(18,true);height=Math.abs(v.getInt32(22,true));}}
 else if(b.length>=4&&b[0]===255&&b[1]===216){let offset=2;while(offset+4<=b.length){if(b[offset++]!==255)break;while(b[offset]===255)offset++;const marker=b[offset++];if(marker===217||marker===218)break;if(marker===1||(marker>=208&&marker<=215))continue;const length=v.getUint16(offset);if(length<2||offset+length>b.length)break;if([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker)&&length>=8){height=v.getUint16(offset+3);width=v.getUint16(offset+5);break;}offset+=length;}}
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<2||height<2||width*height>25_000_000)throw Error('图片尺寸无效或超过 2500 万像素。请先裁去无关区域。');
 return {width,height};
}
