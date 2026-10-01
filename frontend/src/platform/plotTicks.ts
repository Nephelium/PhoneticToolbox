// Preserve distinguishable labels for faint amplitudes and microscopic windows.
export function plotTickLabel(value:number,step?:number):string {
 if(!Number.isFinite(value))return '';
 if(value===0)return '0';
 const size=Math.abs(value),spacing=step&&Number.isFinite(step)?Math.abs(step):size/1000;
 if(size<.001||size>=1e7)return value.toExponential(2);
 const digits=Math.max(0,Math.min(9,Math.ceil(-Math.log10(spacing))+1));
 return Number(value.toFixed(digits)).toString();
}
