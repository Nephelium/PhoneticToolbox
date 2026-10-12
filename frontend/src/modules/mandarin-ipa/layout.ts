import {effectiveIpaSize,type MandarinIpaDraft} from './state.ts';

// One unit is an IPA/Hanzi pair, or just IPA in the IPA-only display.
export function resultMetrics(draft:MandarinIpaDraft){
  const ipaSize=effectiveIpaSize(draft),ipaHeight=ipaSize*1.59;
  const unitHeight=ipaHeight+(draft.display==='paired'?draft.gap+draft.hanziSize*1.2:0)+4;
  const rowGap=Math.max(0,draft.lineHeight-.8)*(draft.display==='paired'?Math.max(ipaSize,draft.hanziSize):ipaSize);
  return {ipaSize,ipaHeight,unitHeight,rowGap,lineAdvance:unitHeight+rowGap,columnGap:4};
}
