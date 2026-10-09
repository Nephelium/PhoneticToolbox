// Geometric area and the native corrected acoustic tube are distinct readouts.
export function areaReadout(state,index=state.section){
  const section=state.airway_sections[index],position=state.centerline[index][2];
  let end=0,acoustic=state.tube_areas.at(-1);
  for(let i=0;i<state.tube_lengths.length;i++){
    end+=state.tube_lengths[i];if(position<=end+1e-8){acoustic=state.tube_areas[i];break;}
  }
  const gap=section.upper.map((v,i)=>v===null||section.lower[i]===null?0:Math.max(0,v-section.lower[i]));
  const geometric=section.area??gap.slice(1).reduce((sum,v,i)=>sum+(v+gap[i])*.5*7/96,0);
  return {position,geometric,acoustic,closed:!gap.some(v=>v>1e-6),midlineClosed:gap[48]<=1e-6};
}
