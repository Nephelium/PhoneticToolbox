import {tr} from '../../i18n/index.ts';
export interface MapAttribution {id:string;label:string;shortLabel?:string;url:string;license?:string;details?:string}

/** Credits follow the data actually loaded, including older local data sets. */
export function mapAttributions(admin:readonly {source:string;supplementalSources?:string[];secondarySources?:string[]}[]):MapAttribution[]{
 const sources=new Set(admin.map(row=>row.source));
 const supplemental=new Set(admin.flatMap(row=>[...(row.supplementalSources??[]),...(row.secondarySources??[])]));
 const datav=[...sources].some(source=>source.startsWith('datav-'));
 const result:MapAttribution[]=[];
 if(datav)result.push({id:'datav',label:tr("modules.geolinguistics.datav_geoatlas_amap_open_platform"),url:'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/',license:tr("modules.geolinguistics.for_learning_and_exchange_only"),details:tr("modules.geolinguistics.areas_v3_page_version_2021_5")});
 if([...sources].some(source=>source.startsWith('osm-')))result.push({id:'osm',label:'© OpenStreetMap contributors',url:'https://www.openstreetmap.org/copyright',license:'ODbL 1.0',details:'Geofabrik，2026-10-22'});
 if([...sources].some(source=>source.startsWith('nlsc')))result.push({id:'nlsc',label:tr("modules.geolinguistics.national_land_surveying_and_mapping_center_ministry_of_the"),shortLabel:tr("modules.geolinguistics.national_land_surveying_and_mapping_center"),url:'https://data.gov.tw/dataset/7441',license:tr("modules.geolinguistics.open_government_data_license_version_1"),details:tr("modules.geolinguistics.counties_cities_2020_townships_2023_including_supplementary_majia_sanhe")});
 else if([...supplemental].some(source=>source.startsWith('nlsc')))result.push({id:'nlsc',label:tr("modules.geolinguistics.national_land_surveying_and_mapping_center_ministry_of_the"),shortLabel:tr("modules.geolinguistics.national_land_surveying_and_mapping_center"),url:'https://data.gov.tw/dataset/7441',license:tr("modules.geolinguistics.open_government_data_license_version_1"),details:tr("modules.geolinguistics.taiping_island_land_surface_town_moi_1120317_separately_linked")});
 if(sources.has('hk-had-official'))result.push({id:'hk-had',label:tr("modules.geolinguistics.hong_kong_sar_government_home_affairs_department_data_gov"),shortLabel:tr("modules.geolinguistics.hong_kong_government"),url:'https://data.gov.hk/en-data/dataset/hk-had-json1-hong-kong-administrative-boundaries',license:tr("modules.geolinguistics.data_gov_hk_terms_and_conditions"),details:tr("modules.geolinguistics.administrative_boundaries_2026_terms_1_2_2025_05_26")});
 if([...sources].some(source=>source.startsWith('gary-')))result.push({id:'gary',label:'GaryBikini / ChinaAdminDivisonSHP',url:'https://github.com/GaryBikini/ChinaAdminDivisonSHP'});
 if(!datav)result.push({id:'territory',label:tr("modules.geolinguistics.huang_et_al_2024_outline"),shortLabel:tr("modules.geolinguistics.huang_et_al"),url:'https://doi.org/10.5281/zenodo.10697975',license:'CC BY 4.0'});
 return result;
}

/** Text is included visibly in both vector and raster exports. */
export function mapAttributionText(items:readonly MapAttribution[]):string{
 return tr("modules.geolinguistics.basemap")+items.map(item=>item.label+(item.license?'（'+item.license+'）':'')+' '+item.url+(item.details?' · '+item.details:'')).join('；');
}
