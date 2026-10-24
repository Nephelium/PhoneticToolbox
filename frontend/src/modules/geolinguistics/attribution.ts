export interface MapAttribution {id:string;label:string;shortLabel?:string;url:string;license?:string;details?:string}

/** Credits follow the data actually loaded, including older local data sets. */
export function mapAttributions(admin:readonly {source:string;supplementalSources?:string[];secondarySources?:string[]}[]):MapAttribution[]{
 const sources=new Set(admin.map(row=>row.source));
 const supplemental=new Set(admin.flatMap(row=>[...(row.supplementalSources??[]),...(row.secondarySources??[])]));
 const datav=[...sources].some(source=>source.startsWith('datav-'));
 const result:MapAttribution[]=[];
 if(datav)result.push({id:'datav',label:'DataV GeoAtlas / 高德开放平台',url:'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/',license:'仅供学习和交流',details:'areas_v3；页面版本标注 2021.5'});
 if([...sources].some(source=>source.startsWith('osm-')))result.push({id:'osm',label:'© OpenStreetMap contributors',url:'https://www.openstreetmap.org/copyright',license:'ODbL 1.0',details:'Geofabrik，2026-10-22'});
 if([...sources].some(source=>source.startsWith('nlsc')))result.push({id:'nlsc',label:'内政部国土测绘中心',shortLabel:'国土测绘中心',url:'https://data.gov.tw/dataset/7441',license:'政府资料开放授权条款第1版',details:'县市 2020、乡镇 2023，含玛家三和补充面'});
 else if([...supplemental].some(source=>source.startsWith('nlsc')))result.push({id:'nlsc',label:'内政部国土测绘中心',shortLabel:'国土测绘中心',url:'https://data.gov.tw/dataset/7441',license:'政府资料开放授权条款第1版',details:'太平岛实际岛面；TOWN_MOI_1120317，独立显示关联'});
 if(sources.has('hk-had-official'))result.push({id:'hk-had',label:'© 香港特别行政区政府 · 民政事务总署 / DATA.GOV.HK',shortLabel:'© 香港政府',url:'https://data.gov.hk/en-data/dataset/hk-had-json1-hong-kong-administrative-boundaries',license:'DATA.GOV.HK 使用条款',details:'行政区边界 2026，使用条款 1.2（2025-05-26）'});
 if([...sources].some(source=>source.startsWith('gary-')))result.push({id:'gary',label:'GaryBikini / ChinaAdminDivisonSHP',url:'https://github.com/GaryBikini/ChinaAdminDivisonSHP'});
 if(!datav)result.push({id:'territory',label:'Huang 等（2024）轮廓',shortLabel:'Huang 等',url:'https://doi.org/10.5281/zenodo.10697975',license:'CC BY 4.0'});
 result.push({id:'world',label:'Natural Earth',url:'https://www.naturalearthdata.com/about/terms-of-use/',license:'Public domain'});
 return result;
}

/** Text is included visibly in both vector and raster exports. */
export function mapAttributionText(items:readonly MapAttribution[]):string{
 return '底图：'+items.map(item=>item.label+(item.license?'（'+item.license+'）':'')+' '+item.url+(item.details?' · '+item.details:'')).join('；');
}
