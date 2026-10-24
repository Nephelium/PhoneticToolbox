import type {Admin,Resources} from '../../src/modules/geolinguistics/model.ts';

// Read-only sample from GeoAtlas all.json, source SHA256:
// bc41efe39693f3eaa1705ecc599a0c53797e186f4a39f30d522c594084a296a8
// Names, coordinates, level and parent below are original source fields.
// Only test extents, survey rows and point IDs are synthetic; no research data is used.
interface GeoAtlasNode {adcode:number;lng:number;lat:number;name:string;level:Admin['level'];parent:number}
export const geoAtlasNodes:GeoAtlasNode[]=[
 {"adcode":110000,"lng":116.405285,"lat":39.904989,"name":"北京市","level":"province","parent":100000},
 {"adcode":110101,"lng":116.418757,"lat":39.917544,"name":"东城区","level":"district","parent":110000},
 {"adcode":120000,"lng":117.190182,"lat":39.125596,"name":"天津市","level":"province","parent":100000},
 {"adcode":120101,"lng":117.195907,"lat":39.118327,"name":"和平区","level":"district","parent":120000},
 {"adcode":310000,"lng":121.472644,"lat":31.231706,"name":"上海市","level":"province","parent":100000},
 {"adcode":310101,"lng":121.490317,"lat":31.222771,"name":"黄浦区","level":"district","parent":310000},
 {"adcode":410000,"lng":113.665412,"lat":34.757975,"name":"河南省","level":"province","parent":100000},
 {"adcode":410100,"lng":113.665412,"lat":34.757975,"name":"郑州市","level":"city","parent":410000},
 {"adcode":410102,"lng":113.611576,"lat":34.748286,"name":"中原区","level":"district","parent":410100},
 {"adcode":419001,"lng":112.590047,"lat":35.090378,"name":"济源市","level":"city","parent":410000},
 {"adcode":420000,"lng":114.298572,"lat":30.584355,"name":"湖北省","level":"province","parent":100000},
 {"adcode":420100,"lng":114.298572,"lat":30.584355,"name":"武汉市","level":"city","parent":420000},
 {"adcode":420102,"lng":114.30304,"lat":30.594911,"name":"江岸区","level":"district","parent":420100},
 {"adcode":429004,"lng":113.453974,"lat":30.364953,"name":"仙桃市","level":"city","parent":420000},
 {"adcode":429005,"lng":112.896866,"lat":30.421215,"name":"潜江市","level":"city","parent":420000},
 {"adcode":429006,"lng":113.165862,"lat":30.653061,"name":"天门市","level":"city","parent":420000},
 {"adcode":429021,"lng":110.671525,"lat":31.744449,"name":"神农架林区","level":"city","parent":420000},
 {"adcode":440000,"lng":113.280637,"lat":23.125178,"name":"广东省","level":"province","parent":100000},
 {"adcode":440100,"lng":113.280637,"lat":23.125178,"name":"广州市","level":"city","parent":440000},
 {"adcode":440103,"lng":113.243038,"lat":23.124943,"name":"荔湾区","level":"district","parent":440100},
 {"adcode":441900,"lng":113.746262,"lat":23.046237,"name":"东莞市","level":"city","parent":440000},
 {"adcode":442000,"lng":113.382391,"lat":22.521113,"name":"中山市","level":"city","parent":440000},
 {"adcode":460000,"lng":110.33119,"lat":20.031971,"name":"海南省","level":"province","parent":100000},
 {"adcode":460100,"lng":110.33119,"lat":20.031971,"name":"海口市","level":"city","parent":460000},
 {"adcode":460105,"lng":110.282393,"lat":20.008145,"name":"秀英区","level":"district","parent":460100},
 {"adcode":460400,"lng":109.576782,"lat":19.517486,"name":"儋州市","level":"city","parent":460000},
 {"adcode":469001,"lng":109.516662,"lat":18.776921,"name":"五指山市","level":"city","parent":460000},
 {"adcode":469024,"lng":109.687697,"lat":19.908293,"name":"临高县","level":"city","parent":460000},
 {"adcode":500000,"lng":106.504962,"lat":29.533155,"name":"重庆市","level":"province","parent":100000},
 {"adcode":500101,"lng":108.380246,"lat":30.807807,"name":"万州区","level":"district","parent":500000},
 {"adcode":620000,"lng":103.823557,"lat":36.058039,"name":"甘肃省","level":"province","parent":100000},
 {"adcode":620200,"lng":98.277304,"lat":39.786529,"name":"嘉峪关市","level":"city","parent":620000},
 {"adcode":650000,"lng":87.617733,"lat":43.792818,"name":"新疆维吾尔自治区","level":"province","parent":100000},
 {"adcode":650100,"lng":87.617733,"lat":43.792818,"name":"乌鲁木齐市","level":"city","parent":650000},
 {"adcode":650102,"lng":87.620116,"lat":43.796428,"name":"天山区","level":"district","parent":650100},
 {"adcode":659001,"lng":86.041075,"lat":44.305886,"name":"石河子市","level":"city","parent":650000},
 {"adcode":659010,"lng":84.8275959,"lat":44.69288853,"name":"胡杨河市","level":"city","parent":650000},
 {"adcode":710000,"lng":121.509062,"lat":25.044332,"name":"台湾省","level":"province","parent":100000},
 {"adcode":810000,"lng":114.173355,"lat":22.320048,"name":"香港特别行政区","level":"province","parent":100000},
 {"adcode":810001,"lng":114.1543731,"lat":22.28198083,"name":"中西区","level":"district","parent":810000},
 {"adcode":820000,"lng":113.54909,"lat":22.198951,"name":"澳门特别行政区","level":"province","parent":100000},
 {"adcode":820001,"lng":113.5528956,"lat":22.20787,"name":"花地玛堂区","level":"district","parent":820000},
];
export function geoAtlasAdmins():Admin[]{
 const index=new Map(geoAtlasNodes.map(node=>[node.adcode,node]));
 return geoAtlasNodes.map(node=>{
  let province=node;while(province.level!=='province')province=index.get(province.parent)!;
  const parent=index.get(node.parent);
  return {id:`${node.level}:${node.adcode}`,code:String(node.adcode),name:node.name,level:node.level,parent:parent?`${parent.level}:${parent.adcode}`:String(node.parent),province:`province:${province.adcode}`,center:[node.lng,node.lat],extent:[node.lng-.1,node.lat-.1,node.lng+.1,node.lat+.1],source:'GeoAtlas fixture'};
 });
}
export function geoAtlasResources():Resources{
 const admin=geoAtlasAdmins(),leaves=admin.filter(a=>!admin.some(child=>child.parent===a.id));
 const sites=leaves.map(a=>({id:`fixture:${a.code}`,name:a.name,lon:a.center[0],lat:a.center[1],admin:a.id}));
 const rows=leaves.map((a,i)=>({id:`fixture:${a.code}`,sheet:'combined',row:i+1,values:[a.name,admin.find(p=>p.id===a.province)!.name,'','','','','测试大区','测试区','测试片',''],original:[],admin:a.id,match:'fixture'}));
 return {catalog:{version:'m19-local-20261022',features:[{id:'F1_P',volume:'P',number:1,page:1,title:'测试用语言现象',titleReviewed:true,categories:['A1'],labels:{A1:'测试类'},legend:''}],survey:'Synthetic selector test',sources:[]},sites,observations:{F1_P:{values:sites.map(()=>'A1'),raw:{}}},classification:{author:'Test fixture',book:'Test fixture',columns:[],sheets:[{id:'combined',name:'测试表',rows}]},admin,paper:{points:sites.map((_,i)=>({mds:[i,0,0],sharp:[],heat:[]})),matrices:[],colors:{}}};
}
