// Independent Linux ABI probe; input is the validated public a/i/u parameters.
#include <dlfcn.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
template<class T>T symbol(void* lib,const char* name){auto value=dlsym(lib,name);assert(value);return reinterpret_cast<T>(value);}
void array(std::ostream& out,const std::vector<double>& values){out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';assert(std::isfinite(values[i]));out<<values[i];}out<<']';}
int main(int argc,char** argv){
 assert(argc==4);std::string root=argv[1];auto lib=dlopen((root+"/linux-x86_64/libVocalTractLabApi.so").c_str(),RTLD_NOW|RTLD_LOCAL);assert(lib);
 auto geom=dlopen((root+"/linux-x86_64/libgeometry_p2.so").c_str(),RTLD_NOW|RTLD_LOCAL);assert(geom);
 auto initialize=symbol<int(*)(const char*)>(lib,"vtlInitialize");assert(initialize((root+"/JD2.speaker").c_str())==0);
 assert(symbol<int(*)(const char*)>(geom,"p0_open")((root+"/JD2.speaker").c_str())==0);
 auto sections=symbol<int(*)(int)>(geom,"p5_sections");auto update=symbol<int(*)(double*,double,double*)>(geom,"p2_update");
 auto tube=symbol<int(*)(double*,double*,int*,double*)>(geom,"p2_tube");auto reset=symbol<int(*)()>(lib,"vtlResetTractSynthesis");
 auto synth=symbol<int(*)(int,double*,double*,double*,int*,double,double,double,double*)>(lib,"vtlSynthesisAddTube");
 std::ifstream input(argv[2]);std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"cases\":[";int cases,nt,ng,ns;input>>cases>>nt>>ng>>ns;
 for(int c=0;c<cases;++c){
  std::string name;int count;input>>name>>count;std::vector<double> params(nt),glottis(ng),limited(nt),lengths(ns),areas(ns),extras(3),audio(960);std::vector<int> articulators(ns);
  for(auto& v:params)input>>v;for(auto& v:glottis)input>>v;
  assert(count==33||count==65);assert(sections(count)==0);assert(update(params.data(),1.,limited.data())==0);assert(tube(lengths.data(),areas.data(),articulators.data(),extras.data())==0);
  std::vector<double> center(count*5),geometric(count),acoustic(ns);assert(symbol<int(*)(double*,double*,double*)>(geom,"p0_sections")(center.data(),geometric.data(),acoustic.data())==count);
  assert(reset()==0);assert(synth(0,audio.data(),lengths.data(),areas.data(),articulators.data(),extras[0],extras[1],extras[2],glottis.data())==0);
  double energy=0,peak=0;
  for(int i=0;i<10;++i){assert(synth(960,audio.data(),lengths.data(),areas.data(),articulators.data(),extras[0],extras[1],extras[2],glottis.data())==0);for(auto v:audio){assert(std::isfinite(v));energy+=v*v;peak=std::max(peak,std::abs(v));}}
  assert(energy>1e-12);if(c)out<<',';out<<"{\"name\":\""<<name<<"\",\"section_count\":"<<count<<",\"areas\":";array(out,areas);out<<",\"lengths\":";array(out,lengths);out<<",\"geometric\":";array(out,geometric);out<<",\"samples\":9600,\"rms\":"<<std::sqrt(energy/9600)<<",\"peak\":"<<peak<<'}';
 }
 out<<"]}\n";symbol<int(*)()>(lib,"vtlClose")();symbol<void(*)()>(geom,"p0_close")();
}
