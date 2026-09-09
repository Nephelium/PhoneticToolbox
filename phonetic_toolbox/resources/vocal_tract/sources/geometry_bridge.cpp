// SPDX-License-Identifier: GPL-3.0-or-later
// Small readout bridge, compiled with the unmodified VTL 2.4 geometry sources.
#include "VocalTract.h"
#include "TlModel.h"
#include <memory>
#include <cmath>

static std::unique_ptr<VocalTract> tract;
static const int surfaces[] = {
    VocalTract::UPPER_COVER_TWOSIDE, VocalTract::LOWER_COVER_TWOSIDE,
    VocalTract::UPPER_TEETH_TWOSIDE, VocalTract::LOWER_TEETH_TWOSIDE,
    VocalTract::UPPER_LIP_TWOSIDE, VocalTract::LOWER_LIP_TWOSIDE,
    VocalTract::TONGUE, VocalTract::UVULA_TWOSIDE,
    VocalTract::EPIGLOTTIS_TWOSIDE
};
#define API extern "C" __declspec(dllexport)
API int p0_open(const char* file) {
    try { tract.reset(new VocalTract()); tract->readFromXml(file); tract->calculateAll(); return 0; }
    catch (...) { tract.reset(); return 1; }
}
API void p0_close() { tract.reset(); }
API int p0_update(double* params, double* limited) {
    if (!tract) return 1;
    for (int i=0;i<VocalTract::NUM_PARAMS;i++) if (!std::isfinite(params[i])) return 2;
    tract->setParams(params); tract->calculateAll();
    for (int i=0;i<VocalTract::NUM_PARAMS;i++) limited[i] = tract->params[i].limitedX;
    return 0;
}
// Independent transverse lip control. Recompute the native cross-sections
// from the same deformed lip surface used for display and tube synthesis.
API int p2_update(double* params,double width,double* limited) {
    if(!tract||!std::isfinite(width)||width<0.55||width>1.6)return 2;
    for(int i=0;i<VocalTract::NUM_PARAMS;i++)if(!std::isfinite(params[i]))return 2;
    tract->setParams(params);for(int i=0;i<VocalTract::NUM_PARAMS;i++)tract->params[i].limitedX=tract->params[i].x;tract->calcSurfaces();
    if(std::abs(width-1.0)>1e-9){
      const int ids[]={VocalTract::UPPER_LIP,VocalTract::LOWER_LIP,VocalTract::UPPER_LIP_TWOSIDE,VocalTract::LOWER_LIP_TWOSIDE};
      for(int which:ids){auto& s=tract->surfaces[which];
        for(int r=0;r<s.numRibs;r++)for(int j=0;j<s.numRibPoints;j++){
          auto p=s.getVertex(r,j);double t=std::max(0.0,std::min(1.0,(p.x-3.5)/1.4));t=t*t*(3.0-2.0*t);
          p.z*=1.0+(width-1.0)*t;s.setVertex(r,j,p);
        }
      }
    }
    tract->calcCenterLine();tract->calcCrossSections();tract->crossSectionsToTubeSections();
    for(int i=0;i<VocalTract::NUM_PARAMS;i++)limited[i]=tract->params[i].limitedX;
    return 0;
}
API int p2_tube(double* lengths,double* areas,int* articulators,double* extras){
    if(!tract)return 1;Tube tube;tract->getTube(&tube);
    for(int i=0;i<Tube::NUM_PHARYNX_MOUTH_SECTIONS;i++){
      auto& s=tube.pharynxMouthSections[i];lengths[i]=s.length_cm;areas[i]=s.area_cm2;articulators[i]=(int)s.articulator;
    }
    extras[0]=tract->incisorPos_cm;extras[1]=tract->nasalPortArea_cm2;extras[2]=tract->params[VocalTract::TS3].x;return 0;
}
API int p2_transfer(int n,double* magnitude,double* phase){
    if(!tract||n<16)return 1;std::unique_ptr<TlModel> model(new TlModel());tract->getTube(&model->tube);model->tube.resetGlottisSections(0.0);ComplexSignal spectrum;
    model->getSpectrum(TlModel::FLOW_SOURCE_TF,&spectrum,n,Tube::FIRST_PHARYNX_SECTION);
    for(int i=0;i<n;i++){magnitude[i]=spectrum.getMagnitude(i);phase[i]=spectrum.getPhase(i);}return 0;
}
API int p0_mesh(int which, double* xyz, int* triangles, int* counts) {
    if (!tract || which < 0 || which >= 9) return 1;
    Surface& s = tract->surfaces[surfaces[which]];
    counts[0]=s.numVertices; counts[1]=s.numTriangles;
    counts[2]=s.numRibs; counts[3]=s.numRibPoints;
    if (xyz) for (int i=0;i<s.numVertices;i++) {
        xyz[3*i]=s.vertex[i].coord.x;
        xyz[3*i+1]=s.vertex[i].coord.y;
        xyz[3*i+2]=s.vertex[i].coord.z;
    }
    if (triangles) for (int i=0;i<s.numTriangles;i++)
        for (int j=0;j<3;j++) triangles[3*i+j]=s.triangle[i].vertex[j];
    return 0;
}
API int p0_sections(double* center, double* areas, double* tubeAreas) {
    if (!tract) return 1;
    for (int i=0;i<VocalTract::NUM_CENTERLINE_POINTS;i++) {
        auto& c=tract->centerLine[i];
        center[5*i]=c.point.x; center[5*i+1]=c.point.y; center[5*i+2]=c.pos;
        center[5*i+3]=c.normal.x; center[5*i+4]=c.normal.y;
        areas[i]=tract->crossSection[i].area;
    }
    for (int i=0;i<VocalTract::NUM_TUBE_SECTIONS;i++) tubeAreas[i]=tract->tubeSection[i].area;
    return VocalTract::NUM_CENTERLINE_POINTS;
}
API int p0_profile(int section, double* upper, double* lower) {
    if (!tract || section < 0 || section >= VocalTract::NUM_CENTERLINE_POINTS) return -1;
    Tube::Articulator articulator;
    auto& c=tract->centerLine[section];
    tract->getCrossProfiles(c.point,c.normal,upper,lower,true,articulator);
    return VocalTract::NUM_PROFILE_SAMPLES;
}

// Ordered points exactly on the native model's midsagittal meridian.
API int p1_contour(int which, double* xy) {
    if (!tract || which<0 || which>=9) return -1;
    static const int original[] = {VocalTract::UPPER_COVER,VocalTract::LOWER_COVER,
      VocalTract::UPPER_TEETH,VocalTract::LOWER_TEETH,VocalTract::UPPER_LIP,
      VocalTract::LOWER_LIP,VocalTract::TONGUE,VocalTract::UVULA,VocalTract::EPIGLOTTIS};
    Surface& s=tract->surfaces[original[which]];
    bool across=which>=2 && which<=5;
    int n=across?s.numRibPoints:s.numRibs;
    for(int i=0;i<n;i++){
      auto p=across?s.getVertex(s.numRibs-1,i):s.getVertex(i,which==6?s.numRibPoints/2:s.numRibPoints-1);
      if(xy){xy[2*i]=p.x;xy[2*i+1]=p.y;}
    }
    return n;
}
API int p1_nasal(double* lengths,double* areas,double* port){
    if(!tract)return -1;
    Tube tube;tract->getTube(&tube);
    for(int i=0;i<Tube::NUM_NOSE_SECTIONS;i++){
      lengths[i]=tube.noseSections[i].length_cm;areas[i]=tube.noseSections[i].area_cm2;
    }
    port[0]=tract->nasalPortPos_cm;port[1]=tract->nasalPortArea_cm2;
    return Tube::NUM_NOSE_SECTIONS;
}
