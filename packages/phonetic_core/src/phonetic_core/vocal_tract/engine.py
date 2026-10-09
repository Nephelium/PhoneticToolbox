"""VTL 2.4 stable ABI adapter. Geometry is isolated from the audio engine.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import ctypes as C
import hashlib
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import find_peaks
from .source import validate_source
from .native_resources import resolve_libraries
from phonetic_core.vocal_tract.source_models import SOURCE_PRESETS, SOURCE_LIMITS


D=C.POINTER(C.c_double)
I=C.POINTER(C.c_int)
def dp(a): return a.ctypes.data_as(D)
def ip(a): return a.ctypes.data_as(I)
def check(value):
    if value != 0: raise RuntimeError(f'VTL error {value}')

PARTS=['upper_cover','lower_cover','upper_teeth','lower_teeth','upper_lip','lower_lip','tongue','uvula','epiglottis']
LABELS={'HX':'舌骨前后','HY':'舌骨高低','JX':'下颌前后','JA':'下颌开合','LP':'唇前突','LD':'唇开度','VS':'软腭形态','VO':'腭咽开口','TCX':'舌背前后','TCY':'舌背高低','TTX':'舌尖前后','TTY':'舌尖高低','TBX':'舌叶前后','TBY':'舌叶高低','TRX':'舌根前后','TRY':'舌根高低','TS1':'舌后部侧缘','TS2':'舌中部侧缘','TS3':'舌尖侧缘'}

class Engine:
    def __init__(self,speaker='JD2',*,resource_dir):
        resource_dir=Path(resource_dir)
        if speaker not in ('JD2','M01','W02'): raise ValueError('Unknown speaker')
        self.speaker=speaker
        self.speaker_path=resource_dir/f'{speaker}.speaker'
        synthesis_path,analysis_path,geometry_path=resolve_libraries(resource_dir)
        if hashlib.sha256(analysis_path.read_bytes()).digest()!=hashlib.sha256(synthesis_path.read_bytes()).digest():
            raise RuntimeError('Analysis library does not match pinned synthesis library')
        self.lib=C.CDLL(str(synthesis_path))
        signatures={
            'vtlInitialize':([C.c_char_p],C.c_int),'vtlClose':([],C.c_int),
            'vtlGetVersion':([C.c_char_p],None),
            'vtlGetConstants':([I,I,I,I],C.c_int),
            'vtlGetTractParamInfo':([C.c_char_p,D,D,D],C.c_int),
            'vtlGetGlottisParamInfo':([C.c_char_p,D,D,D],C.c_int),
            'vtlGetTractParams':([C.c_char_p,D],C.c_int),
            'vtlTractToTube':([D,D,D,I,D,D,D],C.c_int),
            'vtlGetTransferFunction':([D,C.c_int,D,D],C.c_int),
            'vtlResetTractSynthesis':([],C.c_int),
            'vtlSynthesisAddTract':([C.c_int,D,D,D],C.c_int),
            'vtlSynthesisAddTube':([C.c_int,D,D,D,I,C.c_double,C.c_double,C.c_double,D],C.c_int),
        }
        for name,(args,restype) in signatures.items():
            f=getattr(self.lib,name); f.argtypes=args; f.restype=restype
        check(self.lib.vtlInitialize(str(self.speaker_path).encode()))
        self.analysis=C.CDLL(str(analysis_path))
        for name,(args,restype) in signatures.items():
            f=getattr(self.analysis,name); f.argtypes=args; f.restype=restype
        check(self.analysis.vtlInitialize(str(self.speaker_path).encode()))
        version=C.create_string_buffer(256); self.lib.vtlGetVersion(version)
        self.version=version.value.decode()
        cs=[C.c_int() for _ in range(4)]
        check(self.lib.vtlGetConstants(*[C.byref(c) for c in cs]))
        self.sr,self.ntube,self.ntract,self.nglottis=[c.value for c in cs]
        self.names,self.minimum,self.maximum,self.neutral=self._info('Tract',self.ntract)
        self.legacy_maximum=self.maximum.copy()
        self.maximum[self.names.index('TTX')]=7.5
        self.maximum[self.names.index('TBX')]=6.5
        anatomy=ET.parse(self.speaker_path).getroot().find('vocal_tract_model/anatomy')
        pharynx=anatomy.find('pharynx');body=anatomy.find('tongue/body')
        angle=np.deg2rad(float(pharynx.get('rotation_angle_deg')))
        self.posterior={'x':float(pharynx.get('fulcrum_x')),'y':float(pharynx.get('fulcrum_y')),
            'slope':float(np.cos(angle)/np.sin(angle)),'rx':float(body.get('radius_x')),'ry':float(body.get('radius_y'))}
        self.gnames,self.gmin,self.gmax,self.gneutral=self._info('Glottis',self.nglottis)
        self.params=self.preset('a')
        self.lock=threading.RLock()
        self.geometry_lock=threading.RLock()
        self.analysis_lock=threading.Lock()
        self.geom=C.CDLL(str(geometry_path))
        for name,args in {'p0_open':[C.c_char_p],'p0_update':[D,D],'p0_mesh':[C.c_int,D,I,I],'p0_sections':[D,D,D],'p0_profile':[C.c_int,D,D],'p1_contour':[C.c_int,D],'p1_nasal':[D,D,D]}.items():
            getattr(self.geom,name).argtypes=args; getattr(self.geom,name).restype=C.c_int
        self.geom.p0_close.argtypes=[]; self.geom.p0_close.restype=None
        for name,args in {'p2_update':[D,C.c_double,D],'p2_tube':[D,D,I,D],'p2_transfer':[C.c_int,D,D]}.items():
            getattr(self.geom,name).argtypes=args;getattr(self.geom,name).restype=C.c_int
        check(self.geom.p0_open(str(self.speaker_path).encode()))
        self.geom.p3_manual_root.argtypes=[C.c_int];self.geom.p3_manual_root.restype=C.c_int
        self.geom.p3_profile.argtypes=[C.c_int,D,D];self.geom.p3_profile.restype=C.c_int
        self.geom.p3_lateral_fit.argtypes=[C.c_int];self.geom.p3_lateral_fit.restype=None
        self.geom.p4_larynx.argtypes=[C.c_double];self.geom.p4_larynx.restype=C.c_int
        self.geom.p4_uvula_contact.argtypes=[];self.geom.p4_uvula_contact.restype=C.c_double
        self.geom.p4_blade_rib.argtypes=[];self.geom.p4_blade_rib.restype=C.c_double
        self.geom.p4_section_count.argtypes=[];self.geom.p4_section_count.restype=C.c_int
        self.section_count=self.geom.p4_section_count()
        self.manual_root=False
        self.meshmeta=[]
        for i in range(len(PARTS)):
            counts=np.zeros(4,dtype=np.int32); check(self.geom.p0_mesh(i,None,None,ip(counts)))
            self.meshmeta.append(counts)
        self.presets={}
        for name in ['a','i','u','e','o','E','y','2']:
            try: self.presets[name]=self.preset(name).tolist()
            except RuntimeError: pass
        self.consonants={}
        for symbol,name in [('l','tt-alveolar-lateral(a)'),('s','tt-alveolar-fricative(a)'),('n','tt-alveolar-closure(a)'),('t','tt-alveolar-closure(a)')]:
            pose=self.preset(name)
            if symbol=='l':pose[self.names.index('TS3')]=-.6
            if symbol=='s':pose[self.names.index('TS3')]=.32
            if symbol=='n':pose[self.names.index('VO')]=.5
            self.consonants[symbol]={'params':pose.tolist(),'source':dict(SOURCE_PRESETS['voiceless' if symbol in ('s','t') else 'voiced']),
                'native_shape':name,'label':f'/{symbol}/','adaptation':'M10 lateral geometry / TS3 fit' if symbol=='l' else 'M10 TS3 narrow-constriction start' if symbol=='s' else ''}
            self.presets[symbol]=pose.tolist()

    def set_manual_root(self,manual=False):
        if type(manual) is not bool:raise ValueError('Invalid tongue root mode')
        with self.geometry_lock:
            check(self.geom.p3_manual_root(int(manual)));self.manual_root=manual

    def _info(self,kind,n):
        names=C.create_string_buffer(128*n); arrays=[np.zeros(n) for _ in range(3)]
        check(getattr(self.lib,f'vtlGet{kind}ParamInfo')(names,*[dp(a) for a in arrays]))
        return names.value.decode().split(),*arrays

    def preset(self,name):
        p=np.empty(self.ntract); check(self.lib.vtlGetTractParams(name.encode(),dp(p))); return p

    def validated(self,values):
        p=np.asarray(values,dtype=np.float64)
        if p.shape != (self.ntract,) or not np.isfinite(p).all(): raise ValueError('Invalid tract parameters')
        p=np.clip(p,self.minimum,self.maximum)
        wall=self.posterior
        back=lambda y:wall['x']+(y-wall['y'])*wall['slope']
        # Ellipse support against the model's posterior wall. Native VTL permits
        # construction circles beyond its hull; interactive targets stop at contact.
        cx,cy,rx,ry=[self.names.index(k) for k in ('TCX','TCY','TRX','TRY')]
        tangent=back(p[cy])+np.hypot(wall['rx'],wall['slope']*wall['ry'])
        p[cx]=max(p[cx],tangent);p[rx]=max(p[rx],back(p[ry]))
        return np.ascontiguousarray(p)

    def glottis(self,f0=125,pressure=8000,*,source=None):
        settings=validate_source(source) if source is not None else None
        if settings is not None:pressure=settings['pressure_pa']*10 # native dPa
        if not np.isfinite([f0,pressure]).all() or not 60 <= f0 <= 350 or not 0 <= pressure <= 16000:
            raise ValueError('Invalid source parameters')
        g=self.gneutral.copy(); g[0]=f0; g[1]=pressure
        g[self.gnames.index('flutter')]=0
        if settings is not None:
            for key in ['x_bottom','x_top']:g[self.gnames.index(key)]=settings['opening_mm']/10
            g[self.gnames.index('chink_area')]=settings['posterior_gap_mm2']/100
            g[self.gnames.index('rel_amp')]=settings['vibration']
            # An F0 curve has no physical pitch meaning when vibration is off.
            if settings['vibration']==0:g[0]=125.
        return np.clip(g,self.gmin,self.gmax)

    def reset(self,p,g):
        check(self.lib.vtlResetTractSynthesis())
        z=np.zeros(1); check(self.lib.vtlSynthesisAddTract(0,dp(z),dp(p),dp(g)))

    def block(self,p,g,n=480):
        out=np.empty(n)
        check(self.lib.vtlSynthesisAddTract(n,dp(out),dp(p),dp(g)))
        if not np.isfinite(out).all(): raise RuntimeError('Nonfinite synthesis')
        return out

    def synthesize(self,start,end=None,duration=1.2,f0=125):
        return self.synthesize_path([start,end if end is not None else start],duration,f0)

    def synthesize_path(self,poses,duration=2.4,f0=125):
        if not .1 <= duration <= 5: raise ValueError('Duration must be 0.1 to 5 seconds')
        if not 2 <= len(poses) <= 8: raise ValueError('Expected 2 to 8 poses')
        poses=[self.validated(p) for p in poses]
        g=self.glottis(f0); n=int(round(self.sr*duration)); chunks=[]
        with self.lock:
            self.reset(poses[0],g)
            for k in range(0,n,480):
                t=(k+min(480,n-k))/n
                progress=float(np.clip((t-.15)/.7,0,1))*(len(poses)-1)
                segment=min(int(progress),len(poses)-2)
                t=progress-segment; t=t*t*(3-2*t)
                p,q=poses[segment:segment+2]
                chunks.append(self.block((1-t)*p+t*q,g,min(480,n-k)))
        audio=np.concatenate(chunks)
        fade=min(480,n//4); audio[:fade]*=np.linspace(0,1,fade); audio[-fade:]*=np.linspace(1,0,fade)
        return audio

    @staticmethod
    def validated_larynx(value=0.):
        if isinstance(value,bool):raise ValueError('Invalid larynx height')
        value=float(value)
        if not np.isfinite(value) or not -1.<=value<=1.:raise ValueError('声门高低范围为 -1 至 1 cm')
        return value

    def prepare_tube(self,p,lip_width=1.,larynx_height=0.):
        p=self.validated(p)
        larynx_height=self.validated_larynx(larynx_height)
        if not np.isfinite(lip_width) or not .55<=lip_width<=1.6:raise ValueError('Invalid lip width')
        with self.geometry_lock:
            check(self.geom.p4_larynx(larynx_height))
            limited=np.empty(self.ntract);check(self.geom.p2_update(dp(p),lip_width,dp(limited)))
            return self._current_tube()

    def _current_tube(self):
        lengths=np.empty(self.ntube);areas=np.empty(self.ntube);arts=np.empty(self.ntube,dtype=np.int32);extras=np.empty(3)
        check(self.geom.p2_tube(dp(lengths),dp(areas),ip(arts),dp(extras)))
        return {'lengths':lengths,'areas':areas,'articulators':arts,'extras':extras}

    def reset_tube(self,tube,g):
        check(self.lib.vtlResetTractSynthesis());self.block_tube(tube,g,0)

    def block_tube(self,tube,g,n=960):
        out=np.empty(max(1,n));check(self.lib.vtlSynthesisAddTube(n,dp(out),dp(tube['lengths']),dp(tube['areas']),ip(tube['articulators']),*tube['extras'],dp(g)))
        if not np.isfinite(out[:n]).all():raise RuntimeError('Nonfinite synthesis')
        return out[:n]

    def constrain_nasal_opening(self,p,lip_width=1.,larynx_height=0.):
        p=self.validated(p);index=self.names.index('VO');requested=max(0.,p[index]);base=p.copy();base[index]=0
        if requested<=0:return p,None
        with self.geometry_lock:
            baseline=float(self.prepare_tube(base,lip_width,larynx_height)['areas'][16:].min())
            threshold=min(.08,baseline*.5)
            if baseline<.01 or self.prepare_tube(p,lip_width,larynx_height)['areas'][16:].min()>=threshold:return p,None
            # Find the first unsafe interval, then bisect. Do not assume the
            # entire native response is monotonic over arbitrarily large VO.
            low=0.;high=requested
            for value in np.linspace(0,requested,13)[1:]:
                q=p.copy();q[index]=value
                if self.prepare_tube(q,lip_width,larynx_height)['areas'][16:].min()<threshold:high=value;break
                low=value
            for _ in range(9):
                mid=(low+high)/2;q=p.copy();q[index]=mid
                if self.prepare_tube(q,lip_width,larynx_height)['areas'][16:].min()>=threshold:low=mid
                else:high=mid
            p[index]=max(0.,np.floor(low*100)/100)
        return p,{'requested_area':requested,'accepted_area':float(p[index]),'minimum_oral_area':threshold}

    def snapshot(self,p,section=None,lip_width=1.,larynx_height=0.):
        p=self.validated(p); section=int(np.clip(self.section_count//2 if section is None else section,0,self.section_count-1)); started=time.perf_counter()
        larynx_height=self.validated_larynx(larynx_height)
        if not np.isfinite(lip_width) or not .55<=lip_width<=1.6:raise ValueError('Invalid lip width')
        with self.geometry_lock:
            check(self.geom.p4_larynx(larynx_height))
            limited=np.zeros(self.ntract); check(self.geom.p2_update(dp(p),float(lip_width),dp(limited)))
            meshes=[]
            for i,(nv,nf,nr,np_) in enumerate(self.meshmeta):
                v=np.empty((nv,3)); f=np.empty((nf,3),dtype=np.int32); c=np.zeros(4,dtype=np.int32)
                check(self.geom.p0_mesh(i,dp(v),ip(f),ip(c)))
                meshes.append({'name':PARTS[i],'vertices':v.round(5).ravel().tolist(),'triangles':f.ravel().tolist(),'ribs':int(nr),'points':int(np_)})
            center=np.empty((self.section_count,5)); areas=np.empty(self.section_count); tube=np.empty(self.ntube)
            assert self.geom.p0_sections(dp(center),dp(areas),dp(tube)) == self.section_count
            upper=np.empty(96); lower=np.empty(96)
            assert self.geom.p3_profile(section,dp(upper),dp(lower)) == 96
            contours={}
            for i,name in enumerate(PARTS):
                n=self.geom.p1_contour(i,None);points=np.empty((n,2));assert self.geom.p1_contour(i,dp(points))==n
                contours[name]=points.round(5).tolist()
            nasal_lengths=np.empty(19);nasal_areas=np.empty(19);nasal_port=np.empty(2)
            assert self.geom.p1_nasal(dp(nasal_lengths),dp(nasal_areas),dp(nasal_port))==19
            airway_sections=[]
            for i in range(self.section_count):
                up=np.empty(96);lo=np.empty(96);self.geom.p3_profile(i,dp(up),dp(lo))
                ok=(np.abs(up)<100)&(np.abs(lo)<100)&(up>=lo)
                gap=np.where(ok,np.maximum(0,up-lo),0)
                airway_sections.append({'index':i,'area':float(np.sum((gap[:-1]+gap[1:])*.5)*(7/96)),
                    'upper':[round(float(v),5) if flag else None for v,flag in zip(up,ok)],'lower':[round(float(v),5) if flag else None for v,flag in zip(lo,ok)]})
            acoustic=self._current_tube();lengths=acoustic['lengths'];tube_api=acoustic['areas']
            mag=np.empty(4096); phase=np.empty(4096)
            check(self.geom.p2_transfer(4096,dp(mag),dp(phase)))
            uvula_contact_lift=float(self.geom.p4_uvula_contact())
            tongue_blade_rib=float(self.geom.p4_blade_rib())
        freq=np.arange(4096)*self.sr/4096; keep=(freq<=6000)
        db=20*np.log10(np.maximum(mag,1e-10))
        peaks,_=find_peaks(db[:513],prominence=3,distance=8)
        peaks=peaks[freq[peaks]>100][:4]
        valid=(np.abs(upper)<100)&(np.abs(lower)<100)&(upper>=lower)
        return {'params':p.tolist(),'geometry_version':'m10/3','tongue_blade_rib':tongue_blade_rib,'uvula_contact_lift':uvula_contact_lift,'larynx_height':larynx_height,'manual_root':self.manual_root,'lip_width':float(lip_width),'limited':limited.tolist(),'meshes':meshes,'contours':contours,'airway_sections':airway_sections,
            'nasal':{'lengths':nasal_lengths.round(5).tolist(),'areas':nasal_areas.round(5).tolist(),'port_position':float(nasal_port[0]),'port_area':float(nasal_port[1])},
            'centerline':center.round(5).tolist(),'raw_areas':areas.round(6).tolist(),
            'tube_areas':tube_api.round(6).tolist(),'tube_lengths':lengths.round(6).tolist(),
            'nasal_area':max(.0001,float(nasal_port[1])),'geometry_area_max_error':float(np.max(np.abs(tube-tube_api))),
            'oral_min_area':float(tube_api[16:].min()),
            'section':section,'upper':[float(v) if ok else None for v,ok in zip(upper,valid)],
            'lower':[float(v) if ok else None for v,ok in zip(lower,valid)],
            'frequency':freq[keep].tolist(),'transfer_db':db[keep].round(3).tolist(),
            'resonances':freq[peaks].round(1).tolist(),'compute_ms':(time.perf_counter()-started)*1000}

    def metadata(self):
        return {'version':self.version,'geometry_version':'m10/3','section_count':self.section_count,'larynx_limits':[-1.,1.],'posterior_limits':self.posterior,'speaker':self.speaker,'sample_rate':self.sr,'num_tube_sections':self.ntube,'source_presets':SOURCE_PRESETS,'source_limits':SOURCE_LIMITS,
            'parameters':[{'name':n,'label':LABELS[n],'min':float(lo),'max':float(hi),'neutral':float(ne)} for n,lo,hi,ne in zip(self.names,self.minimum,self.maximum,self.neutral)],
            'presets':self.presets,'consonants':self.consonants,'glottis_names':self.gnames,'glottis_neutral':self.gneutral.tolist(),
            'speaker_sha256':hashlib.sha256(self.speaker_path.read_bytes()).hexdigest()}

    def close(self):
        self.geom.p0_close(); check(self.lib.vtlClose()); check(self.analysis.vtlClose())
