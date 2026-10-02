"""Reproducible OFL module font derivative and coverage audit (no installation).

All original Doulos glyphs/outlines are retained. Added glyphs implement the six
Unicode partial-voicing parentheses, finite catalogue enclosing-circle ligatures,
and Noto Math's explicitly labelled text-substitute delimiters. RFNs are removed
from the derivative family. --write regenerates only M17-owned resources.
"""
from __future__ import annotations
import argparse, copy, hashlib, io, json, math
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'frontend/src/assets/ipa-plus'
BASE=ROOT/'frontend/src/assets/DoulosSIL-Regular.ttf'
OUTPUT=ASSETS/'PTBIPAPlus-Regular.ttf'

def build():
    font=TTFont(BASE,recalcTimestamp=False);cmap=font.getBestCmap();glyphs=font.getGlyphSet()
    original_order=font.getGlyphOrder();order=list(original_order);added=[]
    def add(name,glyph,width,cp=None):
        font['glyf'][name]=glyph;glyph.recalcBounds(font['glyf']);font['hmtx'][name]=(round(width),glyph.xMin if glyph.numberOfContours else 0);order.append(name);added.append(name)
        if cp is not None:
            for table in font['cmap'].tables:
                if table.isUnicode() and (cp<=65535 or table.format==12):table.cmap[cp]=name
        font.setGlyphOrder(list(order));font['glyf'].glyphOrder=list(order)
    def draw_paren(pen,left=True):
        # Thin curved outlines, centred on the modified ring/caron.
        x=-400 if left else 400;direction=1 if left else -1
        pen.moveTo((x+direction*105,-340));pen.qCurveTo((x-direction*75,0),(x+direction*105,340))
        pen.lineTo((x+direction*140,307));pen.qCurveTo((x-direction*8,0),(x+direction*140,-307));pen.closePath()
    for cp in (0x1ABB,0x1ABD,0x1AC1,0x1AC2,0x1AC3,0x1AC4):
        pen=TTGlyphPen(None)
        if cp not in (0x1AC2,0x1AC4):draw_paren(pen,True)
        if cp not in (0x1AC1,0x1AC3):draw_paren(pen,False)
        add(f'm17_{cp:04X}',pen.glyph(),0,cp)
    def ellipse(pen,cx,cy,rx,ry,clockwise=False):
        # Eight quadratic arcs, reversed inner contour for nonzero fill.
        pen.moveTo((cx+rx,cy));direction=-1 if clockwise else 1
        for i in range(8):
            a=direction*i*math.pi/4;b=direction*(i+1)*math.pi/4;m=(a+b)/2
            factor=1/math.cos(math.pi/8)
            pen.qCurveTo((cx+rx*math.cos(m)*factor,cy+ry*math.sin(m)*factor),(cx+rx*math.cos(b),cy+ry*math.sin(b)))
        pen.closePath()
    p=TTGlyphPen(None);ellipse(p,0,600,650,750);ellipse(p,0,600,590,690,True);add('m17_circle',p.glyph(),0,0x20DD)
    p=TTGlyphPen(None);ellipse(p,700,600,650,750);ellipse(p,700,600,590,690,True);add('m17_emptycircle',p.glyph(),1400,0x25EF)
    math_path=ROOT/'output/validation/m17/source-fonts/NotoSansMath-Regular.ttf'
    if not math_path.exists():
        raise RuntimeError('Download official NotoSansMath-Regular.ttf to '+str(math_path)+'; URL and SHA256 are recorded in m17-font-audit.md.')
    math_font=TTFont(math_path);scale=font['head'].unitsPerEm/math_font['head'].unitsPerEm
    for cp in (0x27C5,0x27C6):
        name=math_font.getBestCmap()[cp];p=TTGlyphPen(None);math_font.getGlyphSet()[name].draw(TransformPen(p,(scale,0,0,scale,0,0)))
        add(f'm17_{cp:04X}',p.glyph(),math_font['hmtx'][name][0]*scale,cp)
    cmap=font.getBestCmap();features=[]
    bases=['C','Ȼ','F','L','G','N','P','R','S','T','Ṽ','Ʞ','σ']
    for i,ch in enumerate(bases):
        name=cmap[ord(ch)];g=font['glyf'][name];g.recalcBounds(font['glyf'])
        width=font['hmtx'][name][0];cx=width/2;cy=(g.yMin+g.yMax)/2
        rx=max(width*.8,500);ry=max((g.yMax-g.yMin)*.65+160,690);padding=rx-cx+70
        p=TTGlyphPen(font.getGlyphSet());p.addComponent(name,(1,0,0,1,padding,0))
        # Composite plus contours is disallowed by TrueType, draw outlines.
        p=TTGlyphPen(glyphs);glyphs[name].draw(TransformPen(p,(1,0,0,1,padding,0)))
        ellipse(p,cx+padding,cy,rx,ry);ellipse(p,cx+padding,cy,rx-55,ry-55,True)
        new=f'm17_circled_{i}';add(new,p.glyph(),2*(rx+70))
        features.append(f'sub {name} m17_circle by {new};')
        if ch=='Ṽ':features.append(f'sub {cmap[ord("V")]} {cmap[0x0303]} m17_circle by {new};')
    # Construct additional lookups separately so upstream GSUB/GPOS is unchanged.
    temp=TTFont();temp.setGlyphOrder(order)
    marks=' '.join(f'm17_{cp:04X}' for cp in (0x1ABB,0x1ABD,0x1AC1,0x1AC2,0x1AC3,0x1AC4))
    fea=f'markClass [{marks}] <anchor 0 0> @PARTIAL;\nfeature mkmk {{\n'
    for cp in (0x0325,0x030A,0x032C):
        name=cmap[cp];g=font['glyf'][name];g.recalcBounds(font['glyf'])
        fea+=f'pos mark {name} <anchor {round((g.xMin+g.xMax)/2)} {round((g.yMin+g.yMax)/2)}> mark @PARTIAL;\n'
    fea+='} mkmk;\nfeature ccmp {\n'+'\n'.join(features)+'\n} ccmp;'
    addOpenTypeFeaturesFromString(temp,fea)
    for tag,feature_tag in [('GPOS','mkmk'),('GSUB','ccmp')]:
        original=font[tag].table;extra=temp[tag].table;offset=len(original.LookupList.Lookup)
        original.LookupList.Lookup.extend(extra.LookupList.Lookup);original.LookupList.LookupCount=len(original.LookupList.Lookup)
        indices=[i+offset for r in extra.FeatureList.FeatureRecord if r.FeatureTag==feature_tag for i in r.Feature.LookupListIndex]
        for record in original.FeatureList.FeatureRecord:
            if record.FeatureTag==feature_tag:
                # Circle ligatures precede upstream composition, mark positioning follows it.
                record.Feature.LookupListIndex=(indices+record.Feature.LookupListIndex if tag=='GSUB' else record.Feature.LookupListIndex+indices)
                record.Feature.LookupCount=len(record.Feature.LookupListIndex)
    classes=font['GDEF'].table.GlyphClassDef.classDefs
    for name in added:classes[name]=3 if name in marks.split() or name=='m17_circle' else 1
    names={1:'PTB IPA Plus',2:'Regular',3:'PTB IPA Plus 1.000; D7 module extension',4:'PTB IPA Plus Regular',5:'Version 1.000; based on Doulos SIL 7.000',6:'PTBIPAPlus-Regular',16:'PTB IPA Plus',17:'Regular'}
    for rec in list(font['name'].names):
        if rec.nameID in names:font['name'].setName(names[rec.nameID],rec.nameID,rec.platformID,rec.platEncID,rec.langID)
    font['name'].setName('M17 catalogue additions. Original glyphs copyright SIL Global; delimiter glyphs copyright Noto authors. OFL 1.1.',10,3,1,0x409)
    font['head'].created=font['head'].modified=3852662400
    if 'DSIG' in font:
        font['DSIG'].ulNumSigs=0;font['DSIG'].signatureRecords=[]
    stream=io.BytesIO();font.save(stream,reorderTables=True)
    return stream.getvalue(),added

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data,added=build()
    if args.write:
        OUTPUT.write_bytes(data)
        (ASSETS/'OFL-PTBIPAPlus.txt').write_text(TTFont(BASE)['name'].getDebugName(13)+'\n\nPTB IPA Plus additions: copyright 2026 PhoneticToolbox contributors.\nNoto delimiter outlines: see OFL-Noto.txt.\n','utf-8')
    assert OUTPUT.read_bytes()==data,'font differs from reproducible build'
    base=TTFont(BASE);derived=TTFont(OUTPUT)
    assert all(base['glyf'][g].compile(base['glyf'])==derived['glyf'][g].compile(derived['glyf']) for g in base.getGlyphOrder()),'upstream outline modified'
    d=json.loads((ROOT/'frontend/src/modules/ipa-plus/data/catalog.json').read_text('utf-8'))
    chars=set(''.join(e['insertText']+e['display'] for e in d['entries']));missing=[f'U+{ord(c):04X}' for c in sorted(chars,key=ord) if ord(c) not in derived.getBestCmap()]
    report={'version':'m17-font/1','font':'PTB IPA Plus 1.000','base':'Doulos SIL 7.000','base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'sha256':hashlib.sha256(data).hexdigest(),'added_glyphs':added,'original_glyph_count':len(base.getGlyphOrder()),'original_glyphs_byte_identical':True,'unique_catalog_characters':len(chars),'missing_cmap':missing,'visual_status':'see m17-report.md; cmap is not shape verification','source_urls':['https://software.sil.org/doulos/','https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansMath/NotoSansMath-Regular.ttf'],'license':'OFL-1.1','family_renamed':True}
    assert not missing,missing
    if args.write:(ROOT/'docs/references/m17-font-coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
