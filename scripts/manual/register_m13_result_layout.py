"""Register verified M13 Qt images and revise the affected manual topic in place."""
import hashlib,json,sys
from pathlib import Path
from build import write_atomic
from core import dump,node_text

ROOT=Path(__file__).resolve().parents[2]

def paragraph(text):return {'type':'paragraph','content':[{'type':'text','text':text}]}

def main():
    evidence=Path(sys.argv[1]).resolve();report=json.loads((evidence/'report.json').read_text(encoding='utf-8'))
    assert report['success'] and len(report['captures'])==17 and report['export']['triangleAbsent']
    manual=ROOT/'manual';project=json.loads((manual/'project.json').read_text(encoding='utf-8'));chapter=json.loads((manual/'chapters/m13.json').read_text(encoding='utf-8'))
    blocks=chapter['body']['content'];start=next((i for i,b in enumerate(blocks) if b.get('attrs',{}).get('id')=='m13-tone-pinyin'),None)
    if start is not None:
        end=next(i for i in range(start+1,len(blocks)) if blocks[i]['type']=='heading')
        del blocks[start:end]
    replacements={
      '银行行走女略':'银行行走','女略':'银行行走','独立的儿需要逐项核对':'独立的儿字需要逐项核对',
      '浅色实际浏览器工作台':'浅色实际Qt工作台',
      '女的严式上声为213':'走的严式上声为213',
      '计数12':'计数10',
      '字符计数包含标点、空白、换行和表情。':'字符计数在汉字输入标题右侧，包含标点、空白、换行和表情。左右排布默认让输入与结果两栏等宽，可拖动边界调整。',
      '先调两层字号，再调字音间距和行距。检查首行、中间和末行，避免负间距或过小行距挤压附加符号。需要突出参考汉字时，再开启粗体、斜体或下划线。':'先调两层字号，再调字音间距和行距。行距同时作用于手动换行与自动换行，数值增大时行间空隙随之增大，空行保留。需要对齐各字组时，点击等宽排版。需要突出参考汉字时，再开启粗体、斜体或下划线。',
      '调整行间距离。附加符号拥挤时适当增大；PNG 会保留避免行重叠所需的最小间隔。':'控制结果实际的行间空隙。0.8为紧排，增大数值可拉开各行，页面和PNG都生效；字音间距另外控制同一字组内的两层文字。',
      '。文字、读音、声调和排版采用点击导出时的状态，多音 ▼ 也可能保留。':'。文字、读音、声调、等宽排版和行距采用点击导出时的状态，图片不包含多音小三角或读音选择窗。',
      '，花和月下方有蓝色三角。':'，图中不包含多音小三角。',
      '其他标准保留历史字表，具体著作版次及单字条目仍需按研究用途核对。':'其他标准沿用各自历史记音规则；已核实的异常字音键统一修正，正常映射保持原样。具体著作版次及单字条目仍需按研究用途核对。'
    }
    def revise(node):
        for key in ['text','caption','alt']:
            if key in node:
                for old,new in replacements.items():node[key]=node[key].replace(old,new)
        if 'attrs' in node:revise(node['attrs'])
        for child in node.get('content',[]):revise(child)
    revise(chapter['body'])
    for block in blocks:
        identity=block.get('attrs',{}).get('id')
        if identity=='m13-zhao-example':block['content']=[{'type':'text','text':'选择赵元任《汉语口语语法》，春显示 '},{'type':'text','text':'tʂʰuən˥','marks':[{'type':'ipa'}]},{'type':'text','text':'。比较时保持逐位置读音选择。'}]
        if identity=='m13-strict-onlyipa-example':revise_readings='图中第一处行已选择hang2，第二处保留xing2；切换标准时保持这些逐位置读音决定。';block['content'][-1]['text']='。未手调时音标字号为28px。'+revise_readings
        if identity=='m13-standard-shots-intro':block['content']=[{'type':'text','text':'下列两图使用相同两行文本、字音同显、声调开启、汉字24px、音标16px。银行的行已选择hang2，行走的行保留xing2，便于比较同一读音在不同标准中的记法。'}]
    images={b['attrs']['id']:b['attrs'] for b in blocks if b['type']=='image'}
    images['m13-figure-m13-overview-r2-light']['caption']='工作台总览：输入与结果两栏默认等宽，字符数在输入标题右侧，左侧提供等宽排版；儿化操作区仅在Unt两项下显示。'
    images['m13-figure-m13-overview-r2-light']['alt']='2560×1600、150%DPI的最大化浅色Qt工作台，侧栏收起，输入与结果等宽，输入标题右侧显示0个字符，两框底边对齐。'
    images['m13-reading-choice-detail-figure']['width']='100%'
    images['m13-figure-m13-zhao-comparison-r2-light']['alt']='最大化浅色Qt工作台，输入春江花月夜与银行行走，第一处行已选hang2，第二处保留xing2，赵元任列字音同显，汉字24px和音标16px。'
    images['m13-figure-m13-unt-comparison-r2-light']['alt']='最大化浅色Qt工作台，输入春江花月夜与银行行走，第一处行已选hang2，第二处保留xing2，Unt严式字音同显，附加符号与连音线可见。'
    images['m13-figure-m13-stacked-kaiti-colors-export-r2-light']['assetId']='m13-layout-current-20261011'
    table=next(b for b in blocks if b.get('attrs',{}).get('id')=='m13-controls-table')
    index=next(i for i,r in enumerate(table['content']) if node_text(r['content'][0])=='行距')
    row={'type':'tableRow','content':[{'type':'tableCell','content':[paragraph(t)]} for t in ['等宽排版','默认关闭','开启后，各字音单元按当前结果最宽的字组统一宽度并居中，标点和儿化字组也参与对齐。字号、字体或读音变化时自动重算；再次点击恢复自然宽度。']]}
    if not any(node_text(r['content'][0])=='等宽排版' for r in table['content']):table['content'].insert(index+1,row)
    index=next(i for i,b in enumerate(blocks) if b.get('attrs',{}).get('id')=='m13-figure-m13-stacked-kaiti-colors-export-r2-light')
    if not any(b.get('attrs',{}).get('id')=='m13-figure-equal-width' for b in blocks):blocks[index+1:index+1]=[
      paragraph('点击等宽排版后，每个汉字及其音标共同占一个等宽单元，宽度取当前完整结果中最宽的字组。标点、原样字符和已组合的儿化字组也使用同样宽度。调整左侧字号、更换字体或选择读音时会自动重算，再次点击可恢复各字组的自然宽度。'),
      {'type':'image','attrs':{'id':'m13-figure-equal-width','assetId':'m13-equal-width-20261011','caption':'开启等宽排版：各字组和标点占相同宽度，汉字26px、音标20px；输入与结果两栏默认等宽，字符数显示在输入标题右侧。','alt':'最大化浅色Qt工作台，输入两行春江潮水连海平等文字，右侧所有字组按当前最宽单元对齐，左侧等宽排版按钮开启。','width':'100%','align':'center'}}
    ]
    asset_nodes={b['attrs']['assetId']:b['attrs'] for b in blocks if b['type']=='image'}
    assets={a['id']:a for a in project['assets']}
    evidence_relative=evidence.relative_to(ROOT).as_posix()+'/report.json'
    for capture in report['captures']:
        name=capture['name'];assert name in asset_nodes,name
        assert capture['image']==[2560,1600] and capture['dpr']==1.5 and capture['pageZoom']==1 and capture['maximized']
        data=Path(capture['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==capture['sha256']
        asset=assets.get(name)
        if asset is None:
            asset={'id':name,'path':'assets/software-only/screenshots/'+name+'.png','kind':'image','mime':'image/png','git':False,'distribution':'software-only'};project['assets'].append(asset)
        asset.update({'sha256':capture['sha256'],'width':2560,'height':1600,'caption':asset_nodes[name]['caption'],'sourceType':'实际Qt工作台截图','source':'Windows当前源码，浅色，最大化，侧栏收起；Qt DPR1.5（有效DPI150%）、页面100%；2561×1601取整时只裁右/下各1像素，未整体缩放。','capture':{'report':evidence_relative,'dpr':1.5,'pageZoom':1,'rawImage':capture['raw_image'],'maximized':True}})
        write_atomic(manual/asset['path'],data)
    exported=report['export'];asset=assets['m13-export-kaiti-independent-colors'];data=Path(exported['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==exported['sha256']
    asset.update({'sha256':exported['sha256'],'width':exported['image'][0],'height':exported['image'][1],'caption':asset_nodes[asset['id']]['caption'],'sourceType':'V3实际导出图','source':'当前Qt汉字转换结果真实导出的白底PNG，内置Doulos音标、棕色楷体粗体汉字；不含多音小三角。','capture':{'report':evidence_relative}});write_atomic(manual/asset['path'],data)
    descriptor=next(c for c in project['chapters'] if c['id']=='m13');search=[];sections=[];target=None
    for block in blocks:
        if block['type']=='heading':
            target=block['attrs']['id'];sections.append({'id':target,'title':node_text(block),'level':block['attrs'].get('level',2)})
        text=node_text(block)
        if text:
            entry={'chapterId':'m13','title':chapter['title'],'text':text}
            if target:entry['targetId']=target
            search.append(entry)
    descriptor['sections']=sections
    old_search=project['searchIndex'];first=next(i for i,e in enumerate(old_search) if e['chapterId']=='m13');project['searchIndex']=[e for e in old_search[:first] if e['chapterId']!='m13']+search+[e for e in old_search[first:] if e['chapterId']!='m13']
    write_atomic(manual/'chapters/m13.json',dump(chapter));write_atomic(manual/'project.json',dump(project))
    print(json.dumps({'screenshots':len(report['captures']),'png':exported['image'],'chapter_sha256':hashlib.sha256((manual/'chapters/m13.json').read_bytes()).hexdigest()},ensure_ascii=False))

if __name__=='__main__':main()
