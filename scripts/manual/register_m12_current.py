"""Register reviewed M12 current screenshots without deleting old formal media."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil
from core import dump
ROOT=Path(__file__).resolve().parents[2]
OLD={
 'overview':('m12-figure-m12-overview-r2-light','当前三层工作区。顶部提供吸附与关联、强度贴合、搜索替换和音标键盘，关闭工具弹窗后图下不保留参数行。图中音频为合成立体声演示。'),
 'association':('m12-figure-association-current-light','多个候选时，明确选择本次使用的TextGrid，再点击打开标注。'),
 'association-detail':('m12-association-current-light-detail-figure','关联弹窗局部：先选正确标注文件，再点击打开标注。'),
 'create-layers':('m12-figure-create-layers-current-light','创建标注层时可同时填写音节、音素和声调三个不同名称，第三层可留空。已有层保留。'),
 'create-layers-detail':('m12-create-layers-current-light-detail-figure','创建层弹窗局部：两个主要编辑层和一个可选第三层。'),
 'dictionary-wordlist':('m12-figure-dictionary-wordlist-current-light','词典、词表、顺序开关、首次音素切分和下一音节仍位于图窗下方；本例已粘贴四个演示条目并开启顺序标注。'),
 'sequence-start':('m12-figure-sequence-start-current-light','双击后显示待定起点，等待终点建立当前条目。图中使用合成演示音频，真实标注需结合录音定位。'),
 'sequence-phones':('m12-figure-sequence-phones-current-light','建立zhe4后在音素层内部双击，形成zh/e4两个区间。图中为合成演示，实际切分需试听检查。'),
 'reference-unpaired-lip':('m12-figure-reference-unpaired-lip-current-light','选择同一时间域的参考示例，在0.5–1秒区间之内复用。此例未关联唇形曲线。'),
 'save-success':('m12-figure-save-success-current-light','以_手册示例后缀保存成功，当前TextGrid和目标名一致，中文与IPA文字已写入示例文件。'),
 'save-error':('m12-figure-save-error-current-light','后缀包含路径分隔符时显示文件后缀无效，当前编辑仍保留。改为有效后缀后可重试。'),
 'close-protection':('m12-figure-close-protection-current-light','关闭带未保存修改的标签时，选择取消、放弃或保存后关闭；保存失败时保留编辑。'),
}
NEW={
 'links':('m12-three-tier-linking','吸附与关联弹窗分别控制边界吸附和三个层对的关联，关闭后设置继续生效。'),
 'intensity':('m12-intensity-fit','从顶部打开强度贴合，明确选择声道后设置范围和内收/外扩。本例选择第2声道，未选声道时不能执行。'),
 'search':('m12-search-locate','搜索替换弹窗显示当前位置与匹配数，支持上下项和Enter/Shift＋Enter定位，搜索内容在关闭后保留。'),
 'replace-confirm':('m12-search-replace','全部替换前核对匹配数量和替换文字；取消会返回原搜索弹窗，保留查询与替换内容。'),
 'keyboard':('m12-ipa-keyboard','从顶部音标键盘按钮打开紧凑IPA表。选中标注后可插入符号，标题拖动位置，右下角调整大小。'),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args();rp=a.report.resolve();r=json.loads(rp.read_text('utf-8'));assert r['success'] and len(r['captures'])==17
 project_path=ROOT/'manual/project.json';chapter_path=ROOT/'manual/chapters/m12.json';project=json.loads(project_path.read_text('utf-8'));chapter=json.loads(chapter_path.read_text('utf-8'));content=chapter['body']['content'];assets={a['id']:a for a in project['assets']};template=next(n for n in content if n['type']=='image');by_id={n.get('attrs',{}).get('id'):n for n in content}
 for c in r['captures']:
  name=c['name'];src=rp.parent/(name+'.png');h=hashlib.sha256(src.read_bytes()).hexdigest();assert h==c['sha256'];assert c['dpr']==1.5 and c['pageZoom']==1 and c['maximized'] and c['theme']=='light' and c['navigationCollapsed']
  if name not in OLD:assert c['image']==[2560,1600]
  if 'crop' not in c:assert c['image']==[2560,1600]
  aid='m12-current-'+name+'-'+h[:8];relative='assets/software-only/m12-current-20261021/'+aid+'.png';target=ROOT/'manual'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==h
  else:shutil.copyfile(src,target)
  key,caption=(OLD|NEW)[name];meta={k:v for k,v in c.items() if k!='file'}
  assets[aid]=dict(id=aid,path=relative,kind='image',mime='image/png',sha256=h,width=c['image'][0],height=c['image'][1],caption=caption,git=False,distribution='software-only',sourceType='实际应用截图',source='Windows当前Qt源码，隔离合成立体声与TextGrid演示，浅色、原生DPR1.5、页面100%、最大化，操作截图收起导航；局部图仅从原始截图裁剪。',capture=dict(report=rp.relative_to(ROOT).as_posix(),**meta))
  if name in OLD:node=by_id[key]
  else:
   node=copy.deepcopy(template);node['attrs']['id']='m12-current-'+name+'-figure';start=next(i for i,n in enumerate(content) if n['type']=='heading' and n['attrs']['id']==key);end=next((i for i in range(start+1,len(content)) if content[i]['type']=='heading'),len(content));content.insert(end,node)
  node['attrs'].update(assetId=aid,caption=caption,alt=caption)
 # Keep stable heading IDs and place the current entry instructions before steps.
 def paragraph(text):return dict(type='paragraph',content=[dict(type='text',text=text)])
 start=next(i for i,n in enumerate(content) if n['type']=='heading' and n['attrs']['id']=='m12-search-workflow');content.insert(start+1,paragraph('点击最上方的搜索替换按钮打开弹窗。关闭或按Esc后不占图下空间，查询和替换文字保留。重新打开时会按当前标注重新计算匹配，适用于编辑或撤销后继续查找。'))
 for node in content:
  if node['type']=='table' and node['attrs']['id']=='m12-controls-table':
   for row in [['吸附与关联','打开层间控制弹窗','可切换吸附与各层关联','关闭后参数继续生效'],['强度贴合','打开范围与声道弹窗','明确选择声道后执行','检查选区、内收与声道'],['搜索替换','打开词层搜索与替换','需已加载标注层','全部替换另行确认'],['音标键盘','打开浮动IPA/extIPA/VoQS表','选中区间后插入','支持移动与调整大小']]:
    node['content'].append(dict(type='tableRow',content=[dict(type='tableCell',content=[paragraph(t)]) for t in row]))
 def edit(n):
  if isinstance(n,dict):
   if n.get('type')=='text':
    n['text']=n['text'].replace('：波形、语谱图、标注层共用时间轴，下面有选中区间文本、词典词表及搜索替换。吸附与关联、强度贴合从顶部按钮打开弹窗，关闭后不占图下空间。','：波形、语谱图、标注层共用时间轴。图下保留选中文字、词典和词表；吸附与关联、强度贴合、搜索替换以及音标键盘均从最上方按钮打开。关闭弹窗后不占图下空间。').replace('选中区间后点音标键盘','选中区间后点最上方的音标键盘')
   for v in n.values():edit(v)
  elif isinstance(n,list):
   for v in n:edit(v)
 edit(chapter);project['assets']=list(assets.values());chapter['status']='reviewed';entry=next(c for c in project['chapters'] if c['id']=='m12');entry['status']='reviewed';entry['sections']=[dict(id=n['attrs']['id'],title=''.join(t.get('text','') for t in n.get('content',[])),level=n['attrs']['level']) for n in content if n['type']=='heading' and n['attrs']['level']>=2]
 chapter_path.write_bytes(dump(chapter));project_path.write_bytes(dump(project));print(json.dumps(dict(registered=17,replaced=12,added=5,images=sum(n['type']=='image' for n in content))))
if __name__=='__main__':main()
