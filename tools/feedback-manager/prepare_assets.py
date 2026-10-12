"""Include the existing UI icon and complete bundled component notices."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
def prepare():
    web=ROOT/'output/build-feedback-manager/web'
    for source,name in [(ROOT/'frontend/src/assets/Doulos-OFL.txt','Doulos-OFL.txt'),(ROOT/'frontend/node_modules/vue/LICENSE','Vue-LICENSE.txt'),(ROOT/'third_party/licenses/p01/PYQT6-LICENSE.txt','PyQt6-LICENSE.txt'),(ROOT/'third_party/licenses/p01/PYQT6-WEBENGINE-LICENSE.txt','PyQt6-WebEngine-LICENSE.txt')]:
        if not source.is_file():raise RuntimeError('Missing component notice or icon: '+str(source))
        shutil.copyfile(source,web/name)
    from PIL import Image,ImageDraw
    image=Image.new('RGBA',(256,256),(0,0,0,0));draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((0,0,255,255),radius=48,fill='#00787e')
    draw.rounded_rectangle((48,62,208,184),radius=14,fill='white')
    draw.polygon([(66,173),(66,214),(118,179)],fill='white')
    draw.rounded_rectangle((78,100,177,112),radius=5,fill='#00787e')
    draw.rounded_rectangle((78,136,156,148),radius=5,fill='#00787e')
    image.save(web/'favicon.ico',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
if __name__=='__main__':prepare()
