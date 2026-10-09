"""Render distributed PDFs with the actual runtime, without installing PDF tools."""
import json
from pathlib import Path
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtPdf import QPdfDocument

root=Path(__file__).resolve().parents[1]/'output/paper-reading/2610.00735v1'
app=QApplication(['M18-PDF-QA'])
report={}
for name in ('original','translation'):
    doc=QPdfDocument(None);doc.load(str(root/(name+'.pdf')))
    assert doc.status()==QPdfDocument.Status.Ready
    folder=root/(name+'-pages');folder.mkdir(exist_ok=True)
    sheet=QImage(1250,330*((doc.pageCount()+3)//4),QImage.Format.Format_RGB32);sheet.fill(QColor('lightgray'))
    painter=QPainter(sheet);texts=[]
    for index in range(doc.pageCount()):
        size=doc.pagePointSize(index);im=doc.render(index,QSize(1224,round(1224*size.height()/size.width())))
        assert not im.isNull()
        white=QImage(im.size(),QImage.Format.Format_RGB32);white.fill(QColor('white'))
        flatten=QPainter(white);flatten.drawImage(0,0,im);flatten.end();im=white
        im.save(str(folder/f'{index+1:02d}.png'))
        painter.drawImage((index%4)*312,(index//4)*330,im.scaled(300,310,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        painter.drawText((index%4)*312+10,(index//4)*330+325,f'{name} {index+1}')
        texts.append(doc.getAllText(index).text())
    painter.end();sheet.save(str(root/(name+'-contact.png')))
    report[name]={'pages':doc.pageCount(),'characters':sum(map(len,texts))}
    (root/(name+'-text.txt')).write_text('\n\n'.join(texts),'utf8');doc.close()
(root/'pdf-verification.json').write_text(json.dumps(report,indent=2),'utf8');print(report)
