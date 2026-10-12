"""Audit final package bytes, contained assets and private credential absence."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import zipfile
from PyInstaller.archive.readers import CArchiveReader

ROOT=Path(__file__).resolve().parents[2]

def main():
    spec=importlib.util.spec_from_file_location('feedback_native',ROOT/'desktop/src/ptb_desktop/feedback.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    admin=native.FeedbackClient(Path(os.environ['LOCALAPPDATA'])/'PhoneticToolbox/v3/feedback-manager',admin=True)
    token=admin.state.get('token','').encode()
    assert len(token)>=32,'A configured private admin profile is needed for the audit'
    exe=ROOT/'output/feedback-manager/1.0.1/反馈收件箱.exe'
    apk=ROOT/'output/feedback-manager/1.0.1/反馈收件箱.apk'
    reader=CArchiveReader(str(exe))
    names=list(reader.toc)
    assert not any(name.lower().endswith('icuuc.dll') for name in names)
    for name in names:
        if name=='PYZ.pyz' or name.endswith(('.js','.html','.css','.json','.txt','.pyc')):
            assert token not in reader.extract(name),'Private credential in EXE'
    forbidden=['identity.bin','feedback.sqlite','feedback-release.jks','signing-password.bin','private.json']
    assert not any(item in name for name in names for item in forbidden)
    avatar=(ROOT/'frontend/src/assets/feedback-avatar.jpg').read_bytes()
    avatars=[name for name in names if 'feedback-avatar-' in name and name.endswith('.jpg')]
    assert len(avatars)==1 and reader.extract(avatars[0])==avatar
    licenses=['Doulos-OFL.txt','Vue-LICENSE.txt','PyQt6-LICENSE.txt','PyQt6-WebEngine-LICENSE.txt']
    assert all('frontend\\'+name in names for name in licenses)
    with zipfile.ZipFile(apk) as archive:
        assert all('assets/'+name in archive.namelist() for name in licenses)
        avatars=[name for name in archive.namelist() if 'feedback-avatar-' in name and name.endswith('.jpg')]
        assert len(avatars)==1 and archive.read(avatars[0])==avatar
        assert not any(item in name for name in archive.namelist() for item in forbidden)
        for name in archive.namelist():assert token not in archive.read(name),'Private credential in APK'
    report={'success':True,'privateAdminCredentialAbsent':True,'privateFilesAbsent':True,'fullLicenseTextsPresent':True,'foreignIcuAbsent':True,'authorAvatarIdentical':True,'packages':{}}
    for path in [exe,apk]:
        size=path.stat().st_size;assert size<500_000_000
        report['packages'][path.name]={'bytes':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    destination=ROOT/'output/validation/feedback-manager/packages.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
