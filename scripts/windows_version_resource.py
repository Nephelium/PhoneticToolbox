"""Write Windows PE metadata from the frozen application version."""
import json
from pathlib import Path


def write_version_resource(snapshot):
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable,
        VarFileInfo, VarStruct, VSVersionInfo,
    )
    snapshot = Path(snapshot)
    version = json.loads((snapshot / 'release/version.json').read_text('utf8'))['app_version']
    numbers = tuple(map(int, version.split('-', 1)[0].split('.'))) + (0,)
    if len(numbers) != 4 or any(number < 0 or number > 65535 for number in numbers):
        raise ValueError('Application version cannot be represented in a Windows PE resource')
    resource = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers, mask=0x3f,
                          flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
        kids=[StringFileInfo([StringTable('040904B0', [
            StringStruct('CompanyName', 'PhoneticToolbox'),
            StringStruct('FileDescription', 'PhoneticToolbox'),
            StringStruct('FileVersion', version),
            StringStruct('InternalName', 'PhoneticToolbox'),
            StringStruct('OriginalFilename', 'PhoneticToolbox.exe'),
            StringStruct('ProductName', 'PhoneticToolbox'),
            StringStruct('ProductVersion', version),
        ])]), VarFileInfo([VarStruct('Translation', [0x0409, 1200])])],
    )
    target = snapshot / 'PhoneticToolbox-version.txt'
    target.write_text(str(resource), encoding='utf8', newline='\n')
    return target
