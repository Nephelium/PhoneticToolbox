"""Write reviewable private-staging metadata after local artifact checks."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    destination = Path.home() / 'Desktop/PhoneticToolbox-3.0.0-preview.1'
    files = json.loads((root / 'output/manual-work/delivery-hashes.json').read_text('utf-8-sig'))
    for item in files:
        file = destination / item['name']
        assert file.stat().st_size == item['size']
        with file.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    packages = []
    for item in files:
        if item['name'].endswith('portable.exe'):
            continue  # Self-extractor is a first-download asset; updates use ZIP.
        packages.append(dict(item, platform='windows-x64',
                             kind='portable' if item['name'].endswith('.zip') else 'installer',
                             url='https://www.phonetictoolbox.com/releases/windows-x64/preview/3.0.0-preview.1/'+item['name']))
    candidate = dict(schemaVersion='ptb-release/1', version='3.0.0-preview.1', channel='preview',
                     publishedAt='', notes='首版 Preview：应用内图文说明书、当前用户安装、双源更新和可配置内部结果保留。',
                     packages=packages)
    staging = dict(schema='ptb-private-staging/1', version=candidate['version'], public=False,
                   status='pending-author-license-and-corresponding-source', files=files,
                   blockers=['project total LICENSE choice', 'SoE implementation redistribution provenance',
                             'exact native corresponding-source and model-license completion'])
    for name, value in [('ptb-release.pending.json', candidate), ('staging.json', staging)]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf8')
    (destination / 'SHA256SUMS').write_text(''.join(f"{item['sha256']}  {item['name']}\n" for item in files), 'ascii')
    (destination / '使用方法.txt').write_text('''PhoneticToolbox 3.0.0-preview.1，Windows x64

免安装 EXE：打开 portable.exe，选择一个新的空目录并展开。以后直接运行该目录中的 PhoneticToolbox.exe，不需反复展开。没有卸载登记或系统快捷方式。
免安装 ZIP：完整解压后运行 PhoneticToolbox.exe，保留 _internal 目录。
当前用户安装：运行 setup.exe，默认安装到当前用户 C 盘应用目录，无需管理员权限。可选择创建桌面快捷方式。

完整程序展开约 5.28 GB，包含离线科学环境。一次只需要选择一种包装使用，其余下载包由作者管理。
设置与草稿位于固定用户目录，移动程序或升级后继续使用。卸载不删除研究数据。
未导出内部结果默认保留 30 天，成功导出的内部副本默认保留 7 天，可在设置中调整或关闭。录音、输入、标注、工程、正式导出及活动依赖受保护。

应用左下使用说明和各模块帮助打开应用内章节。生理参数合成章暂空，唇形视频待补录。
说明书作者编辑器单独位于 D:\\PhoneticToolbox\\PhoneticToolbox_v3\\tools\\manual-studio，双击打开说明书编辑器.cmd。
作者改稿后按 manual/README.md 重建阅读资源和应用。

本次是作者核对包。服务器目前只私有暂存，未公开下载或发布 latest.json，GitHub 尚未上传。
公众发布仍需完成项目总许可证、SoE 实现许可依据和对应源码交付。自然例音仅允许随软件分发，不能上传公开 GitHub。
详细验证见工程 docs/testing/2026-10-05-preview-release-report.md。
''', 'utf8')
    print(json.dumps({'destination': str(destination), 'verifiedArtifacts': len(files), 'public': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
