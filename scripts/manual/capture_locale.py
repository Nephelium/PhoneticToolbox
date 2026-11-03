"""Locale helpers for isolated manual capture recipes, never product translation."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


class CaptureLocale:
    def __init__(self, locale: str, namespaces: tuple[str, ...] = (), *, catalogue_root: Path | None = None):
        if locale not in ('zh-CN', 'en', 'ja'):
            raise ValueError('Unsupported capture locale')
        self.locale = locale
        self.root = catalogue_root or ROOT / 'frontend/src/i18n/locales'
        self.namespaces = namespaces
        self.source, self.target, self.origins = {}, {}, {}
        for file in sorted((self.root / 'zh-CN').glob('*.json')):
            source = json.loads(file.read_text('utf-8'))
            translated = self.root / locale / file.name
            target = json.loads(translated.read_text('utf-8')) if translated.is_file() else {}
            for key, text in source.items():
                self.source[key] = text
                self.origins[key] = file.stem
            self.target.update(target)

    def key(self, key: str, values: dict | None = None) -> str:
        text = self.target.get(key)
        if not isinstance(text, str):
            raise ValueError('Missing capture UI translation: ' + key)
        values = values or {}

        def replace(match):
            name = match.group(1)
            if name not in values:
                raise ValueError('Missing capture UI placeholder: ' + name)
            return str(values[name])
        return re.sub(r'\{([A-Za-z][A-Za-z0-9_]*)\}', replace, text)

    def label(self, chinese: str, values: dict | None = None) -> str:
        """Adapt a legacy recipe label only when its catalog mapping is unique.

        File names, IPA, dataset values and paper text must never pass here.
        Prefer explicit key() calls or stable DOM selectors for new recipes.
        """
        if self.locale == 'zh-CN' or chinese in {'SVG', 'PNG', 'CSV', 'JSON', 'WAV', 'TextGrid', 'IPA', 'MFA', 'PDF', 'EXE', 'ZIP', 'F0', 'CQ', 'SQ'}:
            return chinese
        keys = [key for key, source in self.source.items() if source == chinese]
        preferred = [key for key in keys if self.origins[key] in self.namespaces]
        if preferred:
            keys = preferred
        texts = {self.key(key, values) for key in keys}
        if len(texts) != 1:
            raise ValueError(f'Capture label needs an explicit key ({len(texts)} candidates): {chinese}')
        return next(iter(texts))

    def prepare_qt(self, js, until):
        """Use the real visible language control in this recipe-owned host."""
        code = json.dumps(self.locale)
        until('!!document.querySelector(".language-switch select:not(:disabled)")')
        changed = js('(()=>{const e=document.querySelector(".language-switch select:not(:disabled)");'
                     'if(!e||![...e.options].some(o=>o.value===' + code + '))return false;'
                     'e.value=' + code + ';e.dispatchEvent(new Event("change",{bubbles:true}));return true;})()')
        if not changed:
            raise RuntimeError('Capture host has no matching language option')
        until('document.documentElement.lang===' + code + '&&document.documentElement.dataset.fontLocale===' + code + '&&!document.querySelector(".language-switch select:disabled")&&document.fonts.status==="loaded"')

    def metrics(self, js) -> dict:
        result = js('({locale:document.documentElement.lang,fontLocale:document.documentElement.dataset.fontLocale,dpr:devicePixelRatio,theme:document.documentElement.dataset.theme,'
                    'pageZoom:Number(getComputedStyle(document.documentElement).zoom)||1,fonts:document.fonts.status,fontFamily:getComputedStyle(document.querySelector(".language-atlas")||document.body).fontFamily})')
        if not isinstance(result, dict) or result.get('locale') != self.locale or result.get('fontLocale') != self.locale:
            raise RuntimeError('Capture locale has not finished switching')
        if result.get('dpr') != 1.5 or result.get('pageZoom') != 1 or result.get('theme') != 'light' or result.get('fonts') != 'loaded':
            raise RuntimeError('Capture host does not match approved DPR/zoom/theme/font requirements')
        return result
