#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成汉字转国际音标网页
将Excel数据嵌入到HTML文件中，字体使用本地文件
"""

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "IPAtrans.xlsx"
OUTPUT_PATH = BASE_DIR / "ipa_converter.html"
SOURCE_HTML_PATH = BASE_DIR / "ipa_converter.html"

# 转换标准列名（调整顺序，UntPhesoca放最后）
STANDARDS = [
    'Standard Chinese (Beijing)',
    'Standard Chinese (Beijing)严',
    '胡裕树《现代汉语》',
    '黄伯荣、廖序东《现代汉语》',
    '钱乃荣《现代汉语》',
    '吴宗济',
    '赵元任《汉语口语语法》',
    '《汉语方音字汇》',
    'UntPhesoca宽',
    'UntPhesoca严'
]

def remove_zero_tone(ipa_str):
    if ipa_str is None:
        return ""
    ipa_str = str(ipa_str)
    if ipa_str.lower() == "nan":
        return ""
    result = str(ipa_str).replace('⁰', '').replace('˳', '')
    if result.endswith('0'):
        result = result[:-1]
    return result

def load_excel_data() -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_excel(EXCEL_PATH)
    columns_to_keep = ['汉字', '声调', '拼音'] + STANDARDS
    df = df[columns_to_keep]
    
    for idx, row in df.iterrows():
        if row['声调'] == 0:
            for col in STANDARDS:
                if pd.notna(df.at[idx, col]):
                    df.at[idx, col] = remove_zero_tone(df.at[idx, col])
    
    return df.to_dict('records')

def load_data_from_existing_html() -> list[dict[str, Any]]:
    html_text = SOURCE_HTML_PATH.read_text(encoding="utf-8")
    pattern = r"const\s+ipaData\s*=\s*(\[[\s\S]*?\]);"
    match = re.search(pattern, html_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("无法从现有 HTML 中提取 ipaData。")
    return json.loads(match.group(1))

def load_data() -> list[dict[str, Any]]:
    if EXCEL_PATH.exists():
        return load_excel_data()
    if SOURCE_HTML_PATH.exists():
        return load_data_from_existing_html()
    raise RuntimeError("未找到 IPA 数据源。")


def generate_html(data_json):
    """生成HTML内容"""
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>汉字国际音标转换器</title>
    <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
    <style>
        @font-face {{
            font-family: 'DoulosSIL';
            src: url('DoulosSIL-Regular.ttf') format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', '微软雅黑', sans-serif;
            background: radial-gradient(circle at 20% 20%, #94a3ff 0%, #7f8cff 25%, #6f7bf1 55%, #5b5bd6 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }}
        
        header h1 {{
            font-size: 2.5em;
            text-shadow: 0 8px 28px rgba(22, 26, 90, 0.45);
            margin-bottom: 10px;
        }}
        
        .controls {{
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(4px);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 16px 44px rgba(47, 57, 136, 0.22);
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            justify-content: center;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .control-group label {{
            font-weight: 600;
            color: #333;
            white-space: nowrap;
        }}
        
        select, input[type="range"] {{
            padding: 10px 15px;
            border: 1px solid #c6cef8;
            border-radius: 8px;
            font-size: 14px;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: inset 0 1px 2px rgba(32, 40, 112, 0.06);
        }}
        
        select {{
            min-width: 250px;
        }}
        
        input[type="range"] {{
            width: 100px;
            padding: 5px;
        }}
        
        select:hover, input[type="range"]:hover {{
            border-color: #7b88ef;
        }}
        
        select:focus, input[type="range"]:focus {{
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        }}
        
        .font-size-display {{
            min-width: 40px;
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }}
        
        .help-wrapper {{
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 4px;
        }}

        .help-icon {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #6476ef 0%, #8b67e8 100%);
            color: #fff;
            font-size: 16px;
            font-weight: 700;
            cursor: default;
            box-shadow: 0 8px 18px rgba(84, 94, 205, 0.35);
        }}

        .help-tooltip {{
            position: absolute;
            top: calc(100% + 10px);
            right: 0;
            width: 360px;
            background: rgba(20, 24, 54, 0.96);
            color: #f3f5ff;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 12px;
            line-height: 1.5;
            box-shadow: 0 12px 24px rgba(8, 11, 29, 0.35);
            opacity: 0;
            visibility: hidden;
            transform: translateY(-4px);
            transition: opacity 0.2s ease, transform 0.2s ease, visibility 0.2s ease;
            z-index: 100;
        }}

        .help-wrapper:hover .help-tooltip {{
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }}
        
        .toggle-btn {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        
        .toggle-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .toggle-btn.active {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .save-btn {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        
        .save-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4);
        }}

        .doc-help-btn {{
            padding: 10px 18px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 700;
        }}

        .doc-help-btn:hover {{
            background: #34c759;
            transform: translateY(-1px);
            box-shadow: 0 5px 16px rgba(36, 156, 85, 0.35);
        }}
        
        .format-btn {{
            width: 32px;
            height: 36px;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 700;
        }}
        
        .format-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .format-btn.active {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .format-btn.italic {{
            font-style: italic;
        }}
        
        .format-btn.underline {{
            text-decoration: underline;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            transition: all 0.3s ease;
        }}
        
        .main-content.vertical {{
            grid-template-columns: 1fr;
        }}
        
        .panel {{
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 18px;
            padding: 25px;
            box-shadow: 0 16px 42px rgba(47, 57, 136, 0.2);
            min-height: 400px;
        }}
        
        .main-content.vertical .panel {{
            min-height: 300px;
        }}
        
        .panel h2 {{
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .panel h2::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }}
        
        .input-area {{
            width: 100%;
            min-height: 350px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-family: 'SimSun', '宋体', serif;
            font-size: 28px;
            line-height: 1.8;
            resize: vertical;
            transition: border-color 0.3s ease;
        }}
        
        .main-content.vertical .input-area {{
            min-height: 200px;
        }}
        
        .input-area:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        }}
        
        .output-area {{
            min-height: 350px;
            padding: 15px;
            border: 1px solid #d9e0fb;
            border-radius: 10px;
            font-family: 'DoulosSIL', 'Doulos SIL', serif;
            font-size: 28px;
            line-height: 1.8;
            background: linear-gradient(180deg, #fbfcff 0%, #f4f7ff 100%);
            overflow-y: auto;
        }}
        
        .main-content.vertical .output-area {{
            min-height: 200px;
        }}
        
        .char-wrapper {{
            display: inline-block;
            cursor: pointer;
            padding: 2px 4px;
            border-radius: 4px;
            transition: all 0.2s ease;
            position: relative;
        }}
        
        .char-wrapper:hover,
        .char-wrapper.highlight {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        }}
        
        .char-wrapper.has-variants {{
            cursor: pointer;
        }}
        
        .char-wrapper.has-variants::after {{
            content: '▼';
            font-size: 8px;
            color: #667eea;
            position: absolute;
            bottom: -2px;
            right: 0;
            opacity: 0.7;
        }}
        
        /* 未知字符使用宋体 */
        .unknown-char {{
            font-family: 'SimSun', '宋体', serif;
        }}
        
        /* 带汉字显示模式 */
        .with-hanzi .char-wrapper {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            margin: 0 2px;
            vertical-align: bottom;
            gap: 0;
        }}

        .with-hanzi {{
            line-height: var(--hanzi-line-height, 1.8);
        }}
        
        .with-hanzi .ipa-text {{
            font-family: 'DoulosSIL', 'Doulos SIL', serif;
            color: #667eea;
            line-height: 1;
        }}
        
        .with-hanzi .hanzi-text {{
            font-family: 'SimSun', '宋体', serif;
            color: #333;
            line-height: 1;
            margin-top: var(--ipa-hanzi-gap, 0px);
        }}
        
        /* 拼音显示模式 */
        .pinyin-text {{
            font-family: 'Microsoft YaHei', sans-serif;
            color: #667eea;
        }}

        /* 多音字弹窗 */
        .variant-popup {{
            position: fixed;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 15px;
            z-index: 1000;
            min-width: 200px;
            max-width: 350px;
            display: none;
        }}
        
        .variant-popup.show {{
            display: block;
            animation: fadeIn 0.2s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .variant-popup h3 {{
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
            font-size: 16px;
        }}
        
        .variant-popup h3 .hanzi {{
            font-family: 'SimSun', '宋体', serif;
            color: #667eea;
            font-size: 24px;
        }}
        
        .variant-item {{
            padding: 10px 12px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
        }}
        
        .variant-item:hover {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        }}
        
        .variant-item.selected {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .variant-item .pinyin {{
            font-size: 14px;
            opacity: 0.8;
        }}
        
        .variant-item .ipa {{
            font-family: 'DoulosSIL', 'Doulos SIL', serif;
            font-size: 18px;
        }}
        
        .close-popup {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #999;
            transition: color 0.2s;
        }}
        
        .close-popup:hover {{
            color: #333;
        }}
        
        /* 响应式设计 */
        @media (max-width: 900px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8em;
            }}
            
            .controls {{
                flex-direction: column;
            }}
            
            select {{
                width: 100%;
            }}
        }}
        
        /* 加载提示 */
        .loading {{
            text-align: center;
            padding: 50px;
            color: #666;
            font-family: 'SimSun', '宋体', serif;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>汉字国际音标转换器</h1>
        </header>
        
        <div class="controls">
            <div class="control-group">
                <label for="standard">转换标准：</label>
                <select id="standard">
                    <option value="Standard Chinese (Beijing)">Standard Chinese (Beijing)</option>
                    <option value="Standard Chinese (Beijing)严" selected>Standard Chinese (Beijing)严</option>
                    <option value="胡裕树《现代汉语》">胡裕树《现代汉语》</option>
                    <option value="黄伯荣、廖序东《现代汉语》">黄伯荣、廖序东《现代汉语》</option>
                    <option value="钱乃荣《现代汉语》">钱乃荣《现代汉语》</option>
                    <option value="吴宗济">吴宗济</option>
                    <option value="赵元任《汉语口语语法》">赵元任《汉语口语语法》</option>
                    <option value="《汉语方音字汇》">《汉语方音字汇》</option>
                    <option value="UntPhesoca宽">UntPhesoca宽</option>
                    <option value="UntPhesoca严">UntPhesoca严</option>
                    <option value="汉语拼音">汉语拼音</option>
                </select>
            </div>
            <div class="control-group">
                <label>汉字字号：</label>
                <input type="range" id="fontSizeSlider" min="16" max="72" value="24">
                <span class="font-size-display" id="fontSizeDisplay">24px</span>
            </div>
            <div class="control-group">
                <label>音标字号：</label>
                <input type="range" id="ipaFontSizeSlider" min="12" max="48" value="16">
                <span class="font-size-display" id="ipaFontSizeDisplay">16px</span>
            </div>
            <div class="control-group">
                <label>文字音标距离：</label>
                <input type="range" id="ipaHanziGapSlider" min="-12" max="20" value="0">
                <span class="font-size-display" id="ipaHanziGapDisplay">0px</span>
            </div>
            <div class="control-group">
                <label>文字+音标行间距：</label>
                <input type="range" id="lineHeightSlider" min="8" max="30" value="18">
                <span class="font-size-display" id="lineHeightDisplay">1.8</span>
            </div>
            <div class="control-group">
                <button class="format-btn" id="boldBtn" title="加粗">B</button>
                <button class="format-btn italic" id="italicBtn" title="斜体">I</button>
                <button class="format-btn underline" id="underlineBtn" title="下划线">U</button>
                <button class="toggle-btn active" id="toggleMode">仅显示音标</button>
                <button class="toggle-btn" id="toggleLayout">上下排布</button>
                <button class="save-btn" id="saveImage">保存为图片</button>
                <button class="doc-help-btn" id="docHelpBtn">帮助</button>
                <div class="help-wrapper" aria-label="显示说明">
                    <button class="help-icon" type="button">?</button>
                    <div class="help-tooltip">字号、加粗、斜体、下划线、文字音标距离、行间距仅对“国际音标”部分有效；文字音标距离支持负值，调到负值可让音标与文字更贴近。</div>
                </div>
            </div>
        </div>
        
        <div class="main-content" id="mainContent">
            <div class="panel">
                <h2>输入汉字</h2>
                <textarea class="input-area" id="inputText" placeholder="在此输入汉字..."></textarea>
            </div>
            <div class="panel" id="outputPanel">
                <h2>国际音标</h2>
                <div class="output-area" id="outputArea">
                    <span class="loading">等待输入</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="variant-popup" id="variantPopup">
        <button class="close-popup" onclick="closePopup()">&times;</button>
        <h3>选择读音：<span class="hanzi" id="popupHanzi"></span></h3>
        <div id="variantList"></div>
    </div>

    <script>
        // 数据
        const ipaData = {json.dumps(data_json, ensure_ascii=False)};
        
        // 构建汉字索引
        const hanziIndex = {{}};
        ipaData.forEach((item, idx) => {{
            const char = item['汉字'];
            if (!hanziIndex[char]) {{
                hanziIndex[char] = [];
            }}
            hanziIndex[char].push(item);
        }});
        
        // 声调符号映射
        const toneMarks = {{
            'a': ['ā', 'á', 'ǎ', 'à', 'a'],
            'e': ['ē', 'é', 'ě', 'è', 'e'],
            'i': ['ī', 'í', 'ǐ', 'ì', 'i'],
            'o': ['ō', 'ó', 'ǒ', 'ò', 'o'],
            'u': ['ū', 'ú', 'ǔ', 'ù', 'u'],
            'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ', 'ü'],
            'v': ['ǖ', 'ǘ', 'ǚ', 'ǜ', 'ü']
        }};
        
        // 将拼音转换为带声调符号的形式
        function addToneMark(pinyin, tone) {{
            if (!pinyin) return pinyin;
            if (tone < 1 || tone > 5 || tone === 0) return pinyin;
            
            const vowels = 'aeiouüv';
            let pinyinLower = pinyin.toLowerCase();
            let targetIndex = -1;
            
            for (let i = 0; i < pinyinLower.length; i++) {{
                if (pinyinLower[i] === 'a' || pinyinLower[i] === 'e') {{
                    targetIndex = i;
                    break;
                }}
            }}
            
            if (targetIndex === -1) {{
                const ouIndex = pinyinLower.indexOf('ou');
                if (ouIndex !== -1) {{
                    targetIndex = ouIndex;
                }}
            }}
            
            if (targetIndex === -1) {{
                for (let i = pinyinLower.length - 1; i >= 0; i--) {{
                    if (vowels.includes(pinyinLower[i])) {{
                        targetIndex = i;
                        break;
                    }}
                }}
            }}
            
            if (targetIndex === -1) return pinyin;
            
            const targetVowel = pinyinLower[targetIndex];
            const toneIndex = tone - 1;
            
            if (toneMarks[targetVowel]) {{
                const result = pinyin.substring(0, targetIndex) + 
                               toneMarks[targetVowel][toneIndex] + 
                               pinyin.substring(targetIndex + 1);
                return result;
            }}
            
            return pinyin;
        }}
        
        // 状态
        let currentStandard = 'Standard Chinese (Beijing)严';
        let showWithHanzi = true;
        let verticalLayout = false;
        let hanziFontSize = 24;
        let ipaFontSize = 16;
        let ipaFontSizeUserSet = false;  // 用户是否手动调整过音标字号
        let ipaHanziGap = 0;
        let hanziLineHeight = 1.8;
        let selectedVariants = {{}};
        let currentHighlightIndex = -1;
        let hanziBold = false;
        let hanziItalic = false;
        let hanziUnderline = false;
        
        // DOM元素
        const inputText = document.getElementById('inputText');
        const outputArea = document.getElementById('outputArea');
        const outputPanel = document.getElementById('outputPanel');
        const standardSelect = document.getElementById('standard');
        const toggleBtn = document.getElementById('toggleMode');
        const toggleLayoutBtn = document.getElementById('toggleLayout');
        const saveImageBtn = document.getElementById('saveImage');
        const docHelpBtn = document.getElementById('docHelpBtn');
        const mainContent = document.getElementById('mainContent');
        const variantPopup = document.getElementById('variantPopup');
        const popupHanzi = document.getElementById('popupHanzi');
        const variantList = document.getElementById('variantList');
        const fontSizeSlider = document.getElementById('fontSizeSlider');
        const fontSizeDisplay = document.getElementById('fontSizeDisplay');
        const ipaFontSizeSlider = document.getElementById('ipaFontSizeSlider');
        const ipaFontSizeDisplay = document.getElementById('ipaFontSizeDisplay');
        const ipaHanziGapSlider = document.getElementById('ipaHanziGapSlider');
        const ipaHanziGapDisplay = document.getElementById('ipaHanziGapDisplay');
        const lineHeightSlider = document.getElementById('lineHeightSlider');
        const lineHeightDisplay = document.getElementById('lineHeightDisplay');
        const boldBtn = document.getElementById('boldBtn');
        const italicBtn = document.getElementById('italicBtn');
        const underlineBtn = document.getElementById('underlineBtn');
        const HELP_DOC_URL = '../../../../Phonetic_Export/index.html#s1765796015349';
        
        // 获取汉字样式字符串
        function getHanziStyle() {{
            let style = `font-size: ${{hanziFontSize}}px`;
            if (hanziBold) style += '; font-weight: bold';
            if (hanziItalic) style += '; font-style: italic';
            if (hanziUnderline) style += '; text-decoration: underline';
            return style;
        }}

        // 获取显示文本
        function getDisplayText(entry) {{
            if (currentStandard === '汉语拼音') {{
                const pinyin = entry['拼音'] || '';
                const tone = entry['声调'];
                if (tone === 0) return pinyin;
                return addToneMark(pinyin, tone);
            }} else {{
                return entry[currentStandard] || '?';
            }}
        }}
        
        // 转换函数
        function convertToIPA(text) {{
            if (!text.trim()) {{
                return '<span class="loading">等待输入</span>';
            }}
            
            let result = '';
            const chars = [...text];
            const isPinyin = currentStandard === '汉语拼音';
            
            chars.forEach((char, index) => {{
                const entries = hanziIndex[char];
                
                if (entries && entries.length > 0) {{
                    const key = `${{char}}_${{index}}`;
                    let selectedEntry = selectedVariants[key];
                    
                    if (!selectedEntry) {{
                        selectedEntry = entries[0];
                    }}
                    
                    const displayText = getDisplayText(selectedEntry);
                    const hasVariants = entries.length > 1;
                    const variantClass = hasVariants ? 'has-variants' : '';
                    const textClass = isPinyin ? 'pinyin-text' : 'ipa-text';
                    
                    if (showWithHanzi) {{
                        result += `<span class="char-wrapper ${{variantClass}}" data-index="${{index}}" data-char="${{char}}" onmouseenter="highlightInput(${{index}})" onmouseleave="unhighlightInput()" onclick="showVariants('${{char}}', ${{index}}, event)">
                            <span class="${{textClass}}" style="font-size: ${{ipaFontSize}}px">${{displayText}}</span>
                            <span class="hanzi-text" style="${{getHanziStyle()}}">${{char}}</span>
                        </span>`;
                    }} else {{
                        const ipaOnlyFontSize = ipaFontSizeUserSet ? ipaFontSize : 28;
                        result += `<span class="char-wrapper ${{variantClass}}" data-index="${{index}}" data-char="${{char}}" onmouseenter="highlightInput(${{index}})" onmouseleave="unhighlightInput()" onclick="showVariants('${{char}}', ${{index}}, event)" style="font-size: ${{ipaOnlyFontSize}}px">${{displayText}}</span>`;
                    }}
                }} else if (char === '\\n') {{
                    result += '<br>';
                }} else if (char.match(/\\s/)) {{
                    result += char;
                }} else {{
                    // 非汉字字符（标点符号等）：音标部分留空，只显示原字符
                    if (showWithHanzi) {{
                        result += `<span class="char-wrapper" data-index="${{index}}">
                            <span class="ipa-text" style="font-size: ${{ipaFontSize}}px">&nbsp;</span>
                            <span class="hanzi-text" style="${{getHanziStyle()}}">${{char}}</span>
                        </span>`;
                    }} else {{
                        const ipaOnlyFontSize = ipaFontSizeUserSet ? ipaFontSize : 28;
                        result += `<span class="char-wrapper unknown-char" data-index="${{index}}" style="font-size: ${{ipaOnlyFontSize}}px">${{char}}</span>`;
                    }}
                }}
            }});
            
            return result;
        }}
        
        // 更新输出
        function updateOutput() {{
            const text = inputText.value;
            outputArea.innerHTML = convertToIPA(text);
            outputArea.style.setProperty('--ipa-hanzi-gap', `${{ipaHanziGap}}px`);
            outputArea.style.setProperty('--hanzi-line-height', `${{hanziLineHeight}}`);
            if (showWithHanzi) {{
                outputArea.classList.add('with-hanzi');
            }} else {{
                outputArea.classList.remove('with-hanzi');
            }}
        }}
        
        // 高亮输入区对应字符
        function highlightInput(index) {{
            currentHighlightIndex = index;
            document.querySelectorAll('.char-wrapper').forEach((el, i) => {{
                if (parseInt(el.dataset.index) === index) {{
                    el.classList.add('highlight');
                }} else {{
                    el.classList.remove('highlight');
                }}
            }});
        }}
        
        function unhighlightInput() {{
            currentHighlightIndex = -1;
            document.querySelectorAll('.char-wrapper').forEach(el => {{
                el.classList.remove('highlight');
            }});
        }}
        
        // 显示多音字选择
        function showVariants(char, index, event) {{
            event.stopPropagation();
            
            const entries = hanziIndex[char];
            if (!entries || entries.length <= 1) return;
            
            popupHanzi.textContent = char;
            
            const key = `${{char}}_${{index}}`;
            const currentSelected = selectedVariants[key];
            
            let html = '';
            entries.forEach((entry, i) => {{
                const displayText = getDisplayText(entry);
                const pinyin = entry['拼音'] || '';
                const tone = entry['声调'];
                const toneDisplay = tone === 0 ? '(轻声)' : tone;
                const isSelected = currentSelected && 
                    currentSelected['拼音'] === entry['拼音'] && 
                    currentSelected['声调'] === entry['声调'];
                const selectedClass = isSelected ? 'selected' : '';
                
                html += `<div class="variant-item ${{selectedClass}}" onclick="selectVariant('${{char}}', ${{index}}, ${{i}})">
                    <span class="pinyin">${{pinyin}}${{toneDisplay}}</span>
                    <span class="ipa">${{displayText}}</span>
                </div>`;
            }});
            
            variantList.innerHTML = html;
            
            const clickX = event.clientX;
            const clickY = event.clientY;
            
            variantPopup.style.left = Math.min(clickX, window.innerWidth - 260) + 'px';
            variantPopup.style.top = Math.min(clickY + 10, window.innerHeight - 300) + 'px';
            variantPopup.classList.add('show');
        }}
        
        // 选择多音字读音
        function selectVariant(char, index, variantIndex) {{
            const entries = hanziIndex[char];
            const key = `${{char}}_${{index}}`;
            selectedVariants[key] = entries[variantIndex];
            closePopup();
            updateOutput();
        }}
        
        // 关闭弹窗
        function closePopup() {{
            variantPopup.classList.remove('show');
        }}
        
        // 保存为图片
        async function saveAsImage() {{
            try {{
                saveImageBtn.textContent = '正在生成...';
                saveImageBtn.disabled = true;
                
                const canvas = await html2canvas(outputPanel, {{
                    scale: 3,
                    backgroundColor: '#ffffff',
                    useCORS: true,
                    logging: false
                }});
                
                const link = document.createElement('a');
                link.download = 'ipa_output_' + new Date().getTime() + '.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
                
                saveImageBtn.textContent = '保存为图片';
                saveImageBtn.disabled = false;
            }} catch (error) {{
                console.error('保存图片失败:', error);
                alert('保存图片失败，请重试');
                saveImageBtn.textContent = '保存为图片';
                saveImageBtn.disabled = false;
            }}
        }}
        
        // 事件监听
        inputText.addEventListener('input', updateOutput);
        
        standardSelect.addEventListener('change', (e) => {{
            currentStandard = e.target.value;
            updateOutput();
        }});
        
        toggleBtn.addEventListener('click', () => {{
            showWithHanzi = !showWithHanzi;
            toggleBtn.classList.toggle('active', showWithHanzi);
            toggleBtn.textContent = showWithHanzi ? '仅显示音标' : '显示文字音标';
            updateOutput();
        }});
        
        toggleLayoutBtn.addEventListener('click', () => {{
            verticalLayout = !verticalLayout;
            toggleLayoutBtn.classList.toggle('active', verticalLayout);
            toggleLayoutBtn.textContent = verticalLayout ? '左右排布' : '上下排布';
            mainContent.classList.toggle('vertical', verticalLayout);
        }});
        
        boldBtn.addEventListener('click', () => {{
            hanziBold = !hanziBold;
            boldBtn.classList.toggle('active', hanziBold);
            updateOutput();
        }});
        
        italicBtn.addEventListener('click', () => {{
            hanziItalic = !hanziItalic;
            italicBtn.classList.toggle('active', hanziItalic);
            updateOutput();
        }});
        
        underlineBtn.addEventListener('click', () => {{
            hanziUnderline = !hanziUnderline;
            underlineBtn.classList.toggle('active', hanziUnderline);
            updateOutput();
        }});
        
        saveImageBtn.addEventListener('click', saveAsImage);
        docHelpBtn.addEventListener('click', () => {{
            window.open(HELP_DOC_URL, '_blank');
        }});
        
        fontSizeSlider.addEventListener('input', (e) => {{
            hanziFontSize = parseInt(e.target.value);
            fontSizeDisplay.textContent = hanziFontSize + 'px';
            updateOutput();
        }});
        
        ipaFontSizeSlider.addEventListener('input', (e) => {{
            ipaFontSize = parseInt(e.target.value);
            ipaFontSizeUserSet = true;  // 用户手动调整过
            ipaFontSizeDisplay.textContent = ipaFontSize + 'px';
            updateOutput();
        }});

        ipaHanziGapSlider.addEventListener('input', (e) => {{
            ipaHanziGap = parseInt(e.target.value);
            ipaHanziGapDisplay.textContent = ipaHanziGap + 'px';
            updateOutput();
        }});

        lineHeightSlider.addEventListener('input', (e) => {{
            hanziLineHeight = parseInt(e.target.value) / 10;
            lineHeightDisplay.textContent = hanziLineHeight.toFixed(1);
            updateOutput();
        }});
        
        // 点击其他地方关闭弹窗
        document.addEventListener('click', (e) => {{
            if (!variantPopup.contains(e.target) && !e.target.closest('.char-wrapper')) {{
                closePopup();
            }}
        }});
        
        // 初始化
        updateOutput();
    </script>
</body>
</html>'''
    return html

def main():
    print("正在加载IPA数据...")
    data = load_data()
    print(f"加载了 {len(data)} 条记录")
    
    print("正在生成HTML文件...")
    html_content = generate_html(data)
    
    OUTPUT_PATH.write_text(html_content, encoding='utf-8')
    
    print(f"HTML文件已生成: {OUTPUT_PATH}")
    print(f"文件大小: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print("\n注意：请确保字体文件 DoulosSIL-Regular.ttf 与 HTML 文件在同一目录下")

if __name__ == '__main__':
    main()
