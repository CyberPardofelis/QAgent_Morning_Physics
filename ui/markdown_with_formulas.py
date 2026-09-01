"""
v7 markdown → HTML with formulas
================================

替代 v3 时代 QWebEngineView + KaTeX 实时渲染,改用:
- markdown lib 转 HTML
- FormulaRenderer 离线渲染公式为 PNG
- QTextBrowser 渲染 HTML(<img> 加载 PNG)

设计要点:
- 公式原文作为 caption / inline-source 显示在图片旁边,方便用户复制
- 公式渲染失败 → fallback 显示 LaTeX 原文(红色等宽)
- 流式 chunks:每次重 setHtml 整段(公式已缓存,<10ms)
- 代码块(```code```)和 inline code(`code`)被保护,不被公式提取破坏
"""
from __future__ import annotations
import html as _html
import re
from typing import List, Tuple

import markdown

from ui.formula_renderer import get_formula_renderer


# ============================================================
# 公式提取 + 渲染
# ============================================================

# 公式占位符:用 \x00 (NUL) 包裹,markdown lib 不会触碰
_FORMULA_PLACEHOLDER = '\x00FORMULA{}\x00'
_CODE_PLACEHOLDER = '\x00CODE{}\x00'

# display 公式 $$...$$ 优先(必须在 inline 之前)
_DISPLAY_FORMULA = re.compile(r'\$\$([^\$]+?)\$\$', re.DOTALL)
# inline 公式 $...$(前后不是 $)
_INLINE_FORMULA = re.compile(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)')

# 代码块(```...```)和 inline code (`...`)保护 — 公式提取不进入
_CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)
_INLINE_CODE = re.compile(r'`[^`\n]+`')


def _escape_latex_for_caption(latex: str) -> str:
    """escape LaTeX 原文做 caption 文本(防 XSS)"""
    return _html.escape(latex)


def _render_formula_html(latex: str, is_display: bool) -> str:
    """
    渲染单个公式为 HTML 片段。
    - 成功:<img> + caption 原文(可选)
    - 失败:<code> 原文(红色等宽)
    """
    renderer = get_formula_renderer()
    path = renderer.render(latex, display_mode=is_display)
    safe_latex = _escape_latex_for_caption(latex)

    if path is None:
        # 渲染失败(KaTeX 不支持 / 网络 / 缓存损坏) → fallback 原文
        marker = '$$' if is_display else '$'
        return (
            f'<code class="formula-error" '
            f'style="color:#c00; font-family:Consolas,monospace; '
            f'background:#fee; padding:1px 4px; border-radius:2px;">'
            f'{marker} {safe_latex} {marker}</code>'
        )

    # Windows 路径转 file:// URL(正斜杠)
    file_url = 'file:///' + str(path).replace('\\', '/').lstrip('/')

    # v17:img 标签加 HTML 属性 width/height = PNG 物理像素
    # 强制 QTextBrowser 的 QTextImageFormat 按 HTML 属性显示(不被容器/DPR 调整)
    # "渲染多少像素就显示多少像素" — user 要求
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as _pi:
            _pw, _ph = _pi.size
        _size_attr = f' width="{_pw}" height="{_ph}"'
    except Exception:
        _size_attr = ''

    if is_display:
        # v18.1:display 公式 — 不用外层 div,直接放 img 进 <p>(之前用 div 包裹导致 QTextBrowser 把它当块,布局错乱)
        # img HTML width/height 强制 = PNG 物理像素
        # ★ 加 max-width:none / max-height:none 压制 .formula-block 容器自身的
        #   max-width:100% 行为(避免 QTextBrowser 把 img 缩到容器宽)
        return (
            f'<p class="formula-block" '
            f'style="text-align:center; margin:8px 0; max-width:none;">'
            f'<img src="{file_url}" alt="{safe_latex}"{_size_attr} '
            f'style="vertical-align:middle; max-width:none; max-height:none;"/>'
            f'</p>'
        )
    else:
        # v18.1:inline 公式 — 不用 inline-block span 包裹!
        # 之前用 <span class="formula-inline" style="display:inline-block"> 包裹,
        # 导致 QTextBrowser 渲染时 image 被压成 1 像素(inline-block span 在 QTextBrowser 中行为异常)。
        # 直接放 img 进 p,QTextBrowser 正确按 PNG 物理显示。
        return (
            f'<img class="formula-inline" '
            f'src="{file_url}" alt="{safe_latex}"{_size_attr} '
            f'style="vertical-align:middle; max-width:none; max-height:none;"/>'
        )


def md_to_html_with_formulas(md_text: str) -> str:
    """
    markdown → HTML,公式渲染为 <img> + caption。

    流程:
    1. 保护代码块(```...``` 和 `...`)→ 占位
    2. 提取 $$...$$(display)和 $...$(inline)公式 → 占位
    3. markdown → HTML(markdown lib)
    4. 公式占位 → 渲染成 <img> + caption HTML
    5. 恢复代码块占位

    返回的 HTML body 不含 <html>/<head>,只 body 内容
    """
    if not md_text:
        return ''

    code_blocks: List[str] = []
    formulas: List[Tuple[str, bool]] = []  # (latex, is_display)

    def save_code(m):
        idx = len(code_blocks)
        code_blocks.append(m.group(0))
        return _CODE_PLACEHOLDER.format(idx)

    def save_display(m):
        idx = len(formulas)
        formulas.append((m.group(1).strip(), True))
        return _FORMULA_PLACEHOLDER.format(idx)

    def save_inline(m):
        idx = len(formulas)
        formulas.append((m.group(1).strip(), False))
        return _FORMULA_PLACEHOLDER.format(idx)

    # Step 1+2: 保护代码 + 提取公式(同步进行,代码保护必须先)
    protected = _CODE_BLOCK.sub(save_code, md_text)
    protected = _INLINE_CODE.sub(save_code, protected)
    protected = _DISPLAY_FORMULA.sub(save_display, protected)
    protected = _INLINE_FORMULA.sub(save_inline, protected)

    # Step 3: markdown → HTML
    md = markdown.Markdown(
        extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'],
        output_format='html',
    )
    html_body = md.convert(protected)

    # Step 4: 公式占位 → 渲染 HTML
    for idx, (latex, is_display) in enumerate(formulas):
        formula_html = _render_formula_html(latex, is_display)
        html_body = html_body.replace(_FORMULA_PLACEHOLDER.format(idx), formula_html)

    # Step 5: 恢复代码块占位(markdown lib 把 ```code``` 包成 <pre><code>,
    #   我们的占位是 ```code```,可能被 markdown 二次解析;直接替换为转义后的 code 块)
    for idx, code in enumerate(code_blocks):
        # 简化处理:把 ```code``` 转成 <pre><code>...</code></pre>
        inner = code.strip('`').strip('\n')
        # 简单处理:行 ```lang\n code \n ``` → <pre><code class="lang-lang">code</code></pre>
        lines = inner.split('\n', 1)
        if len(lines) == 2:
            lang, body = lines
            lang = lang.strip() or ''
            escaped = _html.escape(body)
            replacement = f'<pre><code class="language-{lang}">{escaped}</code></pre>'
        else:
            replacement = f'<pre><code>{_html.escape(lines[0])}</code></pre>'
        html_body = html_body.replace(_CODE_PLACEHOLDER.format(idx), replacement)

    return html_body


# ============================================================
# HTML 包装(body CSS 样式)
# ============================================================

DEFAULT_BODY_CSS = """
body {
    margin: 0; padding: 8px 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "微软雅黑", sans-serif;
    font-size: 14px; line-height: 1.6;
    word-wrap: break-word;
}
h1, h2, h3, h4, h5, h6 { margin: 0.6em 0 0.3em 0; font-weight: 600; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.25em; }
h3 { font-size: 1.1em; }
p { margin: 0.4em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
li { margin: 0.15em 0; }
blockquote { margin: 0.5em 0; padding: 0.2em 0.8em; border-left: 3px solid #ccc; color: #666; }
pre { background: #f4f4f4; padding: 8px; border-radius: 4px; overflow-x: auto;
      font-family: Consolas, monospace; font-size: 0.9em; line-height: 1.4; }
code { font-family: Consolas, monospace; font-size: 0.9em; background: #f4f4f4;
       padding: 1px 4px; border-radius: 2px; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 0.5em 0; }
th, td { border: 1px solid #ddd; padding: 4px 8px; }
th { background: #f8f8f8; }
hr { border: 0; border-top: 1px solid #ddd; margin: 0.8em 0; }
.formula-block { margin: 8px 0; padding: 0; max-width: none; }
.formula-inline { display: inline-block; vertical-align: middle; max-width: none; }
.formula-source { user-select: text; }
.formula-error { user-select: text; }
/* v18: 删 img { max-width: 100% } — 全局限制会把公式图片水平压缩
   公式图片走自己 HTML width/height + inline max-width:none,按 PNG 物理像素显示 */
"""


def wrap_html(
    body_html: str,
    body_css: str = DEFAULT_BODY_CSS,
    bg: str = 'transparent',
    text_color: str = '#222222',
) -> str:
    """把 body HTML 包成完整 HTML 文档(可直接给 QTextBrowser.setHtml)"""
    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
{body_css}
body {{ background: {bg}; color: {text_color}; }}
</style></head>
<body>{body_html}</body></html>'''
