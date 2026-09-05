"""
v7 公式渲染器(off-screen QWebEngineView + KaTeX + 截图 + PNG 缓存)

设计:
- 公式作为 QTextBrowser 内嵌 <img> 资源
- KaTeX 渲染公式为 PNG(按 md5 缓存),QTextBrowser 显示
- off-screen QWebEngineView 一次性创建,所有公式共享
- 失败 fallback 原文
- 单例(整个 app 一个 renderer)
- 懒启动:第一次调用时初始化 QWebEngineView

注意:
- QWebEngineView 必须在主线程(Qt 限制)
- 同步实现(简化),后续可改 QThread 异步
- 缓存目录: data/cache/formulas/<md5>.png(按 app 数据根)
"""
from __future__ import annotations
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import QUrl, QSize, Qt, QEventLoop, QTimer, QBuffer, QIODevice
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import QApplication, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False


# 2026-09-05 P0 fix: 模块级 _DISABLED 标志, 避免 render() 每次读 process env
# 触发条件: QWebEngineView (Chromium) + BGE (sentence-transformers) 共享 native 资源
# 在 LLM 输出 150+ 公式时偶发 0xC0000005 段错误
# 设 DISABLE_FORMULA_RENDERER=1 让公式显示 LaTeX 原文, 不渲染 PNG
_DISABLED = os.getenv("DISABLE_FORMULA_RENDERER", "").lower() in ("true", "1", "yes")
if _DISABLED:
    print("[FormulaRenderer] DISABLED via env DISABLE_FORMULA_RENDERER=1 (P0 段错误绕开)", flush=True)
    HAS_QTWEBENGINE = True
except ImportError:
    HAS_QTWEBENGINE = False


# 公式占位 HTML 模板(KaTeX 渲染用,内边距 4px)
# 不用 auto-render.min.js:它在 file:// 协议下 UMD 不挂全局 renderMathInElement
# 改用 katex.renderToString(核心 API,UMD 必挂全局 katex)
_FORMULA_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{katex_css}">
<script src="{katex_js}"></script>
<style>
html, body {{
    margin: 0; padding: 0; background: transparent;
    display: inline-block;
    font-size: 60px;  /* v18.4: 20 → 60 (3x 高分辨率渲染)
                        原因: 字号 20 渲染的 PNG 物理 121x33,直接给 QTextBrowser 显示
                              按 setWidth=80 fixed 缩,字符内部细节被 LANCZOS 一次性降到 1x,
                              user 觉得"模糊/大小不稳定"。
                              现在字号 60 渲染物理 363x99,缩到 80 logical 时 3x 降采样,
                              字符内部细节锐利,跟 14px 文字协调 */
}}
.katex {{ color: #000; }}
</style>
</head><body>
<span id="formula">__FORMULA_LATEX__</span>
<script>
    try {{
        const latex = document.getElementById('formula').textContent;
        const displayMode = (__DISPLAY_MODE__ === '1');
        // 不指定 output: 用默认 htmlAndMathml,这样 KaTeX 才会 wrap <span class="katex">
        // 指定 output: 'html' 会让 KaTeX 不加 .katex wrapper,querySelectorAll('.katex').length=0
        // throwOnError: true → KaTeX 失败抛 Error → JS catch 设 title='error:msg' →
        //   Python 端 _wait_for_typeset 检测 title !== 'ready' → return None
        // throwOnError: false → KaTeX 输出 fallback 原文(包 .katex wrapper),
        //   之前的 .katex 节点 > 0 验证是假阳性 — fallback 文本也被算成"成功"
        const html = katex.renderToString(latex, {{
            throwOnError: true,
            displayMode: displayMode
        }});
        document.getElementById('formula').innerHTML = html;
        document.title = 'ready';
    }} catch (e) {{
        document.title = 'error:' + e.message;
    }}
</script>
</body></html>"""


def _md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


class FormulaRenderer:
    """
    单例公式渲染器。

    使用:
        renderer = get_formula_renderer()
        path = renderer.render("$E=mc^2$")  # 同步,首次 ~200-300ms,缓存命中 ~0ms
        if path is None:
            # 失败,fallback 原文
            ...
    """
    _instance: Optional['FormulaRenderer'] = None

    def __init__(self):
        if not HAS_QTWEBENGINE:
            raise RuntimeError("QWebEngineView 不可用,FormulaRenderer 无法初始化")
        self._view: Optional[QWebEngineView] = None
        self._cache_dir: Optional[Path] = None
        self._ready = False
        # v8 改动: 资源路径不再依赖 ui.message_bubble_1._BASE_DIR(已删除),自己定位
        # 路径: ui/formula_renderer.py 同目录的 resources/katex/
        self._resource_base = Path(__file__).resolve().parent
        # 必须 resolve 成绝对路径,否则 QUrl.fromLocalFile() 会输出 file:xxx(单冒号,加载失败)
        self._katex_dir = (self._resource_base / 'resources' / 'katex').resolve()
        self._load_finished: list = []  # 一次性的 loadFinished 回调栈
        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'render_failures': 0,
            'total_render_time_ms': 0.0,
        }

    def _ensure_view(self) -> QWebEngineView:
        """懒创建 off-screen QWebEngineView(关键:必须 show + WA_DontShowOnScreen)"""
        if self._view is not None:
            return self._view
        # 创建 view(不放任何容器内)
        view = QWebEngineView()
        view.setGeometry(0, 0, 1200, 400)
        # v8 关键修复: 必须设 WA_TranslucentBackground + setStyleSheet transparent
        from PySide6.QtCore import Qt as _Qt
        view.setAttribute(_Qt.WidgetAttribute.WA_TranslucentBackground, True)
        view.setStyleSheet("background: transparent;")
        view.page().setBackgroundColor(Qt.transparent)
        # 关键:QWebEngineView 在 widget 不可见时不 paint → view.grab() 拿到空 pixmap
        # 修法:show() + WA_DontShowOnScreen(显示但不绘制到屏幕)→ widget 实际 paint
        # 但不占用户屏幕,也不抢焦点
        view.setAttribute(_Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        view.show()
        # 隐藏窗口的标题(Windows 任务栏可能短暂显示)
        view.setWindowTitle("QAgent Formula Renderer")
        self._view = view
        return view

    def _ensure_cache_dir(self) -> Path:
        """懒创建缓存目录"""
        if self._cache_dir is not None:
            return self._cache_dir
        # data/cache/formulas/ 在 QAgent 项目根的 data/ 下(绝对路径)
        project_root = Path(__file__).resolve().parent.parent
        cache_dir = (project_root / 'data' / 'cache' / 'formulas').resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir
        return cache_dir

    def _cache_path(self, formula: str, display_mode: bool) -> Path:
        key = _md5(f"{'d' if display_mode else 'i'}:{formula}")
        return self._ensure_cache_dir() / f"{key}.png"

    def _is_valid_png(self, path: Path) -> bool:
        """检查 PNG 文件是否有效(可能上次写入失败/截图为空)"""
        if not path.exists():
            return False
        if path.stat().st_size < 100:  # 太小 = 空白图
            return False
        return True

    def _wait_for_load(self, view: QWebEngineView, timeout_ms: int = 10000) -> bool:
        """同步等待 view load 完成(用 polling,不阻塞 Qt event loop)"""
        import time as _t
        finished = {'fired': False, 'ok': False}
        def on_load(ok):
            finished['fired'] = True
            finished['ok'] = ok
        view.loadFinished.connect(on_load)
        # 强制 processEvents 让 Chromium 内部事件跑起来
        deadline = _t.time() + timeout_ms / 1000.0
        while not finished['fired'] and _t.time() < deadline:
            QApplication.processEvents()
            _t.sleep(0.02)  # 20ms
        # 最后再 processEvents 一次让回调彻底 fire
        for _ in range(5):
            QApplication.processEvents()
            _t.sleep(0.01)
        return finished['ok']

    def _wait_for_typeset(self, view: QWebEngineView, timeout_ms: int = 3000) -> bool:
        """等待 KaTeX typeset 完成(通过 page title 变化)"""
        import time as _t
        deadline = _t.time() + timeout_ms / 1000.0
        # 给一个初始延迟,等 JS 启动
        _t.sleep(0.05)
        QApplication.processEvents()
        while _t.time() < deadline:
            QApplication.processEvents()
            try:
                title = view.page().title()
                if title == 'ready':
                    # 额外多等 100ms 让 typeset 完全稳定
                    _t.sleep(0.1)
                    QApplication.processEvents()
                    return True
            except Exception:
                pass
            _t.sleep(0.05)
        return False

    def render(self, formula: str, display_mode: bool = False) -> Optional[Path]:
        """
        同步渲染公式为 PNG,返回路径;失败返回 None。
        - 缓存命中立即返回(0ms) + console 标 "cache hit"(便于排查"console 重复 log")
        - 缓存未命中:首次需要 KaTeX 渲染(200-500ms) + console 标实际渲染

        2026-09-05 P0 fix: DISABLE_FORMULA_RENDERER=1 env 时直接返 None
        触发条件: QWebEngineView + BGE (sentence-transformers/PyTorch) 多线程
        在 LLM 生成大量公式输出(150+)时偶发 0xC0000005 段错误 + 0xC00000FD 栈溢出
        临时绕开: 让公式显示为 LaTeX 原文, 不渲染 PNG
        """
        # P0 紧急: 显式禁用公式渲染 (模块级 _DISABLED 标志, 避免每次 render 读 process env)
        if _DISABLED:
            return None
        # 1. 检查缓存
        cache_path = self._cache_path(formula, display_mode)
        if self._is_valid_png(cache_path):
            self.stats['cache_hits'] += 1
            # v18.7: console 显式标 cache hit — user 之前报"控制台对已有公式重复渲染"
            # 实测 console 没重复 log(只有 _do_render 渲染时打 log),但 user 看错以为重复
            # 加 hit 日志让 cache 复用情况透明,看 chat 时能区分 hit vs miss
            # 阈值: hit 超过 cache_hits_logged_threshold 才 log(避免每条公式都打)
            if self.stats['cache_hits'] <= 5 or self.stats['cache_hits'] % 50 == 0:
                print(f'[FormulaRenderer] cache hit #{self.stats["cache_hits"]}: {formula[:50]!r}')
            return cache_path

        # 2. 渲染
        self.stats['cache_misses'] += 1
        t0 = time.perf_counter()
        try:
            png_path = self._do_render(formula, display_mode)
        except Exception as e:
            print(f'[FormulaRenderer] 渲染失败: {formula[:50]!r} -> {e}')
            self.stats['render_failures'] += 1
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats['total_render_time_ms'] += elapsed_ms
        if png_path is None:
            self.stats['render_failures'] += 1
        return png_path

    def _do_render(self, formula: str, display_mode: bool) -> Optional[Path]:
        """实际渲染逻辑"""
        import time as _t
        view = self._ensure_view()
        cache_path = self._cache_path(formula, display_mode)

        # 准备 HTML
        formula_text = formula.strip()
        # 公式参数可能带 $...$ / $$...$$ 包裹(strip 掉,KaTeX 自己处理 math mode)
        # 因为 cache key 来自调用者传入的 formula,所以调用者传什么都行
        # v18.6: 强制 displayMode=False(忽略 $$ 强制 inline)
        # 原因: user 截图显示 $$...$$ 公式 vs $...$ 公式字符 visual 大小不一致
        #       display 模式 KaTeX 字符更大 + 上下留白多,即使 v18.5 setHeight=50 fixed
        #       字符实际 visual 大小仍比 inline 大 25%+ → user 反馈"大小不统一"
        # 修法: 全部按 inline 渲染 → 字符物理一致 → v18.5 标准化后 visual 一致
        if formula_text.startswith('$$') and formula_text.endswith('$$'):
            formula_text = formula_text[2:-2].strip()
        elif formula_text.startswith('$') and formula_text.endswith('$') and not formula_text.startswith('$$'):
            formula_text = formula_text[1:-1].strip()
        # 始终 inline 模式(忽略 $$ 强制 inline,见上面 v18.6 注释)
        # 公式字符串 escape:写到 HTML <span> 文本节点,浏览器 textContent 原样保留
        # 反斜杠 \\ 不需要 escape(HTML 文本节点不解析 \)
        # 只 escape </ 避免提前结束 <script>(KHTML/HTML parser 安全)
        # 不再 escape ${ 和 `(没有用 JS 字符串嵌入,直接写 HTML 文本节点)
        formula_js_safe = (
            formula_text
            .replace('</', '<\\/')
        )

        # 用 QUrl.fromLocalFile() 构造路径(Windows 路径转 file:// 更可靠)
        katex_css_url = QUrl.fromLocalFile(str(self._katex_dir / 'katex.min.css')).toString()
        katex_js_url = QUrl.fromLocalFile(str(self._katex_dir / 'katex.min.js')).toString()

        html = _FORMULA_HTML_TEMPLATE.format(
            katex_css=katex_css_url,
            katex_js=katex_js_url,
        )
        # 公式和 display_mode 走字符串替换(避免 f-string 跟 JS 嵌套 {})
        html = html.replace('__FORMULA_LATEX__', formula_js_safe)
        html = html.replace('__DISPLAY_MODE__', '1' if display_mode else '0')

        # 关键:setHtml() 在 Chromium 里走 data:text/html 协议,
        # file:// 协议的外部 JS 加载被阻止(即使语法正确,katex 也 UNDEFINED)
        # 修法:写临时 HTML 文件 + view.load(file://) 走完整 page load 流程
        # 关键:必须写到**系统 temp 目录**(`tempfile.gettempdir()`),写到项目目录可能被
        # Chromium 当作"跨域"或 CSP 限制,file:// JS 加载失败
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8',
            dir=tempfile.gettempdir()
        ) as f:
            tmp_html_path = Path(f.name)
            f.write(html)
        # 加载 HTML
        t_load_start = _t.time()
        view.load(QUrl.fromLocalFile(str(tmp_html_path)))
        # 同步等 load + typeset
        if not self._wait_for_load(view):
            tmp_html_path.unlink(missing_ok=True)
            print(f'[FormulaRenderer] load 超时')
            return None
        t_typeset_start = _t.time()
        load_ms = (t_typeset_start - t_load_start) * 1000
        typeset_ok = self._wait_for_typeset(view)
        t_grab_start = _t.time()
        typeset_ms = (t_grab_start - t_typeset_start) * 1000
        if not typeset_ok:
            try:
                title = view.page().title()
                # 诊断:实际 view URL + typeof katex + node count
                debug_state = {'done': False, 'url': '', 'kt': '', 'nodes': -1}
                def dc(v, key):
                    debug_state[key] = v
                view.page().runJavaScript("location.href", lambda v: dc(v, 'url'))
                view.page().runJavaScript("typeof katex", lambda v: dc(v, 'kt'))
                view.page().runJavaScript(
                    "document.querySelectorAll('.katex').length",
                    lambda v: dc(v, 'nodes')
                )
                import time as _tt
                _t0 = _tt.time()
                while not debug_state['done'] and _tt.time() - _t0 < 1.0:
                    QApplication.processEvents()
                    _tt.sleep(0.02)
            except Exception as e:
                title = f'?({e})'
                debug_state = {'done': True, 'url': '?', 'kt': '?', 'nodes': -1}
            tmp_html_path.unlink(missing_ok=True)
            print(f'[FormulaRenderer] typeset 失败 load={load_ms:.0f}ms '
                  f'title={title!r} url={debug_state.get("url","?")!r} '
                  f'typeof katex={debug_state.get("kt","?")!r} '
                  f'.katex={debug_state.get("nodes","?")} '
                  f'formula={formula[:30]!r}')
            return None

        # 测 #formula 元素位置/大小
        # 重要:不依赖 JS getBoundingClientRect — 在 off-screen QWebEngineView 下不可靠
        # (会拿到 0x0 fallback,导致 setFixedSize 把 view 缩到 13x13,paint 失败,grab 拿到 Qt 占位图)
        # 改方案:view 保持 1200x400,grab 整张,扫描黑色像素 bbox,crop
        # KaTeX 节点验证保留(throwOnError: true 模式下,成功一定有 .katex 节点)
        node_count_state = {'done': False, 'value': 0}
        def node_cb(v):
            node_count_state['value'] = v
            node_count_state['done'] = True
        view.page().runJavaScript(
            "document.querySelectorAll('.katex').length",
            node_cb
        )
        nc_deadline = _t.time() + 2.0
        while not node_count_state['done'] and _t.time() < nc_deadline:
            QApplication.processEvents()
            _t.sleep(0.02)
        if not node_count_state['done'] or int(node_count_state['value'] or 0) == 0:
            # 失败 — 拿更详细信息
            try:
                kt_state = {'done': False, 'v': ''}
                html_state = {'done': False, 'v': ''}
                view.page().runJavaScript("typeof katex", lambda v: (kt_state.update(v=v), kt_state.update(done=True)))
                view.page().runJavaScript(
                    "document.getElementById('formula').innerHTML.substring(0, 200)",
                    lambda v: (html_state.update(v=v), html_state.update(done=True))
                )
                import time as _tt
                _t0 = _tt.time()
                while (not kt_state['done'] or not html_state['done']) and _tt.time() - _t0 < 1.0:
                    QApplication.processEvents()
                    _tt.sleep(0.02)
                tmp_html_path.unlink(missing_ok=True)
                print(f'[FormulaRenderer] typeset 检查失败: '
                      f'title={view.page().title()!r}, '
                      f'typeof katex={kt_state["v"]!r}, '
                      f'#formula.innerHTML={html_state["v"]!r}, '
                      f'formula={formula[:30]!r}')
            except Exception as e:
                tmp_html_path.unlink(missing_ok=True)
                print(f'[FormulaRenderer] typeset 失败 + 诊断失败: {e}')
            return None

        # 截图
        # view 保持 setGeometry 时的 1200x400(setFixedSize 删掉了 — 在 off-screen 模式下
        # 拿到错误的 bounding rect 0x0,setFixedSize 把 view 缩到 13x13,paint 失败,
        # grab 拿到 Qt invalid placeholder)
        # grab 整张 → 用 PIL 扫描黑色像素 bbox(KaTeX 公式字符是 #000) → crop
        try:
            # 关键:view 默认背景是 transparent,grab 出来是 RGBA,到了 PIL 不一定是 RGBA
            #   可能是 RGB+tRNS(索引透明)或 P 模式,导致 PIL convert('RGB') 透明像素处理
            #   不一致 → bbox 扫描失败 → crop 出来"大面积透明"
            # 修法:grab 之前**临时**把 view 背景设白,grab 完恢复
            # 这样 grab 出来是 RGB 白底 + 黑色公式字符,不需要 alpha_composite
            from PySide6.QtCore import Qt as _Qt
            saved_qss = view.styleSheet()
            saved_bg = view.page().backgroundColor()
            # v14:不再 setZoomFactor(2.0) — zoom 让 LANCZOS resize 时降清晰度
            # 直接 1x 物理渲染,字号 20px 字符物理 33 高(v11 状态,user 觉得"两次之间"合适)
            view.setStyleSheet("background: white;")
            view.page().setBackgroundColor(Qt.white)
            # 强制 repaint
            for _ in range(8):
                QApplication.processEvents()
                _t.sleep(0.04)
            pixmap = view.grab()
            # 恢复 transparent
            view.setStyleSheet(saved_qss)
            view.page().setBackgroundColor(saved_bg)
        except Exception as e:
            tmp_html_path.unlink(missing_ok=True)
            print(f'[FormulaRenderer] 截图失败: {e}')
            return None

        # 清理临时 HTML(在截图后)
        tmp_html_path.unlink(missing_ok=True)

        if pixmap is None or pixmap.isNull() or pixmap.width() < 5 or pixmap.height() < 5:
            print(f'[FormulaRenderer] 截图无效 ({pixmap.width()}x{pixmap.height()})')
            return None

        # 用 PIL 扫描非白像素 bbox, crop
        # 现在 grab 出来是 RGB 白底,直接读 + invert + getbbox 即可
        # 关键:QPixmap.save 不接受 io.BytesIO!PySide6 要 QIODevice
        # 修法:用 QBuffer(继承 QIODevice)+ bytes() 转 Python bytes 给 PIL
        try:
            from PIL import Image, ImageOps
            from io import BytesIO
            qbuf = QBuffer()
            qbuf.open(QIODevice.WriteOnly)
            if not pixmap.save(qbuf, 'PNG'):
                print(f'[FormulaRenderer] pixmap.save 失败')
                return None
            data = bytes(qbuf.data())
            qbuf.close()
            img = Image.open(BytesIO(data)).convert('RGB')
            # 找非白像素的 bbox(白底,公式黑字)
            gray = img.convert('L')
            # 黑色像素值 = 0, getbbox 默认找非零像素 → 不对
            # 解决:invert 反转 → 黑色变白 → getbbox 找 255 像素的 bbox
            inverted = ImageOps.invert(gray)
            bbox = inverted.getbbox()
            if not bbox:
                print(f'[FormulaRenderer] bbox 扫描:没找到非白像素 ({img.size[0]}x{img.size[1]})')
                return None
            # padding
            PAD_X, PAD_Y = 6, 4
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0 - PAD_X)
            y0 = max(0, y0 - PAD_Y)
            x1 = min(img.size[0] - 1, x1 + PAD_X)
            y1 = min(img.size[1] - 1, y1 + PAD_Y)
            cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
            # v18.9 撤回 (2026-09-01 17:19):
            #   user 反馈 v18.9 "更模糊了"。分析:
            #   PIL Lanczos 4x 预降采样 + 1:1 物理显示,理论应该比 v18.8 (Qt bilinear 4x 实时缩放) 更清晰,
            #   但实际看到的更糊。
            #   根因猜测: Lanczos 对 4x downsample 这种激进缩放 + 字符本身在 PNG 里占少数像素的组合,
            #   会产生 ringing 伪影(边缘"过度锐化"反而显糊),而且 1:1 物理显示 + DPI 缩放链上
            #   还会被 Qt 再做一次缩放,叠加反而更差。
            #   v18.8 (Qt bilinear 4x 实时缩放) 反而是最稳的: 字符 visual 跟 v18.4 接近, 字符 visual OK。
            # 撤回: 恢复 v18.7 注释 + 不做 PIL resize,让 PNG 保留 KaTeX 60 字号原始物理尺寸。
            #   message_bubble_1.py 同时改回 SCALE 1/4 + setWidth/Height 按比例算 (v18.8 状态)。
            cw, ch = cropped.size
            # PIL Image.save 不返回 bool,失败时抛 OSError — 不要用 if not
            cropped.save(str(cache_path), 'PNG')
        except Exception as e:
            print(f'[FormulaRenderer] bbox crop 失败: {e}')
            # fallback:保存整张(QPixmap.save 接受 str 路径)
            pixmap.save(str(cache_path), 'PNG')
            cw, ch = pixmap.width(), pixmap.height()

        total_ms = (_t.time() - t_load_start) * 1000
        if total_ms > 50:
            print(f'[FormulaRenderer] {formula[:30]!r} load={load_ms:.0f}ms typeset={typeset_ms:.0f}ms '
                  f'total={total_ms:.0f}ms crop={cw}x{ch}')

        return cache_path


# 全局单例(在主线程调用)
_instance: Optional[FormulaRenderer] = None


def get_formula_renderer() -> FormulaRenderer:
    """获取全局公式渲染器单例(懒初始化)"""
    global _instance
    if _instance is None:
        _instance = FormulaRenderer()
    return _instance


# ============================================================
# 公式提取(markdown 文本 → 公式列表)
# ============================================================
_FORMULA_PATTERN = re.compile(
    r'(\$\$[^$]+\$\$|\$[^$\n]+\$)',
    re.DOTALL
)


def extract_formulas(md_text: str) -> list[Tuple[str, bool, Tuple[int, int]]]:
    """
    从 markdown 提取所有公式。
    返回 [(formula_text, is_display, (start, end)), ...]
    """
    result = []
    for m in _FORMULA_PATTERN.finditer(md_text):
        f = m.group(0)
        display_mode = f.startswith('$$')
        formula = f[2:-2].strip() if display_mode else f[1:-1].strip()
        if formula:
            result.append((formula, display_mode, (m.start(), m.end())))
    return result
