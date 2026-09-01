"""
v7 user 真机渲染验证 — 不依赖 headless 进程

修复:之前的判定标准错了(gray < 250 把灰底 #f0f0f0 全算非白,显示 100% 假阳性)
新判定:黑色像素占比(KaTeX 公式是黑色,白底应 > 70% 白色 + 5-30% 黑色 = 公式字符)

用法:user 在真桌面(双击图标、cmd 启动)下跑这个脚本
- 不依赖 PowerShell
- 不强制 offscreen
- 输出每个公式 PNG 的实际大小 + 黑色像素占比
- user 直接双击打开 PNG 看内容
"""
import os
import sys
import time

# 强制不 offscreen(让 Qt 选真桌面平台)
os.environ.pop('QT_QPA_PLATFORM', None)

# 项目根
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

# 1) 启动 app,立即打印实际 platform
app = QApplication.instance() or QApplication(sys.argv)
print(f"[1] Qt platform: {app.platformName()}")
print(f"[1] QT_QPA_PLATFORM env: {os.environ.get('QT_QPA_PLATFORM', '(not set)')}")
assert app.platformName() == 'windows', \
    f"❌ 平台不是 'windows' = {app.platformName()!r},不是真桌面 session!"


# 2) 清空旧缓存
cache_dir = project_root / 'data' / 'cache' / 'formulas'
if cache_dir.exists():
    for p in cache_dir.glob('*.png'):
        p.unlink()
print(f"[2] 已清空缓存目录: {cache_dir}")


# 3) 渲染 5 个公式 + 1 个 pmatrix(之前失败那个)
from ui.formula_renderer import get_formula_renderer

renderer = get_formula_renderer()

# 覆盖:简单 inline / 复杂 inline / display / 量子力学 / norm / pmatrix
test_cases = [
    ('E=mc^2', False, 'E=mc² (简单 inline)'),
    ('\\int_0^1 x^2 dx', False, '∫x²dx (积分 inline)'),
    ('\\frac{\\partial f}{\\partial x}', False, '∂f/∂x (偏导 inline)'),
    ('\\langle \\psi | P | \\psi \\rangle', False, '<ψ|P|ψ> (量子 bra-ket inline)'),
    ('\\|\\nabla \\mathcal{L}\\|_2', False, '||∇L||₂ (norm inline)'),
    ('A = \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}', True, 'A=pmatrix (矩阵 display)'),
]

print("\n[3] 渲染 6 个公式:")
results = []
for formula, display, desc in test_cases:
    t0 = time.perf_counter()
    path = renderer.render(formula, display_mode=display)
    dt = (time.perf_counter() - t0) * 1000

    if not path:
        print(f"  [FAIL] {dt:6.0f}ms {desc}")
        results.append((desc, None, None, 0, 0))
        continue

    # 用黑色像素占比判定(公式字符是 #000,白底 #fff)
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    pixels = list(img.getdata())
    total = len(pixels)
    # 黑色像素(公式字符)
    black = sum(1 for r, g, b in pixels if r < 100 and g < 100 and b < 100)
    # 白色像素(背景)
    white = sum(1 for r, g, b in pixels if r > 240 and g > 240 and b > 240)
    # 灰色(占位 / 透明 / 中间色)
    gray = total - black - white

    black_pct = black / total * 100
    white_pct = white / total * 100
    gray_pct = gray / total * 100
    print(f"  [ok]   {dt:6.0f}ms {path.name} ({w}x{h})  "
          f"black={black_pct:5.1f}% white={white_pct:5.1f}% gray={gray_pct:5.1f}%  {desc}")
    results.append((desc, path, (w, h), black_pct, white_pct))


# 4) 总结
print("\n[4] 判定:")
print(f"  {'公式':<35} {'尺寸':<12} {'黑%':<6} {'白%':<6} 判定")
ok_count = 0
for desc, path, size, black_pct, white_pct in results:
    if path is None:
        verdict = '❌ 渲染失败'
    elif size[0] > 800 or size[1] > 200:
        verdict = f'⚠️  尺寸异常(应该 auto-fit 公式,不是 1200x400)'
    elif black_pct < 0.5:
        verdict = '❌ 没黑色像素(空图)'
    elif white_pct < 30:
        verdict = '⚠️  背景不够白(可能 grab 整个 view)'
    else:
        verdict = '✅ OK'
        ok_count += 1
    size_str = f'{size[0]}x{size[1]}' if size else '-'
    print(f"  {desc:<35} {size_str:<12} {black_pct:<6.1f} {white_pct:<6.1f} {verdict}")

print(f"\n成功 {ok_count}/{len(results)}")

# 5) 提示 user
print("\n[5] 请双击打开以下 PNG 自己看内容:")
for desc, path, size, _, _ in results:
    if path:
        print(f"  {path}")
print(f"\n总览图: {project_root / '_archive_v7_mathimg_demo' / 'v7_user_desktop_overview.png'}")


# 6) 合成总览图(让 user 一眼判断)
from PIL import Image, ImageDraw, ImageFont
overview_w = 1000
row_h = 100
overview_h = row_h * len(results)
overview = Image.new('RGB', (overview_w, overview_h), 'white')

from PIL import ImageDraw, ImageFont
try:
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
except Exception:
    font = ImageFont.load_default()
draw = ImageDraw.Draw(overview)

y = 0
for desc, path, size, black_pct, white_pct in results:
    # 公式描述
    draw.text((10, y + 10), desc, fill='black', font=font)
    # 统计数据
    stats = f'black={black_pct:.1f}% white={white_pct:.1f}%'
    color = 'green' if path and black_pct > 0.5 and white_pct > 30 else 'red'
    draw.text((10, y + 35), stats, fill=color, font=font)
    # 贴 PNG(右半边)
    if path:
        img = Image.open(path)
        x = 500 - img.size[0] // 2
        overview.paste(img, (x, y + (row_h - img.size[1]) // 2))
    y += row_h

out = project_root / '_archive_v7_mathimg_demo' / 'v7_user_desktop_overview.png'
out.parent.mkdir(exist_ok=True)
overview.save(out)
print(f"总览图已存: {out}")
