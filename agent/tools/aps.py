import os
import re
from bs4 import BeautifulSoup
from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.utils import JOURNAL_CONFIG
from agent.tools.registry import register_tool
from agent.tools.browser_pool import get_browser_pool


def fetch_aps_page_via_browser(journal: str = 'prl'):
    """
    通用APS期刊页面抓取函数。
    Args:
        journal: 期刊标识，如 'prl', 'pra'，对应配置文件中的键。
    """
    import time as _t
    _t0 = _t.time()
    print(f"[aps] fetch_aps_page_via_browser start: journal={journal}", flush=True)

    config = JOURNAL_CONFIG.get(journal)
    if not config:
        print(f"[aps] ! 不支持的期刊: {journal} 耗时 {_t.time()-_t0:.1f}s", flush=True)
        raise ValueError(f"不支持的期刊: {journal}")

    url = config['url_template'].format(journal=journal, section=config['section'])
    link_selector = config['link_selector']
    print(f"[aps]   url={url}  selector={link_selector}", flush=True)

    pool = get_browser_pool()
    print(f"[aps]   拿 browser_pool, 准备 acquire_page (firefox)...", flush=True)
    try:
        with pool.acquire_page(
            browser_type="firefox",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        ) as page:
            print(f"[aps]   acquire_page ok 耗时 {_t.time()-_t0:.1f}s, 开始 goto", flush=True)
            page.set_default_timeout(60000)
            page.goto(url, wait_until="domcontentloaded")
            print(f"[aps]   goto ok 耗时 {_t.time()-_t0:.1f}s, 准备 wait_for_selector({link_selector}, 15s)", flush=True)
            page.wait_for_selector(link_selector, timeout=15000)
            print(f"[aps]   wait_for_selector ok 耗时 {_t.time()-_t0:.1f}s, 拿 page.content()", flush=True)
            html = page.content()
            print(f"[aps]   page.content() 完成 ({len(html)} 字符)  累计耗时 {_t.time()-_t0:.1f}s", flush=True)
            return html
    except Exception as e:
        print(f"[aps] ! 浏览器抓取失败: {e}  累计耗时 {_t.time()-_t0:.1f}s", flush=True)
        raise Exception(f"浏览器自动化抓取失败: {str(e)}")


def parse_aps_papers_from_html(html: str) -> list:
    soup = BeautifulSoup(html, 'html.parser')
    papers = []
    article_cards = soup.find_all('div', class_='browse-result-card')

    for idx, card in enumerate(article_cards):
        try:
            paper = {}

            link_tag = card.find('a', class_='default-link-no-flex', href=True)
            if not link_tag:
                continue
            relative_link = link_tag['href']
            paper['link'] = 'https://journals.aps.org' + relative_link
            paper['title'] = link_tag.get_text().strip()
            if not paper['title']:
                continue

            doi_match = re.search(r'10\.1103/[\w\.-]+', relative_link)
            if not doi_match:
                continue
            paper['doi'] = doi_match.group()
            paper['id'] = paper['doi']

            info_paragraphs = card.find_all('p', class_='t-sm-regular text-text-regular-neutral')
            paper['authors'] = info_paragraphs[0].get_text().strip() if len(info_paragraphs) > 0 else ""
            if len(info_paragraphs) > 1:
                date_text = info_paragraphs[1].get_text().strip()
                date_match = re.search(r'Published (.+)$', date_text)
                paper['published_date'] = date_match.group(1) if date_match else date_text
            else:
                paper['published_date'] = ""

            summary_div = card.find('div', class_='summary')
            if summary_div:
                abstract_p = summary_div.find('p')
                paper['abstract'] = abstract_p.get_text().strip() if abstract_p else ""
            else:
                paper['abstract'] = ""

            papers.append(paper)
        except Exception:
            continue

    return papers


@register_tool(name="generate_aps_daily_summary", is_streaming=False)
def generate_aps_daily_summary(journal: str = 'prl') -> str:
    """
    通过浏览器自动化抓取 APS 最新论文，调用 LLM 生成总结。

    controller 会识别这是"直接输出"工具，单独分气泡渲染。
    """
    config = JOURNAL_CONFIG.get(journal)
    if not config:
        return f"❌ 不支持的期刊: {journal}"

    try:
        html = fetch_aps_page_via_browser(journal)
        papers = parse_aps_papers_from_html(html)
        print(f"[aps] 解析到 {len(papers)} 篇 papers", flush=True)
        if not papers:
            return f"未找到 {config['name']} 论文列表，请稍后重试。"

        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"{i}. **{p['title']}**\n"
            papers_text += f"   - ID: `{p['id']}` | [PRL链接]({p['link']})\n"
            papers_text += f"   - 作者: {p.get('authors', 'N/A')[:150]}...\n"
            papers_text += f"   - 摘要: {p['abstract'][:300]}...\n\n"

        prompt = f"""
        你现在是一个科研助手。以下是从 {config['name']} 期刊抓取的"量子信息科学与技术"类别最新论文列表（共 {len(papers)} 篇）。
        请先分类，然后按照类别为**每一篇论文**生成一份详细的阅读笔记，按照以下格式输出：

        #### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加"等"）
        - **DOI**：列出文章ID（使用 [DOI 链接](https://doi.org/)）
        - **发表日期**：（日期）
        - **核心贡献**：用 2-3 句话概括论文的主要创新点
        - **方法**：简述使用的方法或技术路线（100字左右）
        - **意义与展望**：该工作的潜在影响或后续研究方向（80字左右）

        注意事项：
        - 使用清晰的 Markdown 标题分级。
        - 确保每篇论文都单独列出，不要合并。
        - 输出长度没有限制，请详尽分析。

        论文列表：
        {papers_text}
        """
        return _call_llm(prompt)
    except Exception as e:
        return f"❌ 抓取或总结失败: {str(e)}"


def _download_with_playwright(doi: str, save_dir: str, filename: str = None) -> str:
    """
    通过 chromium 下载 APS 论文 PDF。
    使用 BrowserPool 复用浏览器进程，page/context 用完即关。
    """
    os.makedirs(save_dir, exist_ok=True)

    pool = get_browser_pool()
    with pool.acquire_page(
        browser_type="chromium",
        accept_downloads=True,
        viewport={'width': 1280, 'height': 800},
    ) as page:
        page.goto(f"https://doi.org/{doi}", wait_until="domcontentloaded", timeout=30000)
        final_url = page.url

        pdf_link = page.locator('a[href*="/pdf/"]').first
        if pdf_link.count() == 0:
            pdf_url = final_url.replace('/abstract/', '/pdf/')
        else:
            pdf_url = pdf_link.get_attribute('href')
            if not pdf_url.startswith('http'):
                pdf_url = 'https://journals.aps.org' + pdf_url

        with page.expect_download() as download_info:
            page.evaluate(f"""
                const a = document.createElement('a');
                a.href = '{pdf_url}';
                a.download = '';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            """)

        download = download_info.value

        if filename is None:
            article_id = final_url.split('/')[-1]
            filename = f"{article_id}.pdf"

        filepath = os.path.join(save_dir, filename)
        download.save_as(filepath)
        return filepath


@register_tool(name="download_aps_paper", is_streaming=False)
def download_aps_paper(doi: str) -> str:
    """下载 APS 期刊论文 PDF 到 PAPER_DIR，返回本地路径"""
    save_dir = str(PAPER_DIR)
    os.makedirs(save_dir, exist_ok=True)
    try:
        path = _download_with_playwright(doi, save_dir)
        return f"✅ 已下载：{path}"
    except Exception as e:
        return f"❌ 下载失败: {str(e)}"
