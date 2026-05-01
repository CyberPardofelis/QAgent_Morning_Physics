import os
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from .config import PAPER_DIR
from .llm import _call_llm
from .utils import JOURNAL_CONFIG

def fetch_aps_page_via_browser(journal: str = 'prl'):
    """
    通用APS期刊页面抓取函数。
    Args:
        journal: 期刊标识，如 'prl', 'pra'，对应配置文件中的键。
    """
    config = JOURNAL_CONFIG.get(journal)
    if not config:
        raise ValueError(f"不支持的期刊: {journal}")

    # 从配置动态构建URL
    url = config['url_template'].format(journal=journal, section=config['section'])
    link_selector = config['link_selector']
    print('正在通过浏览器抓取APS页面...')


    try:
        with sync_playwright() as p:
            # 您已经设置为 headless=False，很好，这样可以看到浏览器行为
            browser = p.firefox.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.set_default_timeout(60000)

            response = page.goto(url, wait_until="domcontentloaded")
            print(f"HTTP状态码: {response.status}")

            # 等待文章链接出现
            page.wait_for_selector(link_selector, timeout=15000)
            print("成功找到文章链接，内容可能已加载。")

            # 获取页面 HTML
            html = page.content()
            '''
            # 尝试保存 HTML 到当前目录（用于调试），忽略保存失败
            try:
                with open('prl_page_debug.html', 'w', encoding='utf-8') as f:
                    f.write(html)
                print("HTML已保存到当前目录的 prl_page_debug.html")
            except Exception as e:
                print(f"保存HTML到文件失败（可忽略）: {e}")
            #'''
            browser.close()
            return html
    except Exception as e:
        raise Exception(f"浏览器自动化抓取失败: {str(e)}")

def parse_aps_papers_from_html(html: str) -> list:
    """
    从 APS 页面 HTML 中解析论文列表。
    基于当前实际页面结构：每篇文章在 div.browse-result-card 中。
    """
    soup = BeautifulSoup(html, 'html.parser')
    papers = []

    # 定位所有文章卡片
    article_cards = soup.find_all('div', class_='browse-result-card')
    print(f"找到 {len(article_cards)} 个文章卡片")

    for card in article_cards:
        paper = {}

        # 1. 提取标题和链接
        title_tag = card.find('h4', class_='title')
        if title_tag:
            link_tag = title_tag.find('a', href=True)
            if link_tag:
                paper['title'] = link_tag.get_text().strip()
                relative_link = link_tag['href']
                paper['link'] = 'https://journals.aps.org' + relative_link
                # 从链接中提取 DOI 作为 ID
                doi_match = re.search(r'10\.1103/[\w\.-]+', relative_link)
                if doi_match:
                    paper['doi'] = doi_match.group()
                    paper['id'] = paper['doi']#.split('/')[-1]  # 提取ID部分作为本地标识
            else:
                continue
        else:
            continue

        # 2. 提取作者（第一个包含作者列表的 <p> 标签）
        # 通常作者在第一个 <p class="t-sm-regular text-text-regular-neutral"> 中
        author_paragraphs = card.find_all('p', class_='t-sm-regular text-text-regular-neutral')
        if author_paragraphs:
            # 第一个 <p> 通常是作者
            paper['authors'] = author_paragraphs[0].get_text().strip()
        else:
            paper['authors'] = ''

        # 3. 提取摘要：摘要位于隐藏的 <div class="summary ..."> 中
        # 这个 div 包含 data-abstract 属性
        abstract_div = card.find('div', class_='summary', attrs={'data-abstract': True})
        if abstract_div:
            paper['abstract'] = abstract_div.get('data-abstract', '').strip()
        else:
            # 如果没有 data-abstract，尝试获取 div 内的文本（但通常 data-abstract 存在）
            abstract_div = card.find('div', class_='summary')
            if abstract_div:
                paper['abstract'] = abstract_div.get_text().strip()
            else:
                paper['abstract'] = ''

        # 4. 提取发表日期（可选，用于信息丰富）
        # 日期通常在第二个 <p class="t-sm-regular text-text-regular-neutral"> 中
        if len(author_paragraphs) > 1:
            date_text = author_paragraphs[1].get_text().strip()
            # 尝试提取日期部分，例如 "Published 26 February, 2026"
            date_match = re.search(r'Published (.+)$', date_text)
            if date_match:
                paper['published_date'] = date_match.group(1)
            else:
                paper['published_date'] = date_text
        else:
            paper['published_date'] = ''

        papers.append(paper)

    return papers

def generate_aps_daily_summary(journal: str = 'prl'):
    """
    通过浏览器自动化抓取 APS 最新论文，调用 LLM 生成总结。
    """
    config = JOURNAL_CONFIG.get(journal)
    if not config:
        raise ValueError(f"不支持的期刊: {journal}")

    try:
        html = fetch_aps_page_via_browser(journal)
        papers = parse_aps_papers_from_html(html)
        if not papers:
            return "未找到 {config['name']} 论文列表，请稍后重试。"

        # 格式化论文文本（同之前）
        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"{i}. **{p['title']}**\n"
            papers_text += f"   - ID: `{p['id']}` | [PRL链接]({p['link']})\n"
            papers_text += f"   - 作者: {p.get('authors', 'N/A')[:150]}...\n"
            papers_text += f"   - 摘要: {p['abstract'][:300]}...\n\n"

        # 构建 prompt（同上）
        prompt = f"""
        你现在是一个科研助手。以下是从 {config['name']} 期刊抓取的“量子信息科学与技术”类别最新论文列表（共 {len(papers)} 篇）。
        请按研究方向分类总结，每个类别挑选 2-3 篇亮点论文，并确保在描述亮点时包含该论文的 ID（DOI）和超链接。

        **任务要求**：
        1. 整体概览：一句话说明今天新上榜论文总数。
        2. 分类总结：根据标题和摘要分类。
        3. 亮点提取：**每篇亮点必须使用以下格式：** `**标题**(id:)`：一句核心贡献。
        4. 输出格式：Markdown。

        论文列表：
        {papers_text}
        """

        analysis = _call_llm(prompt)
        return analysis
    except Exception as e:
        return f"❌ 抓取或总结失败: {str(e)}"


def download_aps_paper(doi: str, save_dir: str = None) -> str:
    if save_dir is None:
        save_dir = PAPER_DIR
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 直接使用同步 Playwright 下载
        return _download_with_playwright(doi, save_dir)
    except Exception as e:
        return f"❌ 下载失败: {str(e)}"


def _download_with_playwright(doi: str, save_dir: str, filename: str = None) -> str:
    os.makedirs(save_dir, exist_ok=True)

    playwright = sync_playwright().start()
    browser = None
    try:
        browser = playwright.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
        context = browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()

        # 访问DOI获取摘要页
        page.goto(f"https://doi.org/{doi}", wait_until="domcontentloaded", timeout=30000)
        final_url = page.url
        print(f"摘要页 URL: {final_url}")

        # 获取PDF下载链接（从页面中提取）
        # 尝试多种选择器找到PDF链接
        pdf_link = page.locator('a[href*="/pdf/"]').first
        if pdf_link.count() == 0:
            # 如果没有，可能链接是别的形式，尝试从元数据或直接构造
            pdf_url = final_url.replace('/abstract/', '/pdf/')
        else:
            pdf_url = pdf_link.get_attribute('href')
            if not pdf_url.startswith('http'):
                pdf_url = 'https://journals.aps.org' + pdf_url

        # 使用JavaScript强制下载
        with page.expect_download() as download_info:
            page.evaluate(f"""
                const a = document.createElement('a');
                a.href = '{pdf_url}';
                a.download = '';  // 空字符串表示使用URL最后部分作为文件名
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            """)

        download = download_info.value

        # 确定文件名
        if filename is None:
            article_id = final_url.split('/')[-1]
            filename = f"{article_id}.pdf"

        filepath = os.path.join(save_dir, filename)
        print(f"正在保存到: {filepath}")
        download.save_as(filepath)
        print("下载完成")

        return filepath

    except Exception as e:
        print(f"下载过程中出错: {e}")
        raise
    finally:
        if browser:
            browser.close()
        playwright.stop()