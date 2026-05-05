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
    soup = BeautifulSoup(html, 'html.parser')
    papers = []
    article_cards = soup.find_all('div', class_='browse-result-card')
    print(f"找到 {len(article_cards)} 个文章卡片")

    for idx, card in enumerate(article_cards):
        try:
            paper = {}

            # 1. 标题和链接：直接从 a.default-link-no-flex 获取
            link_tag = card.find('a', class_='default-link-no-flex', href=True)
            if not link_tag:
                print(f"卡片 {idx} 没有找到链接标签，跳过")
                continue
            relative_link = link_tag['href']
            paper['link'] = 'https://journals.aps.org' + relative_link
            paper['title'] = link_tag.get_text().strip()
            if not paper['title']:
                print(f"卡片 {idx} 标题为空，跳过")
                continue

            # 提取 DOI
            doi_match = re.search(r'10\.1103/[\w\.-]+', relative_link)
            if not doi_match:
                print(f"卡片 {idx} 未找到 DOI，跳过")
                continue
            paper['doi'] = doi_match.group()
            paper['id'] = paper['doi']

            # 2. 作者和日期：获取所有 p.t-sm-regular.text-text-regular-neutral
            info_paragraphs = card.find_all('p', class_='t-sm-regular text-text-regular-neutral')
            # 第一个 p 是作者
            paper['authors'] = info_paragraphs[0].get_text().strip() if len(info_paragraphs) > 0 else ""
            # 第二个 p 包含期刊和日期
            if len(info_paragraphs) > 1:
                date_text = info_paragraphs[1].get_text().strip()
                # 提取 "Published 1 May, 2026" 部分
                date_match = re.search(r'Published (.+)$', date_text)
                paper['published_date'] = date_match.group(1) if date_match else date_text
            else:
                paper['published_date'] = ""

            # 3. 摘要：在 div.summary 内部的 p 标签中
            summary_div = card.find('div', class_='summary')
            if summary_div:
                abstract_p = summary_div.find('p')
                paper['abstract'] = abstract_p.get_text().strip() if abstract_p else ""
            else:
                paper['abstract'] = ""

            papers.append(paper)

        except Exception as e:
            print(f"解析卡片 {idx} 时发生异常: {e}，跳过该论文")
            continue

    print(f"成功解析 {len(papers)} 篇论文")
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
        请先分类，然后按照类别为**每一篇论文**生成一份详细的阅读笔记，按照以下格式输出：

        ### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加“等”）
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
        # print(prompt)
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