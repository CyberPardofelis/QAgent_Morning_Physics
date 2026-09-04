import os
import re
import logging
from bs4 import BeautifulSoup
from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.utils import JOURNAL_CONFIG
from agent.tools.registry import register_tool
from agent.tools.browser_pool import get_browser_pool
from agent.tools.summary import save_summary_item
from agent.tools.summary_parser import parse_summary_markdown

logger = logging.getLogger(__name__)

# ============================================================
# 2026-09-03 接回 LLM 总结字段落库(用户拍板):
# 之前 2026-09-02 P0.9 彻底回退只走了"抓文 + LLM 总结 return 文本",没存任何 LLM 产出。
# 用户原话:"LLM 总结完了就完了?" —— 应该把 LLM 输出的 核心贡献/方法/意义/关键词 都入库。
# 关键安全约束(防 P0.6 蓝屏复刻):
#   - **零后台线程**:主流程同步解析 markdown + 同步 save_summary_item()(走同步版,不走 _async)
#   - **30 篇 paper → 30 次顺序 SQLite INSERT** → 毫秒级,绝对安全
#   - **不调嵌入**:LLM 总结只入 summary_items.db,不入 chromadb
#     (chroma P0.5 嵌入仍由 analyze_paper 走 section-aware chunk 那条线,跟这条解耦)
# ============================================================


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

    2026-09-03: LLM 总结里的 核心贡献/方法/意义/关键词 会同步落库到 summary_items.db
    (零后台线程,主流程同步解析 + 同步 save_summary_item)
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

        # 2026-09-03: 抓文元数据同步落库(title/authors/abstract),不等 LLM
        # 走 save_summary_item() 同步版(不走 _async),绝对安全
        from datetime import date as _date
        today = _date.today().isoformat()
        for p in papers:
            try:
                save_summary_item(
                    paper_id=p['id'],
                    source='aps_daily',
                    summary_date=today,
                    title=p['title'],
                    authors=p.get('authors', ''),
                    abstract=p.get('abstract', ''),
                )
            except Exception as e:
                logger.warning("[aps] 抓文元数据落库失败(已吞): paper=%s err=%s", p['id'], e)

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
        - **关键词**：5-8 个英文关键词，逗号分隔（用于后续检索/分类；不要翻译，保持英文原词）

        注意事项：
        - 使用清晰的 Markdown 标题分级。
        - 确保每篇论文都单独列出，不要合并。
        - 输出长度没有限制，请详尽分析。

        论文列表：
        {papers_text}
        """
        summary_md = _call_llm(prompt)

        # 2026-09-03: 同步解析 LLM 输出 → 补 核心贡献/方法/意义/关键词
        # 主流程同步 save,不创建后台线程(防 P0.6 蓝屏)
        _persist_parsed_summary_fields(
            parsed=parse_summary_markdown(summary_md),
            source='aps_daily',
            summary_date=today,
        )

        return summary_md
    except Exception as e:
        return f"❌ 抓取或总结失败: {str(e)}"


def _persist_parsed_summary_fields(parsed, source: str, summary_date: str) -> None:
    """2026-09-03: 把解析出的 LLM 字段同步 UPDATE 到 summary_item 已存在的行

    为什么 UPDATE 而非 INSERT:
    - 抓文时已 INSERT 一行(只有 title/authors/abstract)
    - LLM 出来后再补 core_method/core_contrib/keywords/outlook
    - UNIQUE(paper_id, source, summary_date) 已存在,直接 UPDATE

    关键(2026-09-03 P0 防 P0.6 蓝屏复刻):
    - **单连接批 UPDATE**:30 篇用 1 个事务做完,不开 30 个连接
      (开 30 个连接 + FTS5 触发器会触发 "database disk image is malformed")
    - **不走 _async**:同步主流程,绝对安全
    - **失败不抛**:单条失败吞掉,主流程不受影响
    """
    if not parsed:
        return
    valid = [ps for ps in parsed if ps.is_valid()]
    if not valid:
        return
    saved = 0
    try:
        from agent.infra.db import connect
        with connect("summary_items") as conn:
            for ps in valid:
                try:
                    cur = conn.execute(
                        """
                        UPDATE summary_item
                        SET core_method = COALESCE(NULLIF(?, ''), core_method),
                            core_contrib = COALESCE(NULLIF(?, ''), core_contrib),
                            keywords = COALESCE(NULLIF(?, ''), keywords),
                            outlook = COALESCE(NULLIF(?, ''), outlook)
                        WHERE paper_id = ? AND source = ? AND summary_date = ?
                        """,
                        (ps.core_method, ps.core_contrib, ps.keywords, ps.outlook,
                         ps.paper_id, source, summary_date),
                    )
                    if cur.rowcount > 0:
                        saved += 1
                except Exception as e:
                    logger.warning("[aps→db] 单条 UPDATE 失败(已吞): paper=%s err=%s",
                                   ps.paper_id, e)
    except Exception as e:
        logger.warning("[aps→db] 批 UPDATE 失败(已吞): %s", e)
    if saved:
        print(f"[aps→db] ✓ 补 LLM 字段完成: {saved}/{len(valid)} 条", flush=True)


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


@register_tool(name="download_aps_paper", is_streaming=False, cost_tier="L0")
def download_aps_paper(doi: str) -> str:
    """下载 APS 期刊论文 PDF 到 PAPER_DIR，返回本地路径"""
    save_dir = str(PAPER_DIR)
    os.makedirs(save_dir, exist_ok=True)
    try:
        path = _download_with_playwright(doi, save_dir)
        return f"✅ 已下载：{path}"
    except Exception as e:
        return f"❌ 下载失败: {str(e)}"
