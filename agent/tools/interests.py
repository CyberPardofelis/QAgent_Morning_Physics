# tools/interests.py
from datetime import datetime

from .database import get_db_connection


class InterestModel:
    def __init__(self, user_id=1):
        self.user_id = user_id
        self.interests = self._load_interests()

    def _load_interests(self):
        """从数据库加载兴趣标签权重"""
        conn = get_db_connection()
        rows = conn.execute("SELECT tag_type, tag_value, weight FROM user_interests WHERE user_id = ?", (self.user_id,)).fetchall()
        conn.close()
        interests = {}
        for row in rows:
            key = f"{row['tag_type']}:{row['tag_value']}"
            interests[key] = row['weight']
        return interests

    def update_from_action(self, paper_id, action_type):
        """根据用户行为更新兴趣权重（示例：查看+0.1，收藏+0.5，分析+0.3）"""
        # 获取论文信息
        from .database import get_paper_by_id
        paper = get_paper_by_id(paper_id)
        if not paper:
            return
        # 为每个关键词、会议、作者增加权重
        increment = 0
        if action_type == 'view':
            increment = 0.1
        elif action_type == 'favorite':
            increment = 0.5
        elif action_type == 'analyze':
            increment = 0.3
        elif action_type == 'dislike':
            increment = -0.2
        if increment == 0:
            return

        conn = get_db_connection()
        c = conn.cursor()
        # 关键词
        for kw in paper.get('keywords', []):
            c.execute('''
                INSERT INTO user_interests (user_id, tag_type, tag_value, weight, last_updated)
                VALUES (?, 'keyword', ?, ?, ?)
                ON CONFLICT(user_id, tag_type, tag_value) DO UPDATE SET
                    weight = weight + ?,
                    last_updated = ?
            ''', (self.user_id, kw, increment, datetime.now().isoformat(), increment, datetime.now().isoformat()))
        # 会议/期刊（如果有）
        # 作者（可选，注意作者名可能含空格，保持原样）
        for author in paper.get('authors', []):
            # 简单处理：只取全名
            c.execute('''
                INSERT INTO user_interests (user_id, tag_type, tag_value, weight, last_updated)
                VALUES (?, 'author', ?, ?, ?)
                ON CONFLICT(user_id, tag_type, tag_value) DO UPDATE SET
                    weight = weight + ?,
                    last_updated = ?
            ''', (self.user_id, author, increment, datetime.now().isoformat(), increment, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        # 重新加载兴趣
        self.interests = self._load_interests()

    def score_paper(self, paper):
        """计算论文与兴趣的匹配分"""
        score = 0.0
        # 关键词匹配
        for kw in paper.get('keywords', []):
            key = f"keyword:{kw}"
            score += self.interests.get(key, 0)
        # 作者匹配
        for author in paper.get('authors', []):
            key = f"author:{author}"
            score += self.interests.get(key, 0) * 0.5  # 作者权重可调低
        return score

    def get_top_interests(self, n=10):
        """返回权重最高的兴趣标签"""
        sorted_items = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

# 数据库表初始化（在 database.py 的 init_db 中应增加）
def init_interest_tables():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_interests (
            user_id INTEGER,
            tag_type TEXT,
            tag_value TEXT,
            weight REAL DEFAULT 1.0,
            last_updated TEXT,
            PRIMARY KEY (user_id, tag_type, tag_value)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            paper_id TEXT,
            action_type TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
