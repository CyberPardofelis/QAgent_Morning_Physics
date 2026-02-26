import os
from datetime import datetime
from agent.llm_clients import query_llm

def summarize_and_save(title, content):
    prompt = f"Summarize this physics paper:\n\n{content}"

    summary = query_llm(prompt)

    os.makedirs("summaries", exist_ok=True)

    filename = f"summaries/{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n\n")
        f.write(summary)

    print(f"[Summary saved] {filename}")