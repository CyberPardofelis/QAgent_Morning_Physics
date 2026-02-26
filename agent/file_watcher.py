from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from agent.paper_indexer import index_paper

class PaperHandler(FileSystemEventHandler):

    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            print(f"[Watcher] New paper detected: {event.src_path}")
            index_paper(event.src_path)

def start_watching():
    observer = Observer()
    observer.schedule(PaperHandler(), path="papers", recursive=False)
    observer.start()
    print("[Watcher] Monitoring papers directory...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()