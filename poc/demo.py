"""Demo-скрипт PoC: прогоняет mock-тикеты через пайплайн и печатает решения.

Запуск:
    python3 poc/demo.py             # обычный режим
    python3 poc/demo.py --llm-down  # имитация недоступности LLM API
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import KnowledgeBase, MockLLM, load_json, process_ticket

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-down", action="store_true",
                        help="имитировать недоступность LLM API (fallback-путь)")
    args = parser.parse_args()

    kb = KnowledgeBase(load_json(HERE / "data" / "kb.json"))
    tickets = load_json(HERE / "data" / "tickets.json")
    llm = MockLLM(available=not args.llm_down)
    log_path = HERE / "decision_log.jsonl"

    mode = "LLM НЕДОСТУПЕН (деградация)" if args.llm_down else "обычный режим"
    print(f"=== Обработка {len(tickets)} тикетов, {mode} ===\n")

    for ticket in tickets:
        d = process_ticket(ticket, kb, llm, log_path)
        print(f"[{d['ticket_id']}] тема={d['topic']} риск={d['risk']} "
              f"confidence={d['confidence']} PII={d['pii_masked'] or '—'}")
        print(f"  → действие: {d['action']}  ({d['reason']})")
        if d["kb_article"]:
            print(f"  → статья KB: {d['kb_article']['title']} (score={d['kb_article']['score']})")
        if d["draft"]:
            print(f"  → черновик: {d['draft'][:100]}…")
        print()

    print(f"Все решения записаны в decision log: {log_path}")


if __name__ == "__main__":
    main()
