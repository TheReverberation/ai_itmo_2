#!/usr/bin/env python3
"""Demo PoC: прогоняет mock-тикеты через пайплайн.

Запуск из корня репозитория:
    python3 poc/demo.py                # happy path + risky path
    python3 poc/demo.py --llm-down     # fallback: LLM API недоступен

Результаты: poc/out/audit_log.jsonl (лог решений)
            poc/out/operator_inbox.jsonl (очередь оператора)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import DATA_DIR, TicketPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-down", action="store_true",
                        help="Смоделировать недоступность LLM API (fallback path)")
    args = parser.parse_args()

    pipeline = TicketPipeline(llm_available=not args.llm_down)
    tickets = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))

    mode = "LLM НЕДОСТУПЕН (fallback)" if args.llm_down else "обычный режим"
    print(f"=== Demo: {len(tickets)} тикетов, {mode} ===\n")

    for ticket in tickets:
        decision = pipeline.process(ticket)
        print(f"[{decision['ticket_id']}] {ticket['text'][:60]}...")
        print(f"  тема={decision['topic']} conf={decision['confidence']}"
              f" risky={decision['risky']} действие={decision['action']}")
        if decision.get("risk_reasons"):
            print(f"  причины риска: {decision['risk_reasons']}")
        if decision.get("draft"):
            print(f"  черновик: {decision['draft'][:90]}...")
        print()

    print("Логи: poc/out/audit_log.jsonl, poc/out/operator_inbox.jsonl")


if __name__ == "__main__":
    main()
