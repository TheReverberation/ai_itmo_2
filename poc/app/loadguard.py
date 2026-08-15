"""Fallback-предохранители для пиковой нагрузки (упрощение PoC).

Два независимых механизма, срабатывающих ПЕРЕД дорогим LLM-вызовом:

1. LLMBudget — бюджет на LLM-инференс. При исчерпании генерация деградирует
   до retrieval-only/шаблонов (см. docs/architecture.md, fallback #3
   «Превышен бюджет LLM»).
2. IncidentDeduplicator — схлопывает похожие тикеты в кластеры инцидента.
   Лидер кластера генерируется один раз, «последователи» переиспользуют его
   ответ БЕЗ повторного LLM-вызова (см. раздел «Дедупликация при инцидентах»).

Упрощения PoC и чем заменяется в целевой архитектуре:
- бюджет — in-memory счётчик; в целевой системе — распределённый счётчик в Redis
  с дневным лимитом и алертами;
- дедуп — blocking по теме + Jaccard по токенам маскированного текста;
  в целевой системе — embeddings + LSH + streaming-кластеризация (слои A/B/C).
Всё на stdlib, без зависимостей.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


class LLMBudget:
    """Бюджет на LLM-вызовы. Когда денег не хватает на следующий вызов —
    can_spend() возвращает False, и пайплайн деградирует до retrieval-only."""

    def __init__(self, limit_rub: float, cost_per_call_rub: float = 3.0):
        self.limit_rub = limit_rub
        self.cost_per_call_rub = cost_per_call_rub
        self.spent_rub = 0.0
        self.calls = 0

    def remaining(self) -> float:
        return max(0.0, self.limit_rub - self.spent_rub)

    def can_spend(self) -> bool:
        """Хватит ли бюджета ещё на один вызов."""
        return self.spent_rub + self.cost_per_call_rub <= self.limit_rub + 1e-9

    def charge(self) -> None:
        """Списать стоимость одного вызова. Вызывать только после реальной генерации."""
        self.spent_rub += self.cost_per_call_rub
        self.calls += 1


@dataclass
class DedupResult:
    """Итог привязки одного тикета к кластеру инцидента.

    Возвращается из `IncidentDeduplicator.assign` и говорит пайплайну,
    нужно ли тратить LLM на этот тикет или можно переиспользовать готовый ответ.
    """
    cluster_id: str
    is_leader: bool          # True — первый тикет кластера (его и генерируем через LLM)
    leader_ticket_id: str
    cluster_size: int
    cached_answer: str | None = None  # ответ лидера, если он уже сгенерирован


@dataclass
class _Cluster:
    """Кластер инцидента — группа похожих тикетов с одним общим ответом.

    Кластер определяется тремя признаками (см. `IncidentDeduplicator`):
    - `topic`  — тема-blocking-ключ: тикеты разных тем никогда не объединяются;
    - `tokens` — «отпечаток» лидера, с которым сравниваем кандидатов на сходство;
    - `ts`     — время создания: кластер живёт лишь внутри временного окна.
    """
    id: str
    topic: str               # blocking-ключ: сравниваем только внутри одной темы
    tokens: frozenset        # набор токенов лидера — эталон для оценки сходства
    leader_ticket_id: str    # тикет, ответ которого переиспользуют последователи
    ts: float                # момент создания кластера (для скользящего окна)
    members: list = field(default_factory=list)  # все тикеты кластера (лидер + последователи)
    answer: str | None = None  # ответ лидера; заполняется через set_answer


class IncidentDeduplicator:
    """Потоковая leader-follower дедупликация тикетов при инциденте.

    Идея: во время инцидента прилетает волна похожих тикетов. Дорого генерировать
    ответ для каждого. Поэтому первый тикет кластера («лидер») генерируется через
    LLM, а все похожие следом («последователи») переиспользуют его ответ без LLM.

    Как тикет попадает в кластер — три последовательных фильтра:
      1. ОКНО (когда): кластеры старше `window_seconds` забываются — инцидент
         локален во времени, старую волну не смешиваем с новой.
      2. BLOCKING (о чём): сравниваем тикет только с кластерами той же `topic`.
         Это и грубый фильтр по смыслу, и защита от лишних сравнений.
      3. СХОДСТВО (насколько похоже): среди кандидатов ищем самый похожий кластер
         по Jaccard-сходству токенов; принимаем, только если оно >= `threshold`.

    Если подходящий кластер найден — тикет становится последователем. Если нет —
    заводим новый кластер, а тикет становится его лидером.

    Упрощение PoC: Jaccard по токенам вместо эмбеддингов + LSH (см. модуль docstring).
    """

    def __init__(self, similarity_threshold: float = 0.6, window_seconds: int = 600):
        # Порог сходства [0..1]: во сколько токенов должны пересечься тикеты,
        # чтобы считаться одним инцидентом. Выше порог — строже дедуп.
        self.threshold = similarity_threshold
        # Ширина скользящего окна: за его пределами кластеры «протухают».
        self.window_seconds = window_seconds
        # Активные кластеры (только внутри окна) и счётчик для генерации id.
        self._clusters: list[_Cluster] = []
        self._seq = 0

    # --- Как считаем сходство двух тикетов -------------------------------

    @staticmethod
    def _tokenize(text: str) -> frozenset:
        """Текст → множество уникальных токенов (слова/числа в нижнем регистре)."""
        return frozenset(_TOKEN_RE.findall(text.lower()))

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        """Сходство двух наборов токенов = |пересечение| / |объединение| ∈ [0..1].

        1.0 — наборы токенов совпадают, 0.0 — не пересекаются вовсе.
        """
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # --- Три фильтра отбора кластера --------------------------------------

    def _evict_expired(self, now: float) -> None:
        """Фильтр 1 (ОКНО): выбрасываем кластеры старше временного окна."""
        self._clusters = [
            c for c in self._clusters if now - c.ts <= self.window_seconds
        ]

    def _find_best_cluster(self, topic: str, tokens: frozenset) -> _Cluster | None:
        """Фильтры 2–3: среди кластеров той же темы ищем самый похожий.

        Возвращает кластер с максимальным сходством при условии, что оно
        достигает порога `threshold`; иначе None (значит, нужен новый кластер).
        """
        best: _Cluster | None = None
        best_sim = 0.0
        for cluster in self._clusters:
            if cluster.topic != topic:      # фильтр 2 (BLOCKING): другая тема — пропускаем
                continue
            sim = self._jaccard(tokens, cluster.tokens)  # фильтр 3 (СХОДСТВО)
            if sim >= self.threshold and sim > best_sim:
                best, best_sim = cluster, sim
        return best

    def _open_cluster(self, topic: str, tokens: frozenset,
                      leader_ticket_id: str, now: float) -> _Cluster:
        """Заводит новый кластер, где переданный тикет становится лидером."""
        self._seq += 1
        cluster = _Cluster(
            id=f"inc-{self._seq}",
            topic=topic,
            tokens=tokens,
            leader_ticket_id=leader_ticket_id,
            ts=now,
            members=[leader_ticket_id],
        )
        self._clusters.append(cluster)
        return cluster

    # --- Публичный API ----------------------------------------------------

    def assign(self, ticket_id: str, topic: str, masked_text: str,
               now: float | None = None) -> DedupResult:
        """Привязывает тикет к существующему кластеру либо открывает новый (лидер).

        Порядок: протухшие кластеры → поиск похожего той же темы → решение
        follower / leader.
        """
        now = time.time() if now is None else now
        tokens = self._tokenize(masked_text)

        self._evict_expired(now)
        match = self._find_best_cluster(topic, tokens)

        if match is not None:
            # Последователь: примыкает к кластеру и переиспользует его ответ.
            match.members.append(ticket_id)
            return DedupResult(
                cluster_id=match.id,
                is_leader=False,
                leader_ticket_id=match.leader_ticket_id,
                cluster_size=len(match.members),
                cached_answer=match.answer,
            )

        # Лидер: похожего кластера нет — открываем новый.
        cluster = self._open_cluster(topic, tokens, ticket_id, now)
        return DedupResult(
            cluster_id=cluster.id,
            is_leader=True,
            leader_ticket_id=ticket_id,
            cluster_size=1,
            cached_answer=None,
        )

    def set_answer(self, cluster_id: str, answer: str) -> None:
        """Сохраняет ответ лидера, чтобы последователи переиспользовали его."""
        for cluster in self._clusters:
            if cluster.id == cluster_id:
                cluster.answer = answer
                return
