"""Offline checks for the stable public API shell and HTTP transport.

    python scripts/check_api.py --quiet

Algorithm v2 is intentionally not tested here until its tools are introduced.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
import sys
import threading
import urllib.error
import urllib.request
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.api.contract import (  # noqa: E402
    CONTRACT_VERSION,
    PARAM_SPEC,
    ApiError,
    AskRequest,
    AskResponse,
    MetaResponse,
    SourceInfo,
    WeekInfo,
    new_request_id,
)
from hotspot.api.engine import (  # noqa: E402
    Engine,
    _history_for_request,
    _load_week_infos,
)
from hotspot.api.http import create_server  # noqa: E402
from hotspot.agent.runtime import (  # noqa: E402
    _AgentOutput,
    build_history_context,
    comparison_terms,
    ground_agent_answer,
    merge_search_results,
)
from hotspot.core.models import (  # noqa: E402
    AgentAnswer,
    AnswerClaim,
    PaperDetail,
    SearchHit,
    SearchResult,
)
from hotspot.reasoning.grounding import verify_and_render  # noqa: E402
from scripts.run_v2_eval import holdout_is_retired  # noqa: E402

FAILS: list[str] = []
CHECKS = 0
QUIET = False


def check(name: str, cond: bool, detail="") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        if not QUIET:
            print(f"  ok  {name}")
        return
    FAILS.append(f"{name}: {detail}")
    if not QUIET:
        print(f"  FAIL {name}  {detail}")


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _post(url: str, payload=None, *, raw: bytes | None = None,
          headers: dict | None = None) -> tuple[int, str, dict]:
    data = raw if raw is not None else json.dumps(
        payload or {}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), dict(exc.headers)


class StubEngine:
    def meta(self) -> MetaResponse:
        return MetaResponse(
            sources=[SourceInfo(key="meeting", label="会议", papers=1)],
            params=[p.to_dict({}) for p in PARAM_SPEC],
            capabilities={"stream": True, "algorithm_v2_rebuilding": True},
        )

    def ask(self, request, on_event=None) -> AskResponse:
        req = request if isinstance(request, AskRequest) else AskRequest.from_dict(request)
        return AskResponse(
            request_id=new_request_id(),
            source=req.source,
            mode=req.mode,
            route="local",
            text="stub",
            meta={"stub": True},
        )


def check_contract() -> None:
    print("[1] contract")
    req = AskRequest.from_dict({
        "question": "测试问题",
        "source": "meeting",
        "mode": "followup",
        "history": [
            {"role": "user", "text": "上一问"},
            {"role": "assistant", "text": "上一答", "citation_ids": ["p1"]},
        ],
        "llm": {"base_url": "https://x/v1", "model": "m", "api_key": "secret"},
        "params": {"search_limit": 16},
        "scope": {"years": [2025], "years_explicit": True},
    })
    dumped = req.to_dict()
    check("request roundtrip", dumped["mode"] == "followup")
    check("api key hidden", dumped["llm"]["api_key"] == "***")
    check("scope retained", dumped["scope"]["years"] == [2025])
    check("params normalized", dumped["params"] == {"search_limit": 16})
    week_info = WeekInfo(
        week_id="2026-W44",
        status="open",
        member_count=12,
        latest_day="2026-10-26",
    )
    check("week metadata contract",
          SourceInfo(
              key="arxiv",
              label="arXiv",
              weeks=[week_info],
          ).to_dict()["weeks"][0] == {
              "week_id": "2026-W44",
              "status": "open",
              "member_count": 12,
              "latest_day": "2026-10-26",
          })

    oversized = AskRequest.from_dict({
        "question": "x" * (AskRequest.MAX_QUESTION + 1),
        "history": [
            {"role": "user", "text": f"问题 {index}"}
            for index in range(AskRequest.MAX_HISTORY + 1)
        ],
    })
    check("question truncation recorded", oversized.question_truncated)
    check("history truncation recorded", oversized.history_truncated)
    check("history keeps newest messages",
          oversized.history[0].text == "问题 1")
    week_request = AskRequest.from_dict({
        "question": "2026-W44 有哪些新论文？",
        "source": "arxiv",
        "scope": {"week_id": "2026-W44"},
    })
    check("ISO week retained",
          week_request.scope.get("week_id") == "2026-W44")
    try:
        AskRequest.from_dict({
            "question": "测试",
            "scope": {"week_id": "2026-W99"},
        })
        invalid_week_rejected = False
    except ApiError:
        invalid_week_rejected = True
    check("invalid ISO week rejected", invalid_week_rejected)
    check("new mode ignores history", _history_for_request(oversized) == [])
    oversized.mode = "followup"
    check("followup mode passes history",
          len(_history_for_request(oversized)) == AskRequest.MAX_HISTORY)
    unsupported_scope = AskRequest.from_dict({
        "question": "测试",
        "scope": {"subfields": ["RL"]},
    })
    check("unsupported scope warned",
          any("subfields" in warning for warning in unsupported_scope.scope_warnings))

    class HistoryStore:
        def get_papers(self, ids):
            return [
                PaperDetail(
                    paper_id=paper_id,
                    title=f"Title {paper_id}",
                    abstract="A" * 1300,
                )
                for paper_id in ids
                if paper_id == "p1"
            ]

    context, context_ids = build_history_context(
        [
            {"role": "user", "text": "上一问"},
            {
                "role": "assistant",
                "text": "上一答",
                "citation_ids": ["p1"],
            },
        ],
        HistoryStore(),
        limit=6,
    )
    check("follow-up citation dereferenced",
          context_ids == ["p1"] and "[P:p1]" in context
          and "Title p1" in context and "…" in context)
    check("comparison terms split",
          comparison_terms("GRPO 相比 PPO 改了什么？") == ["GRPO", "PPO"]
          and comparison_terms(
              "TD3 和 SAC 在连续控制上的核心差异是什么？"
          ) == ["TD3", "SAC"]
          and comparison_terms(
              "逆强化学习与偏好学习在理论和方法上的共同来源是什么？"
          ) == ["逆强化学习", "偏好学习"])
    merged = merge_search_results(
        "GRPO PPO",
        [
            SearchResult(
                query="GRPO",
                results=[SearchHit(paper_id="g1", title="GRPO")],
            ),
            SearchResult(
                query="PPO",
                results=[SearchHit(paper_id="p1", title="PPO")],
            ),
        ],
        limit=2,
    )
    check("comparison search balance",
          {hit.paper_id for hit in merged.results} == {"g1", "p1"})
    check(
        "retired holdout blocked",
        holdout_is_retired(
            ROOT / "eval" / "v2_comparison_holdout_v1.json"
        ),
    )

    resp = StubEngine().ask(req)
    body = resp.to_dict()
    expected = {
        "contract_version", "request_id", "source", "mode", "route",
        "refused", "refuse_reason", "answer", "topics", "graph",
        "usage", "warnings", "error", "meta",
    }
    check("response keys", expected <= set(body))
    check("response json", isinstance(json.dumps(body, ensure_ascii=False), str))
    check("contract version", body["contract_version"] == CONTRACT_VERSION)


def check_http() -> None:
    print("[2] HTTP")
    server = create_server("127.0.0.1", 0, engine=StubEngine(), quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, text = _get(base + "/api/health")
        check("health", status == 200 and json.loads(text)["ok"] is True)

        status, text = _get(base + "/api/meta")
        meta = json.loads(text)
        check("meta", status == 200 and meta["contract_version"] == CONTRACT_VERSION)
        check("params drive settings", len(meta["params"]) == len(PARAM_SPEC))

        status, text, _ = _post(base + "/api/ask", {"question": "测试问题"})
        body = json.loads(text)
        check("ask", status == 200 and body["answer"]["text"] == "stub")

        status, text, _ = _post(
            base + "/api/ask", {"question": "x", "params": {"nope": 1}},
        )
        check("unknown param rejected",
              status == 400 and json.loads(text)["error"]["code"] == "bad_request")

        status, _, _ = _post(base + "/api/ask", raw=b"{bad json")
        check("bad json rejected", status == 400)

        status, _, _ = _post(base + "/api/nope", {})
        check("unknown route", status == 404)

        status, text, headers = _post(
            base + "/api/ask",
            {"question": "测试", "stream": True},
        )
        check("sse content type",
              status == 200 and "text/event-stream" in headers.get("Content-Type", ""))
        check("sse final event",
              "event: result" in text and '"contract_version": "2"' in text)
    finally:
        server.shutdown()
        server.server_close()


def check_real_meta() -> None:
    print("[3] real engine shell")
    meta = Engine().meta().to_dict()
    check("real meta", bool(meta.get("sources")))
    check("algorithm v2", meta["capabilities"].get("algorithm_version") == "2")
    check("week scope capability",
          meta["capabilities"].get("week_scope") is True)
    llm_meta = meta.get("llm") or {}
    check("meta llm defaults",
          bool(llm_meta.get("base_url"))
          and bool(llm_meta.get("model"))
          and isinstance(llm_meta.get("has_server_key"), bool))
    timeouts = meta["capabilities"].get("timeouts") or {}
    check("independent timeout policy",
          timeouts.get("scope_s") == 30
          and timeouts.get("agent_s") == 150
          and timeouts.get("verify_s") == 30)
    check("v2 params", {
        "search_limit", "fetch_limit", "max_claims",
        "agent_request_limit", "agent_tool_limit",
        "scope_timeout_s", "agent_timeout_s", "verify_timeout_s",
    } == {p["key"] for p in meta.get("params", [])})
    arxiv = next(
        (item for item in meta.get("sources", []) if item.get("key") == "arxiv"),
        {},
    )
    check("source index metadata",
          isinstance(arxiv.get("indexed_papers"), int)
          and isinstance(arxiv.get("index_ready"), bool)
          and isinstance(arxiv.get("weeks"), list)
          and isinstance(arxiv.get("new_this_week"), int))
    if arxiv.get("index_ready"):
        check("arXiv index coverage",
              arxiv["indexed_papers"] <= arxiv["papers"])
    else:
        check("arXiv index unavailable is explicit",
              "索引未构建" in str(arxiv.get("note") or ""))

    truncated = Engine().ask({
        "question": "x" * (AskRequest.MAX_QUESTION + 1),
        "history": [
            {"role": "user", "text": f"问题 {index}"}
            for index in range(AskRequest.MAX_HISTORY + 1)
        ],
        "dry_run": True,
    })
    check("request truncation warnings",
          any("问题超过 2000 字" in warning for warning in truncated.warnings)
          and any("历史消息超过 40 条" in warning
                  for warning in truncated.warnings))


def check_week_metadata() -> None:
    print("[4] week metadata")
    root = ROOT / "workspace" / f"week-meta-check-{uuid.uuid4().hex}"
    try:
        for week_id, status, members, days in [
            ("2026-W41", "partial", 804, ["2026-10-09"]),
            ("2026-W44", "open", 17, ["2026-10-26"]),
        ]:
            week_dir = root / week_id
            week_dir.mkdir(parents=True)
            (week_dir / "manifest.json").write_text(json.dumps({
                "week_id": week_id,
                "status": status,
                "member_count": members,
                "completed_days": days,
                "observed_days": days,
            }), encoding="utf-8")
        weeks, warnings = _load_week_infos(root)
        check("weeks sorted newest first",
              [item.week_id for item in weeks]
              == ["2026-W44", "2026-W41"])
        check("week manifest metadata",
              weeks[0].status == "open"
              and weeks[0].member_count == 17
              and weeks[0].latest_day == "2026-10-26")
        check("valid manifests no warnings", warnings == [])

        bad_dir = root / "2026-W42"
        bad_dir.mkdir()
        (bad_dir / "manifest.json").write_text("{bad json", encoding="utf-8")
        weeks, warnings = _load_week_infos(root)
        check("bad week manifest is explicit",
              "2026-W42" not in {item.week_id for item in weeks}
              and any("2026-W42" in warning for warning in warnings))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_verify_timeout_fallback() -> None:
    print("[5] verify timeout fallback")

    class Paper:
        title = "DPO removes the explicit reward model"
        abstract = "Direct Preference Optimization trains directly from preferences."

    class Corpus:
        def get(self, pid):
            return Paper() if pid == "p1" else None

    class SlowClient:
        usage = type("Usage", (), {"failures": 0})()

        def ask(self, *args, **kwargs):
            time.sleep(2.0)
            return type("Call", (), {"ok": False, "data": {}, "error": "slow"})()

    answer = AgentAnswer(claims=[
        AnswerClaim(
            text="DPO removes the explicit reward model",
            evidence_ids=["p1"],
        ),
    ])
    result = verify_and_render(
        answer,
        corpus=Corpus(),
        cfg={},
        client=SlowClient(),
        timeout_s=0.05,
        fallback_ids=["p1"],
    )
    check("verify 超时使用本地抽取式降级",
          not result.refused and len(result.citation_ids) == 1
          and result.verification_status == "timeout_fallback"
          and any("超时" in warning for warning in result.warnings),
          str(result))


def check_agent_output_recovery() -> None:
    print("[6] agent output recovery")
    repaired = _AgentOutput.model_validate({
        "claims": [
            {"text": "supported", "evidence_ids": ["p1"]},
            "no_answer_reason, limitations: []}",
        ],
        "no_answer_reason": "",
        "limitations": [],
    })
    check(
        "malformed claim entry dropped",
        len(repaired.claims) == 1
        and repaired.claims[0].evidence_ids == ["p1"]
        and any("非对象" in item for item in repaired.limitations),
    )

    draft = AgentAnswer(claims=[
        AnswerClaim(text="mixed", evidence_ids=["p1", "p2", "p1"]),
        AnswerClaim(text="missing only", evidence_ids=["p2"]),
    ])
    _, retry = ground_agent_answer(
        draft,
        fetched={"p1"},
        max_claims=6,
        can_retry_for_fetch=True,
    )
    check(
        "missing fetch requests bounded retry",
        retry is not None and "p2" in retry,
    )

    draft = AgentAnswer(claims=[
        AnswerClaim(text="mixed", evidence_ids=["p1", "p2", "p1"]),
        AnswerClaim(text="missing only", evidence_ids=["p2"]),
    ])
    grounded, retry = ground_agent_answer(
        draft,
        fetched={"p1"},
        max_claims=6,
        can_retry_for_fetch=False,
    )
    check(
        "exhausted grounding degrades safely",
        retry is None
        and len(grounded.claims) == 1
        and grounded.claims[0].evidence_ids == ["p1"]
        and any("未 fetch" in item for item in grounded.limitations),
    )


def main(argv: list[str] | None = None) -> int:
    global QUIET
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    QUIET = args.quiet
    sink = io.StringIO() if QUIET else None
    if QUIET:
        with contextlib.redirect_stdout(sink):
            check_contract()
            check_http()
            check_real_meta()
            check_week_metadata()
            check_verify_timeout_fallback()
            check_agent_output_recovery()
    else:
        check_contract()
        check_http()
        check_real_meta()
        check_week_metadata()
        check_verify_timeout_fallback()
        check_agent_output_recovery()
    if FAILS:
        print(f"✗ checks={CHECKS} failures={len(FAILS)}")
        for item in FAILS:
            print(" -", item)
        return 1
    print(f"✓ checks={CHECKS} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
