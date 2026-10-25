"""The ``driftwatch`` command: serve the API, replay a scenario into it, list scenarios, benchmark."""

from __future__ import annotations

import argparse
import os
import sys

import httpx
import uvicorn

from driftwatch.bench.__main__ import main as bench_main
from driftwatch.data.scenarios import SCENARIOS
from driftwatch.replay import ReplayConfig, replay


def _serve(args: argparse.Namespace) -> int:
    uvicorn.run("driftwatch.api.app:create_app", factory=True, host=args.host, port=args.port, log_config=None)
    return 0


def _replay(args: argparse.Namespace) -> int:
    config = ReplayConfig(
        args.scenario,
        args.seed,
        args.batches,
        args.batch_size,
        args.onset,
        args.label_delay,
        args.interval,
        args.prefix,
    )
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.api, headers=headers, timeout=60.0) as client:
        try:
            counts = replay(client, config)
        except httpx.HTTPStatusError as exc:
            print(f"the service refused a request: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
            return 1
        except httpx.TransportError as exc:
            print(f"cannot reach {args.api}: {exc}", file=sys.stderr)
            return 1
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def _scenarios(_: argparse.Namespace) -> int:
    for s in SCENARIOS.values():
        print(f"{s.name:18s} {s.family:9s} {s.description}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    return bench_main(args.rest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch", description="Model monitoring and safe deployment for a credit model."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8774)
    serve.set_defaults(func=_serve)
    rep = sub.add_parser("replay", help="stream a simulated scenario into a running service")
    rep.add_argument("--api", default="http://127.0.0.1:8774")
    rep.add_argument("--scenario", choices=sorted(SCENARIOS), default="covariate_abrupt")
    rep.add_argument("--seed", type=int, default=11)
    rep.add_argument("--batches", type=int, default=180)
    rep.add_argument("--batch-size", type=int, default=400)
    rep.add_argument("--onset", type=int, default=60)
    rep.add_argument("--label-delay", type=int, default=30)
    rep.add_argument("--interval", type=float, default=0.0, help="seconds to wait between batches")
    rep.add_argument("--prefix", default="", help="batch id prefix (defaults to scenario and seed)")
    rep.add_argument("--api-key", default=os.environ.get("APP_API_KEY"))
    rep.set_defaults(func=_replay)
    sub.add_parser("scenarios", help="list the simulated scenarios").set_defaults(func=_scenarios)
    bench = sub.add_parser("bench", help="run the benchmark (same flags as python -m driftwatch.bench)")
    bench.add_argument("rest", nargs=argparse.REMAINDER)
    bench.set_defaults(func=_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
