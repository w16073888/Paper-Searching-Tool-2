---
tags: [vendor, provenance]
---

# arXiv weekly vendored program

- Source: `D:\大学编程学习\arXiv周报\程序`
- Copied: 2026-10-25
- Files: 65
- Copied tree SHA-256:
  `e193a2b308bacbc4e991896b3665e4bef772bdcb2d4232e88dbabff891ec2a39`

The source has no repository-level Git metadata. The tree digest above covers all
copied files after the local Linux path adaptation.

Local changes:

- `rllib/cli.py` reads weekly data/cache roots from `ARXIV_WEEKLY_DATA` and
  `ARXIV_WEEKLY_CACHE` instead of Windows absolute paths.

Do not edit upstream behavior here unless the provenance is updated. The
integration entrypoint is `scripts/daily_arxiv_sync.py`.
