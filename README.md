# Metareview Tool (Streamlit)

A hosted version of the metareview skill validated against Persona 3: Reload — same rubric,
same category-threshold logic, same xlsx/docx output shapes, wrapped in a web UI so producers
don't need a Claude session to run one.

Auth and LLM access follow the same pattern as Sega's other internal Streamlit tools (e.g.
`narrative_qa.py`): AWS SES for OTP email, AWS Bedrock for the LLM (falling back to a direct
Anthropic key), everything read from `st.secrets` — nothing typed into the UI, nothing in code.

**Status: proof-of-concept grade.** The rubric/matrix/report logic is the same as the validated
skill. The app shell (Streamlit wrapper, OTP gate, deployment config) is new and has been
smoke-tested locally (see `tests/`) but not run against real production traffic — pilot it with
a couple of producers before trusting it unsupervised.

## What's real vs. what you need to plug in

| Piece | Status |
|---|---|
| Rubric, category threshold, matrix/report structure | Validated against P3R POC |
| xlsx/docx generation, weighted-score math | Unit tested (`tests/test_pipeline.py`), no network/credentials needed |
| HTML extraction from uploaded zips | Unit tested |
| Auth (SES OTP + signed token) | Written, matches the house pattern — needs real `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SES_REGION`/`EMAIL_FROM` secrets to send real email; falls back to on-screen dev-mode code if unset |
| LLM client (Bedrock → Anthropic fallback) | Written, matches `get_client()` in the house pattern — needs real `AWS_BEDROCK_*` or `ANTHROPIC_API_KEY` secrets to try live; not run end-to-end from the environment this was built in (no egress + no credentials there) |
| Live outbound fetch of pasted review URLs | Written (`requests`), needs to be tried from wherever you deploy this — some outlets block automated fetching regardless of who's asking, use the upload path for those |
| Live re-fetch of Metacritic's curated list | Written with a bundled fallback snapshot (`app/curated_outlets_snapshot.json`, dated in the file) — wire up `rubric.fetch_curated_list_live()` with a real fetch function once deployed |
| SSO | Not built. `app/auth.py` is the single file to replace when Sega's SSO is ready |

## Secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:

- `ALLOWED_DOMAIN`, `COOKIE_SIGNING_KEY` — auth config (signing key should be a long random string, not the example placeholder)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SES_REGION` / `EMAIL_FROM` — SES, for OTP email
- `AWS_BEDROCK_ACCESS_KEY_ID` / `AWS_BEDROCK_SECRET_ACCESS_KEY` / `AWS_BEDROCK_REGION` — Bedrock, for the LLM (preferred)
- `ANTHROPIC_API_KEY` — only used if the Bedrock secrets above aren't set

Without SES secrets, login runs in dev mode (the OTP code is shown on-screen instead of
emailed) — fine for local testing, not for a shared deployment. Without Bedrock or Anthropic
secrets, the sidebar shows "No LLM credentials configured" and the run button is disabled.

## Local run

```bash
pip install -r requirements.txt
playwright install chromium   # once — Metacritic full-review-text fallback, app/metacritic.py
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in at least one LLM credential set
streamlit run app/streamlit_app.py
```

The `playwright install chromium` step downloads a browser binary (~150-300MB) into a local
cache the first time; you don't need Docker for this — it works the same on a plain venv on
your laptop, a bare VM, or a systemd service. On Linux, if it later fails at runtime with
missing shared-library errors, re-run as `playwright install --with-deps chromium` instead
(needs sudo, since it apt-installs the OS packages Chromium needs — this is what the Dockerfile
does automatically). macOS and Windows don't normally need `--with-deps`. If you'd rather skip
Playwright entirely, it's optional: leave it uninstalled and everything else works as-is — the
Metacritic tab's full-text fetch just falls back to reporting a clear error for any outlet that
needed a browser to render, instead of crashing.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Covers extraction, matrix building, report building, and the weighted-score formula — the parts
that don't need live credentials or network access. It does **not** cover the actual Bedrock/
Anthropic calls, SES email delivery, or live web fetches — try those manually once deployed with
real secrets and normal internet egress.

## Deploying

```bash
docker build -t metareview-tool .
docker run -p 8501:8501 -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml metareview-tool
```

Docker isn't required, though — it's just a convenient way to ship the Chromium install alongside
the app. Running it directly on a host works the same way as the Local run section above (`pip
install -r requirements.txt`, `playwright install chromium`, `streamlit run app/streamlit_app.py
--server.port=8501 --server.address=0.0.0.0`), kept alive with whatever process manager Sega
already uses for long-running services (systemd, pm2, supervisord, etc.) instead of a container.

Put this behind whatever Sega already uses to expose internal tools (a reverse proxy, an
internal load balancer, a VPN-only network) — **do not expose this directly to the public
internet** even with the OTP gate in place. OTP-over-email is a reasonable stopgap for a small
trusted pilot, not a substitute for being on internal infrastructure.

### Streamlit Community Cloud

Community Cloud (the free `share.streamlit.io` hosting that deploys straight from a GitHub
repo) works for this app, but it gives you no shell/sudo access, so neither the Dockerfile nor
the Local run section's `playwright install --with-deps chromium` step can run there directly.
Two things this repo ships to cover that gap:

- **`packages.txt`** (repo root) — Community Cloud auto-detects this file and apt-installs
  whatever's listed, one package per line; it's the OS-level libraries Chromium itself needs to
  run, equivalent to the Dockerfile's `--with-deps` flag. Already included here — no edits
  needed unless Chromium's own dependencies change in a future Playwright release.
- **The Chromium *browser binary*** is a separate problem `packages.txt` doesn't solve —
  `pip install -r requirements.txt` installs the `playwright` Python package but never
  downloads the actual browser, and Community Cloud's build step has no hook to run a follow-up
  `playwright install chromium` command the way Docker's `RUN` step does. `app/metacritic.py`
  handles this by installing it lazily instead: the first time the Playwright fallback tier is
  actually needed, it installs the browser on the spot (see
  `_ensure_playwright_chromium_installed()`), then proceeds. That first fallback fetch in a
  freshly-started app will be slower (a one-time ~150–300MB download) than every one after it;
  this repeats after every cold start / redeploy, since Community Cloud's filesystem doesn't
  persist across those — that's expected, not a bug. If the install itself fails (e.g. no disk
  space), it's reported as that review's fetch error same as any other Playwright failure,
  rather than crashing the run.

If you hit an apt dependency error on `packages.txt` after a Streamlit-side base-image update,
or a Playwright error about unmet dependencies after upgrading `playwright` in
`requirements.txt`, try pinning to a slightly older `playwright` version first — Community
Cloud's Debian image has occasionally lagged what the newest Playwright releases expect.

### Upload size limit

Streamlit defaults to a 200MB cap on uploaded files, which the zip-of-reviews upload (especially
with PDFs mixed in) can hit fast. `.streamlit/config.toml` raises this to 1024MB (`[server]
maxUploadSize = 1024`) — bump it further if a batch still doesn't fit; it's a plain non-secret
config value, safe to edit and commit. Whatever sits in front of this app (nginx, an ALB, any
reverse proxy) may enforce its own request-body-size limit independently of Streamlit's — if
uploads still fail above roughly 200MB after this change, check that layer too (e.g. nginx's
`client_max_body_size`). There's no hard ceiling on the app side beyond available memory: the
whole zip is read into memory at once, so very large batches (multiple GB) need a host with the
RAM to match.

## Performance: parallel requests

Metacritic full-text fetching and both classification passes (main categories, then the
emergent-category follow-up pass) run concurrently in bounded batches rather than strictly one
review at a time — the sidebar's "Parallel requests" slider (default 5, range 1-10) controls how
many run at once. Set it to 1 to go back to fully sequential/one-at-a-time behavior. Turning it
up speeds up a large batch, but two tradeoffs are worth knowing: a high value can trip the
Anthropic/Bedrock API's per-minute rate limit on a big run, and on the Metacritic tab specifically
it can spike memory if several reviews fall back to a concurrent headless-browser (Playwright)
render at once — worth keeping lower on a memory-constrained host like Streamlit Community
Cloud's free tier.

## Auth: OTP now, SSO later

`app/auth.py` is deliberately isolated so the swap is contained to one file:

- **Now**: email in → SES sends a 6-digit code → verified → a signed token goes into
  `st.query_params` so the session survives page reloads (same approach as the house pattern),
  soft-restricted to `@segaamerica.com` via `ALLOWED_DOMAIN`. No audit log, no IP rate limiting
  — fine for a small pilot, not for anything wider.
- **Later**: replace `require_login()`'s body with your OIDC/SAML flow once available.
  Everything else reads `st.session_state["auth_email"]` and doesn't care how it got set.

## Known gaps carried over from the skill / POC

- Non-English outlets need the upload path (saved HTML) — live fetch is English-outlet-biased in
  practice since that's what was tested.
- Outlets that actively block automated fetching (e.g. IGN, Eurogamer in the P3R test) need the
  upload path too.
- The curated-outlet-list snapshot bundled here will go stale — Metacritic revises it "several
  times a year." Re-fetch live when you can; the app flags the snapshot date in the sidebar.
- This has been run once, on one title, at a 12-review sample. Pilot it on 2-3 more real titles
  with a producer checking the output before trusting it unsupervised.
