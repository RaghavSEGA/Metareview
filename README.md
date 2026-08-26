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
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in at least one LLM credential set
streamlit run app/streamlit_app.py
```

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

Put this behind whatever Sega already uses to expose internal tools (a reverse proxy, an
internal load balancer, a VPN-only network) — **do not expose this directly to the public
internet** even with the OTP gate in place. OTP-over-email is a reasonable stopgap for a small
trusted pilot, not a substitute for being on internal infrastructure.

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