# Deploying it

Two pieces in two places, for one measured reason each.

| | where | why |
|---|---|---|
| React UI | Vercel | Static, on a CDN, and instant even while the API is asleep — so a visitor sees "waking the index" rather than a blank tab. |
| FastAPI + index | Render | The search needs a warm in-memory index. Vercel's Python functions are stateless with a 250 MB bundle cap, so every cold start would re-download a 130 MB model and reload the vectors. |

The numbers behind that split come from `scripts/serving_budget.py`, which
anyone can re-run:

```
python + numpy                        30.7 MB
encoder loaded                       203.0 MB
float16 index beside it (114,581)    288.6 MB   peak 324.6
search                               291.0 MB   150 ms
```

291 MB against a 512 MB instance. It fits, with about 220 MB spare — and
that spare is why `render.yaml` pins `--workers 1`. Each worker loads its own
encoder and its own copy of the index, so two workers is 580 MB on a 512 MB
box: an out-of-memory kill on the first concurrent request, not a slow page.

## 1. Build the bundle

The pipeline database is 700 MB and holds citations, the frozen query set and
ingest history, none of which serving reads. `bundle` writes a second SQLite
with papers, vectors and a full-text index, and nothing else:

```bash
python -m scholarly.cli bundle --out data/serving.db --limit 5000
```

`--limit` is the knob that decides what the deployed instance searches.
Leaving it off bundles everything encoded. The UI reads the size from
`/api/health` and says plainly when it is holding fewer documents than the
published measurement was taken on, so a subset is honest rather than
misleading — it is the same discipline the CI gate applies when it refuses to
compare a partial index against BM25.

The bundle is committed so a deploy needs no artifact plumbing. For the whole
corpus, publish it as a release asset instead and set `BUNDLE_URL`: the API
downloads it on first boot, to a temporary name and then renames, so a
half-written database is never served.

## 2. The API on Render

With `render.yaml` in the repository, use **New → Blueprint** and point it at
the repo. Or by hand: **New → Web Service**, Python, free plan, and

- build: `pip install -e ".[serve]"`
- start: `uvicorn scholarly.api:app --host 0.0.0.0 --port $PORT --workers 1`
- health check path: `/api/health`

Environment:

| key | value |
|---|---|
| `DATABASE_URL` | `sqlite:///data/serving.db` |
| `QUALITY_FILE` | `data/quality.json` |
| `ALLOWED_ORIGINS` | the Vercel URL, once step 3 gives you one |
| `BUNDLE_URL` | only if the bundle is not committed |

The free instance sleeps after 15 minutes. Waking it means loading vectors
and the encoder — `/api/health` answers `warming` throughout, deliberately,
because the port opens before the index is ready. Blocking startup on the
load instead would hand the first visitor a gateway timeout.

## 3. The UI on Vercel

**New Project → import the repo**, then set **Root Directory** to `web`.
Vercel detects Vite; `web/vercel.json` pins the build anyway.

Nothing else to configure. `web/vercel.json` already rewrites `/api/*` to the
Render service:

```json
"rewrites": [
  { "source": "/api/:path*", "destination": "https://scholarly-api-mf6f.onrender.com/api/:path*" }
]
```

That is deliberate rather than the alternative. Vercel performs the rewrite
server side, so the browser only ever talks to one origin: no CORS, nothing to
keep in sync between two dashboards, and the Render URL is not exposed in the
page. `ALLOWED_ORIGINS` and `VITE_API_BASE` both stay unset, and they exist
for the other arrangement — point `VITE_API_BASE` at the API and list the
Vercel domain in `ALLOWED_ORIGINS` if you would rather call it cross-origin.

The one cost of the rewrite is that the service URL lives in a committed
file, so renaming the Render service means editing it.

## 4. Check it

```bash
curl https://scholarly-api-mf6f.onrender.com/api/health
```

`index.documents` should match the bundle, and `warmup.state` should reach
`ready`. `ready` means the encoder is loaded too, not just the vectors — the
first encode call costs 6.7 seconds and is paid during warm-up, because a
flag that goes up before that is a promise the first visitor finds out is
false.

## Local development

```bash
# API, against whatever warehouse you have
DATABASE_URL=sqlite:///scholarly.db python -m uvicorn scholarly.api:app --port 8000

# UI, proxying /api to it
cd web && npm install && npm run dev
```

One thing worth knowing locally: an encode holds an exclusive write
transaction for its whole run, so the database it is writing cannot be read
while it works. That is not a problem in the deployed shape — CI writes, the
instance reads a published bundle — but it does mean you cannot point the API
at a warehouse that is mid-encode.
