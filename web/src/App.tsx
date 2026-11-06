import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getHealth,
  search as runSearch,
  type Health,
  type Retriever,
  type SearchResponse,
} from "./api";
import { QualityPanel } from "./components/QualityPanel";
import { ResultColumn } from "./components/ResultColumn";

const EXAMPLES = [
  "contrastive learning of sentence representations without labels",
  "why do deep networks generalise despite being overparameterised",
  "retrieval augmented generation for long documents",
  "differential privacy in federated training",
];

const COLUMNS: Record<Retriever, ("dense" | "lexical")[]> = {
  both: ["dense", "lexical"],
  dense: ["dense"],
  lexical: ["lexical"],
};

export default function App() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(10);
  const [retriever, setRetriever] = useState<Retriever>("both");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  // While the instance is warming, keep asking: a free tier wakes in seconds
  // and the interface should notice without the visitor reloading.
  useEffect(() => {
    let live = true;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const next = await getHealth();
        if (!live) return;
        setHealth(next);
        if (next.warmup.state === "warming") timer = window.setTimeout(poll, 2000);
      } catch {
        if (live) timer = window.setTimeout(poll, 5000);
      }
    };

    poll();
    return () => {
      live = false;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  const submit = useCallback(
    async (text: string, which: Retriever, howMany: number) => {
      const trimmed = text.trim();
      if (trimmed.length < 2) return;

      // A visitor typing a new query while one is in flight should not have
      // the old answer arrive last and win.
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;

      setLoading(true);
      setProblem(null);
      try {
        setResponse(await runSearch(trimmed, howMany, which, controller.signal));
      } catch (error) {
        if (controller.signal.aborted) return;
        setProblem(
          error instanceof ApiError
            ? error.message
            : "Could not reach the search service. It may be waking up.",
        );
        setResponse(null);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [],
  );

  const warming = health?.warmup.state === "warming";
  const failed = health?.warmup.state === "error";
  const documents = health?.index.documents ?? 0;

  // The published metrics were measured on a complete index. If this instance
  // is serving fewer vectors than that, searching it is not the thing that was
  // measured, and saying so is the same discipline the CI gate applies when it
  // refuses to compare a partial index against BM25.
  const measuredOn = health?.quality?.dense?.index_size ?? 0;
  const subset =
    !warming && !failed && documents > 0 && measuredOn > 0 && documents < measuredOn * 0.99;

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1 className="wordmark">
            scholarly<span>.</span>
          </h1>
          <p className="tagline">
            Semantic search over computer-science abstracts, with the measured accuracy of both
            retrievers published beside the results.
          </p>
        </div>
        <div className="chips">
          <span className="chip">
            <i className={`dot ${health?.warmup.state ?? ""}`} />
            {/* "ready" is a claim about the server, so it is only made once
                the server has said so. Before the first health response there
                is nothing to claim, and an unreachable API is not readiness. */}
            {!health
              ? "connecting"
              : warming
                ? "waking"
                : failed
                  ? "index unavailable"
                  : "ready"}
          </span>
          {documents > 0 && (
            <span className="chip">
              <b>{documents.toLocaleString()}</b> documents
            </span>
          )}
          {health?.index.megabytes ? (
            <span className="chip">
              <b>{health.index.megabytes}</b> MiB · float16
            </span>
          ) : null}
        </div>
      </header>

      <form
        className="searchbar"
        onSubmit={(event) => {
          event.preventDefault();
          void submit(query, retriever, k);
        }}
      >
        <div className="field">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="1.6" />
            <path d="M11 11l3.5 3.5" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Describe the paper you are looking for, in plain English"
            aria-label="Search query"
            maxLength={2000}
            autoFocus
          />
        </div>
        <div className="toggle" role="group" aria-label="Retriever">
          {(["both", "dense", "lexical"] as Retriever[]).map((option) => (
            <button
              type="button"
              key={option}
              aria-pressed={retriever === option}
              onClick={() => {
                setRetriever(option);
                if (response) void submit(query, option, k);
              }}
            >
              {option === "lexical" ? "BM25" : option === "dense" ? "Dense" : "Both"}
            </button>
          ))}
        </div>
        <button className="primary" type="submit" disabled={loading || query.trim().length < 2}>
          {loading ? "Searching" : "Search"}
        </button>
      </form>

      <div className="controls">
        <label>
          results
          <select
            value={k}
            onChange={(event) => {
              const next = Number(event.target.value);
              setK(next);
              if (response) void submit(query, retriever, next);
            }}
          >
            {[5, 10, 20, 50].map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <span>
          {health?.index.model ? `encoder ${health.index.model}` : "encoder loading"}
        </span>
      </div>

      {warming && (
        <p className="banner warn">
          <span>
            <b>Waking the index.</b> This runs on a free instance that sleeps, and reading{" "}
            {documents ? documents.toLocaleString() : "the"} vectors back takes a few seconds.
            BM25 answers now; dense joins when the vectors are in memory
            {health?.warmup.waiting_seconds ? ` (${health.warmup.waiting_seconds}s so far)` : ""}.
          </span>
        </p>
      )}

      {failed && (
        <p className="banner bad">
          <span>
            <b>The vector index did not load.</b> {health?.warmup.error}. BM25 still works, so the
            lexical column is real and the dense one is unavailable.
          </span>
        </p>
      )}

      {subset && (
        <p className="banner">
          <span>
            <b>This instance holds {documents.toLocaleString()} of{" "}
            {measuredOn.toLocaleString()} documents.</b> The numbers at the bottom of the page were
            measured on the complete index in CI, so treat what you search here as a working
            sample rather than the thing that was measured.
          </span>
        </p>
      )}

      {problem && (
        <p className="banner bad">
          <span>{problem}</span>
        </p>
      )}

      {(response || loading) && (
        <div className="columns">
          {COLUMNS[response?.retriever ?? retriever].map((kind) => (
            <ResultColumn
              key={kind}
              kind={kind}
              hits={response?.results[kind]}
              elapsedMs={response?.elapsed_ms[kind]}
              loading={loading}
            />
          ))}
        </div>
      )}

      {!response && !loading && (
        <div className="examples">
          <p>Nothing to compare yet. Try one of these:</p>
          <div>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setQuery(example);
                  void submit(example, retriever, k);
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      {health?.quality && <QualityPanel quality={health.quality} />}

      <footer className="foot">
        <span>
          Built on OpenAlex (CC0). Encoding and evaluation run in CI; this instance only embeds
          your query and searches precomputed vectors.
        </span>
        <a href="https://github.com/Mudabbirarafat/scholarly" target="_blank" rel="noreferrer">
          source on GitHub
        </a>
      </footer>
    </div>
  );
}
