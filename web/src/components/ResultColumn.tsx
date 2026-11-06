import type { Hit } from "../api";

/**
 * One retriever's results.
 *
 * The two columns are deliberately not interchangeable. BM25 can say which
 * terms it matched but has no score comparable to a cosine similarity; the
 * dense retriever has a score but never saw a term. Each column says which
 * of those it is, rather than showing an empty field where the other one's
 * information would go.
 */
type Props = {
  kind: "dense" | "lexical";
  hits: Hit[] | undefined;
  elapsedMs: number | undefined;
  loading: boolean;
};

const LABEL = {
  dense: "Dense · bge-small",
  lexical: "BM25 · SQLite FTS5",
} as const;

const NOTE = {
  dense: "Cosine similarity over 384-dimensional float16 vectors.",
  lexical: "Ranked by BM25. FTS5 has no score comparable to a cosine, so none is shown.",
} as const;

function highlight(snippet: string, terms: string[]) {
  if (!terms.length || !snippet) return snippet;
  // Longest first, so "transformers" is not half-matched by "transform".
  const ordered = [...terms].sort((a, b) => b.length - a.length);
  const pattern = new RegExp(
    `(${ordered.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
    "gi",
  );
  return snippet.split(pattern).map((piece, index) =>
    index % 2 === 1 ? <mark key={index}>{piece}</mark> : piece,
  );
}

export function ResultColumn({ kind, hits, elapsedMs, loading }: Props) {
  return (
    <section className={`column ${kind}`}>
      <div className="column-head">
        <h2>{LABEL[kind]}</h2>
        {elapsedMs !== undefined && <span className="timing">{elapsedMs} ms</span>}
      </div>
      <p className="column-note">{NOTE[kind]}</p>

      {loading && (
        <ul className="hits">
          {[0, 1, 2, 3, 4].map((row) => (
            <li className="hit" key={row}>
              <span className="rank">{row + 1}</span>
              <div className="skeleton" style={{ width: `${88 - row * 11}%` }} />
              <p className="hit-snippet">
                <span className="skeleton" style={{ display: "block", width: "100%" }} />
              </p>
            </li>
          ))}
        </ul>
      )}

      {!loading && hits && hits.length > 0 && (
        <ul className="hits">
          {hits.map((hit) => (
            <li className="hit" key={hit.paper_id}>
              <span className="rank">{hit.rank}</span>
              <h3 className="hit-title">
                <a href={hit.url} target="_blank" rel="noreferrer noopener">
                  {hit.title || "Untitled"}
                </a>
              </h3>
              <div className="hit-meta">
                {hit.year && <span>{hit.year}</span>}
                {hit.score !== null && <span className="score">{hit.score.toFixed(3)}</span>}
                {hit.matched.length > 0 && (
                  <span>
                    matched {hit.matched.length} term{hit.matched.length === 1 ? "" : "s"}
                  </span>
                )}
              </div>
              {hit.snippet && (
                <p className="hit-snippet">{highlight(hit.snippet, hit.matched)}</p>
              )}
            </li>
          ))}
        </ul>
      )}

      {!loading && hits && hits.length === 0 && (
        <p className="empty">Nothing matched. BM25 needs the words; try different ones.</p>
      )}
    </section>
  );
}
