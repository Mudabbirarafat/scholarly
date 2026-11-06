import type { Quality, RetrieverMetrics } from "../api";

/**
 * The published measurement, not a live one.
 *
 * These numbers come from a pipeline run against a frozen query set on a
 * complete index, and the panel says which run. A figure computed here, in a
 * browser, against whatever the instance happens to hold would look exactly
 * as official and mean nothing - which is the whole reason the project keeps
 * the measurement in CI.
 */
const ROWS: { key: keyof RetrieverMetrics; label: string; digits: number; suffix?: string }[] = [
  { key: "recall@10", label: "recall@10", digits: 4 },
  { key: "recall@50", label: "recall@50", digits: 4 },
  { key: "mrr@50", label: "MRR@50", digits: 4 },
  { key: "p50_ms", label: "latency p50", digits: 1, suffix: " ms" },
  { key: "p95_ms", label: "latency p95", digits: 1, suffix: " ms" },
];

function relative(dense?: number, lexical?: number): string {
  if (dense === undefined || lexical === undefined || !lexical) return "";
  const change = (dense / lexical - 1) * 100;
  return `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`;
}

export function QualityPanel({ quality }: { quality: Quality }) {
  const { lexical, dense, corpus, queries } = quality;
  if (!lexical || !dense) return null;

  return (
    <section className="quality">
      <h2>What this is measured at</h2>
      <p className="lede">
        Both retrievers run against the same frozen query set on the same corpus. A paper&rsquo;s
        abstract is the query and the works it cites are the correct answers, so the ground truth
        costs nothing and nobody labelled it by hand.
      </p>

      <table className="metrics">
        <thead>
          <tr>
            <th scope="col">metric</th>
            <th scope="col">BM25</th>
            <th scope="col">dense</th>
            <th scope="col">change</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ key, label, digits, suffix }) => {
            const left = lexical[key];
            const right = dense[key];
            if (left === undefined || right === undefined) return null;
            const lowerIsBetter = key.startsWith("p");
            const denseWins = lowerIsBetter ? right < left : right > left;
            return (
              <tr key={key}>
                <td>{label}</td>
                <td>
                  {left.toFixed(digits)}
                  {suffix}
                </td>
                <td className={denseWins ? "win" : undefined}>
                  {right.toFixed(digits)}
                  {suffix}
                </td>
                <td className="delta">{lowerIsBetter ? "" : relative(right, left)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {corpus && (
        <p className="ceiling">
          <b>The ceiling, quoted beside the number.</b> Only{" "}
          {corpus.retrievable_pct}% of this corpus&rsquo;s{" "}
          {corpus.citation_edges.toLocaleString()} citation edges point at a document the index
          holds; the rest are journal and conference papers an arXiv-only corpus cannot contain,
          and they are excluded from the answer key. A recall of{" "}
          {dense["recall@10"].toFixed(3)} is measured against what a perfect retriever{" "}
          <i>could</i> return, not against every work each paper cited.
        </p>
      )}

      <p className="provenance">
        {corpus && <>{corpus.documents.toLocaleString()} documents · </>}
        {queries && (
          <>
            {queries.count.toLocaleString()} {queries.split} queries from a pool of{" "}
            {queries.pool.toLocaleString()} · {queries.relevant_per_query} relevant each ·{" "}
          </>
        )}
        {quality.commit && <>commit {quality.commit} · </>}
        {quality.run_url ? (
          <a href={quality.run_url} target="_blank" rel="noreferrer noopener">
            pipeline run {quality.run}
          </a>
        ) : (
          <>run {quality.run}</>
        )}
      </p>
    </section>
  );
}
