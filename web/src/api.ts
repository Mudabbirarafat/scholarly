/**
 * The one place that knows the server exists.
 *
 * `VITE_API_BASE` is empty in development and in production on Vercel,
 * because both put the API on the same origin - Vite proxies it, Vercel
 * rewrites it. It exists only as an escape hatch for pointing a local UI at
 * a deployed API.
 */
const BASE = import.meta.env.VITE_API_BASE ?? "";

export type Hit = {
  paper_id: string;
  rank: number;
  score: number | null;
  title: string;
  year: number | null;
  snippet: string;
  url: string;
  matched: string[];
};

export type Retriever = "both" | "dense" | "lexical";

export type SearchResponse = {
  query: string;
  k: number;
  retriever: Retriever;
  results: Partial<Record<"dense" | "lexical", Hit[]>>;
  elapsed_ms: Partial<Record<"dense" | "lexical", number>>;
};

export type Warmup = {
  state: "ready" | "warming" | "error";
  error: string | null;
  load_seconds: number | null;
  waiting_seconds: number | null;
};

export type Health = {
  status: string;
  retrievers: string[];
  index: {
    documents: number;
    model: string | null;
    megabytes: number;
    dtype: string;
  };
  quality: Quality;
  warmup: Warmup;
};

export type RetrieverMetrics = {
  "recall@10": number;
  "recall@50": number;
  "mrr@50": number;
  p50_ms?: number;
  p95_ms?: number;
  index_size?: number;
};

export type Quality = {
  run?: number | string;
  run_url?: string;
  commit?: string;
  measured_at?: string;
  corpus?: { documents: number; citation_edges: number; retrievable_pct: number };
  queries?: { split: string; count: number; relevant_per_query: number; pool: number };
  lexical?: RetrieverMetrics;
  dense?: RetrieverMetrics;
};

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    // FastAPI puts the useful part in `detail`; a 503 from a waking instance
    // says something worth showing the visitor rather than "request failed".
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail;
    } catch {
      /* a non-JSON error body is still an error */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const getHealth = (signal?: AbortSignal) => get<Health>("/api/health", signal);

export const search = (
  query: string,
  k: number,
  retriever: Retriever,
  signal?: AbortSignal,
) =>
  get<SearchResponse>(
    `/api/search?q=${encodeURIComponent(query)}&k=${k}&retriever=${retriever}`,
    signal,
  );

export { ApiError };
