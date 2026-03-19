'use client';
import React from 'react';
import { apiIngest, apiMetrics } from '../lib/api';

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<any>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = async () => {
    const m = await apiMetrics();
    setMetrics(m);
  };

  const ingest = async () => {
    setBusy(true);
    try {
      await apiIngest();
      alert("Documents indexed successfully ✅");
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => { refresh(); }, []);

  return (
    <div className="card">
      <h2>Admin</h2>
      <div style={{display:'flex', gap:8, marginBottom:8}}>
        <button onClick={ingest} disabled={busy} style={{padding:'8px 12px', borderRadius:8, border:'1px solid #111', background:'#fff'}}>
          {busy ? 'Indexing...' : 'Ingest sample docs'}
        </button>
        <button onClick={refresh} style={{padding:'8px 12px', borderRadius:8, border:'1px solid #111', background:'#fff'}}>Refresh metrics</button>
      </div>
      {metrics && (
        <div className="code">
          <div>Docs: {metrics.total_docs}</div>
          <div>Chunks: {metrics.total_chunks}</div>
          <div>Retrieval: {metrics.avg_retrieval_latency_ms} ms</div>
          <div>Generation: {metrics.avg_generation_latency_ms} ms</div>
        </div>
      )}
    </div>
  );
}
