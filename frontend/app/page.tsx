import Chat from '../components/Chat'
import AdminPanel from '../components/AdminPanel'

export default function Page() {
  return (
    <div>
      <h1 style={{ marginBottom: 4 }}>AI Policy & Product Helper</h1>
      <p style={{ color: '#666', marginBottom: 24 }}>
        Local-first RAG assistant with grounded answers and citations.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <div>
          <AdminPanel />
          <div className="card">
            <h3>Quick Test</h3>
            <ol style={{ paddingLeft: 16 }}>
              <li>Click <b>Ingest</b></li>
              <li>Ask: <i>Can a customer return a damaged blender after 20 days?</i></li>
              <li>Ask: <i>What’s the shipping SLA to East Malaysia for bulky items?</i></li>
            </ol>
          </div>
        </div>

        <Chat />
      </div>
    </div>
  );
}
