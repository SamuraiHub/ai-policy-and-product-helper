export const metadata = { title: 'AI Policy Helper' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
    <body>
      <div style={{
        borderBottom: '1px solid #eee',
        padding: '12px 24px',
        marginBottom: 24,
        background: '#fff'
      }}>
        <b>AI Policy Helper</b>
      </div>

      <div style={{ maxWidth: 1000, margin: '0 auto', padding: 24 }}>
        {children}
      </div>
    </body>
    </html>
  );
}
