// App.jsx — BodyPlot application shell
// Globals: React, ReactDOM, window.Viewer, window.StatCard

const { useState, useCallback, useRef, useEffect } = React;
const Viewer   = window.Viewer;
const StatCard = window.StatCard;

const PIPELINE_STAGES = [
  { id: 1, label: 'Human3R Inference', desc: 'Reconstruct 3D scene & body meshes' },
  { id: 2, label: 'Spatial Analytics',  desc: 'Proximity · Gaze · Contact · Metrics' },
  { id: 3, label: 'Complete',           desc: 'Viewer ready' },
];

// ── Design tokens ──────────────────────────────────────────────────────────────
const C = {
  bg:          '#09090b',
  sidebar:     '#0d0d0f',
  surface:     '#111116',
  border:      '#27272a',
  borderWarm:  '#44201a',

  textPrimary: '#fafaf9',
  textSecond:  '#a8a29e',
  textMuted:   '#78716c',
  textFaint:   '#44403c',

  amber:       '#d97706',
  amberBright: '#f59e0b',
  amberDim:    '#451a03',
  amberGlow:   'rgba(217,119,6,0.12)',

  red:         '#dc2626',
  redDim:      '#450a0a',

  green:       '#16a34a',
  greenDim:    '#052e16',

  mono:        '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
};

const HISTORY_KEY = 'sitl_history';

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}
function saveHistory(entry) {
  const h = getHistory().filter(e => e.id !== entry.id);
  h.unshift(entry);
  const trimmed = h.slice(0, 10);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  return trimmed;
}

// ── Upload screen — main canvas idle state ─────────────────────────────────────

function UploadScreen({ onFile, subsample, setSubsample, devJobId, setDevJobId, onDevLoad, history, onHistoryLoad }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = e => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) onFile(f);
  };

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 20,
      background: C.bg,
    }}>
      {/* Logo mark */}
      <div style={{ textAlign: 'center', marginBottom: 4 }}>
        {/* Crosshair / target reticle */}
        <svg width="48" height="48" viewBox="0 0 64 64" fill="none"
          style={{ display: 'block', margin: '0 auto 12px', opacity: 0.75 }}>
          <circle cx="32" cy="32" r="22" stroke={C.amber} strokeWidth="1.5" fill="none" />
          <circle cx="32" cy="32" r="10" stroke={C.amber} strokeWidth="1"   fill="none" opacity="0.6" />
          <circle cx="32" cy="32" r="2.5" fill={C.amber} opacity="0.9" />
          <line x1="32" y1="4"  x2="32" y2="18" stroke={C.amber} strokeWidth="1.5" strokeLinecap="round" />
          <line x1="32" y1="46" x2="32" y2="60" stroke={C.amber} strokeWidth="1.5" strokeLinecap="round" />
          <line x1="4"  y1="32" x2="18" y2="32" stroke={C.amber} strokeWidth="1.5" strokeLinecap="round" />
          <line x1="46" y1="32" x2="60" y2="32" stroke={C.amber} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <div style={{
          fontSize: 18, fontWeight: 700, color: C.textPrimary,
          letterSpacing: '0.18em', fontFamily: C.mono,
        }}>BODYPLOT</div>
        <div style={{
          fontSize: 9, color: C.amber, marginTop: 4,
          letterSpacing: '0.22em', textTransform: 'uppercase', fontFamily: C.mono,
        }}>── CASE FILE ANALYSIS ──</div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          width: 460, maxWidth: '88vw',
          border: `2px dashed ${dragging ? C.amber : C.border}`,
          borderRadius: 10, padding: '38px 24px',
          textAlign: 'center', cursor: 'pointer',
          background: dragging ? C.amberGlow : 'rgba(255,255,255,0.012)',
          transition: 'border-color .18s, background .18s',
        }}
      >
        <div style={{ fontSize: 36, marginBottom: 12, userSelect: 'none', opacity: dragging ? 1 : 0.6 }}>
          ◎
        </div>
        <div style={{ fontSize: 16, fontWeight: 600, color: C.textPrimary, marginBottom: 5 }}>
          Submit footage for analysis
        </div>
        <div style={{ fontSize: 11, color: C.textMuted, marginBottom: 20, fontFamily: C.mono }}>
          mp4 · mov · avi · mkv
        </div>
        <div style={{
          display: 'inline-block', padding: '7px 20px',
          background: C.amberDim, border: `1px solid ${C.amber}`,
          borderRadius: 5, color: C.amberBright, fontSize: 11,
          fontWeight: 600, fontFamily: C.mono, letterSpacing: '0.06em',
        }}>
          OPEN FILE
        </div>
        <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
          onChange={e => { const f = e.target.files[0]; if (f) onFile(f); }}
        />
      </div>

      {/* Subsample presets */}
      <div style={{ width: 460, maxWidth: '88vw' }}>
        <div style={{ ...LABEL, marginBottom: 7 }}>
          Subsample — every {subsample} frame{subsample > 1 ? 's' : ''}
        </div>
        <div style={{ display: 'flex', gap: 5 }}>
          {[1, 2, 3, 4, 8].map(n => (
            <button key={n} onClick={() => setSubsample(n)} style={{
              flex: 1, padding: '7px 0', fontSize: 11, cursor: 'pointer', borderRadius: 5,
              fontFamily: C.mono,
              background: subsample === n ? C.amberDim : C.surface,
              border: `1px solid ${subsample === n ? C.amber : C.border}`,
              color: subsample === n ? C.amberBright : C.textMuted,
              transition: 'all .12s',
            }}>{n}×</button>
          ))}
        </div>
      </div>

      {/* Dev load */}
      <div style={{ width: 460, maxWidth: '88vw', display: 'flex', gap: 7 }}>
        <input
          type="text" value={devJobId}
          onChange={e => setDevJobId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && onDevLoad()}
          placeholder="Load existing case id…"
          style={{
            flex: 1, padding: '8px 10px', fontSize: 11,
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 5, color: C.textPrimary, outline: 'none',
            fontFamily: C.mono,
          }}
        />
        <button onClick={onDevLoad} style={{
          padding: '8px 14px', fontSize: 11, fontWeight: 600,
          background: C.amberDim, border: `1px solid ${C.amber}`,
          borderRadius: 5, color: C.amber, cursor: 'pointer',
          whiteSpace: 'nowrap', fontFamily: C.mono, letterSpacing: '0.04em',
        }}>LOAD ↵</button>
      </div>

      {/* Recent cases */}
      {history.length > 0 && (
        <div style={{ width: 460, maxWidth: '88vw' }}>
          <div style={{ ...LABEL, marginBottom: 6 }}>Recent Cases</div>
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            {history.slice(0, 5).map(e => (
              <button key={e.id} onClick={() => onHistoryLoad(e.id)} style={{
                padding: '4px 10px', fontSize: 10, cursor: 'pointer', borderRadius: 4,
                background: C.surface, border: `1px solid ${C.border}`,
                color: C.amber, fontFamily: C.mono,
                transition: 'border-color .12s',
              }}>{e.id.slice(0, 8)}</button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Processing screen — main canvas during inference ──────────────────────────

function ProcessingScreen({ stage, progress, logs, phase }) {
  const STAGES = [
    { id: 1, label: 'Human3R Inference', desc: 'Reconstructing 3D scene & body meshes' },
    { id: 2, label: 'Spatial Analytics',  desc: 'Computing proximity · gaze · contact' },
    { id: 3, label: 'Complete',           desc: 'Viewer ready' },
  ];
  const cur  = STAGES.find(s => s.id === stage) || STAGES[0];
  const R    = 34;
  const CIRC = 2 * Math.PI * R;
  const pct  = stage === 1 ? (progress / 100) : stage > 1 ? 1 : 0;
  const lastLog = logs.length > 0 ? logs[logs.length - 1] : null;

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      background: C.bg, gap: 0,
    }}>
      {/* Progress ring */}
      <div style={{ position: 'relative', width: 88, height: 88, marginBottom: 22 }}>
        <svg width="88" height="88" viewBox="0 0 80 80"
          style={{ transform: 'rotate(-90deg)', display: 'block' }}>
          <circle cx="40" cy="40" r={R} fill="none" stroke={C.border} strokeWidth="5" />
          <circle cx="40" cy="40" r={R} fill="none" stroke={C.amber} strokeWidth="5"
            strokeDasharray={CIRC}
            strokeDashoffset={CIRC * (1 - pct)}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset .5s ease' }}
          />
        </svg>
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center', flexDirection: 'column',
        }}>
          <span style={{
            fontSize: 18, fontWeight: 700, color: C.amber,
            lineHeight: 1, fontFamily: C.mono,
          }}>{stage}</span>
          <span style={{ fontSize: 9, color: C.textFaint, fontFamily: C.mono }}>
            /{STAGES.length}
          </span>
        </div>
      </div>

      <div style={{ fontSize: 14, fontWeight: 600, color: C.textPrimary, marginBottom: 4, fontFamily: C.mono }}>
        {cur.label}
      </div>
      <div style={{ fontSize: 11, color: C.textMuted, marginBottom: 28 }}>{cur.desc}</div>

      {/* Stage pills */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 24 }}>
        {STAGES.map(s => {
          const done   = s.id < stage || phase === 'completed';
          const active = s.id === stage && phase === 'processing';
          return (
            <div key={s.id} style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '4px 12px', borderRadius: 20, fontSize: 10,
              fontFamily: C.mono,
              background:  done ? C.greenDim : active ? C.amberDim : C.surface,
              border:      `1px solid ${done ? C.green : active ? C.amber : C.border}`,
              color:       done ? C.green : active ? C.amberBright : C.textFaint,
              transition:  'all .3s',
            }}>
              <span>{done ? '✓' : active ? '●' : '○'}</span>
              <span>{s.label}</span>
            </div>
          );
        })}
      </div>

      {/* Latest log line */}
      {lastLog && (
        <div style={{
          fontFamily: C.mono, fontSize: 10, color: C.textFaint,
          maxWidth: 440, overflow: 'hidden', textOverflow: 'ellipsis',
          whiteSpace: 'nowrap', padding: '0 24px',
        }}>{lastLog}</div>
      )}
    </div>
  );
}

// ── Sidebar helpers ────────────────────────────────────────────────────────────

function StageRow({ stage, current, status }) {
  const done   = status === 'completed' || current > stage.id;
  const active = current === stage.id && status === 'processing';
  const failed = current === stage.id && status === 'failed';
  let bg = C.surface, border = C.border, textColor = C.textFaint;
  if (done)   { bg = C.greenDim; border = C.green;  textColor = C.textPrimary; }
  if (active) { bg = C.amberDim; border = C.amber;  textColor = C.textPrimary; }
  if (failed) { bg = C.redDim;   border = C.red;    textColor = C.textPrimary; }
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '6px 0' }}>
      <div style={{
        width: 22, height: 22, flexShrink: 0, borderRadius: '50%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 10, fontWeight: 700, marginTop: 1, fontFamily: C.mono,
        background: bg, border: `1.5px solid ${border}`, color: textColor,
        boxShadow: active ? `0 0 10px ${C.amber}44` : 'none', transition: 'all .2s',
      }}>
        {done ? '✓' : failed ? '✕' : stage.id}
      </div>
      <div>
        <div style={{ fontSize: 12, color: textColor, fontWeight: 500 }}>{stage.label}</div>
        <div style={{ fontSize: 10, color: C.textMuted, marginTop: 1 }}>{stage.desc}</div>
      </div>
    </div>
  );
}

function ProgressBar({ value }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ height: 3, background: C.border, borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          width: `${value}%`, height: '100%',
          background: `linear-gradient(90deg, ${C.amber}, ${C.amberBright})`,
          transition: 'width .4s ease', borderRadius: 2,
        }} />
      </div>
      <div style={{
        fontSize: 10, color: C.textMuted, marginTop: 2,
        textAlign: 'right', fontFamily: C.mono,
      }}>{value}%</div>
    </div>
  );
}

function LogConsole({ logs }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs.length]);
  return (
    <div style={{
      fontFamily: C.mono,
      fontSize: 10, lineHeight: 1.6, color: C.textMuted,
      overflowY: 'auto', padding: '8px 16px', flex: 1, minHeight: 0,
    }}>
      {logs.map((line, i) => (
        <div key={i} style={{
          color: /error|failed|crash/i.test(line) ? C.red
               : /complete|ready|done/i.test(line) ? C.green
               : C.textMuted,
        }}>{line}</div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

function HistoryPanel({ history, onLoad, onRemove }) {
  if (!history.length) return null;
  return (
    <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.border}` }}>
      <div style={{ ...LABEL, marginBottom: 6 }}>Case Archive</div>
      {history.slice(0, 8).map(e => (
        <div key={e.id} style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '4px 0', borderBottom: `1px solid ${C.border}22`,
        }}>
          <div onClick={() => onLoad(e.id)} style={{ flex: 1, cursor: 'pointer' }}>
            <div style={{ fontFamily: C.mono, fontSize: 10, color: C.amber }}>{e.id}</div>
            {e.filename && (
              <div style={{
                fontSize: 9, color: C.textMuted, marginTop: 1,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 150,
              }}>{e.filename}</div>
            )}
          </div>
          <span style={{ fontSize: 9, color: C.textFaint, whiteSpace: 'nowrap', fontFamily: C.mono }}>
            {e.ts ? new Date(e.ts).toLocaleDateString() : ''}
          </span>
          <button onClick={() => onRemove(e.id)} style={{
            background: 'none', border: 'none', color: C.textFaint,
            cursor: 'pointer', fontSize: 10, padding: '0 2px',
          }}>✕</button>
        </div>
      ))}
    </div>
  );
}

// ── Root App ───────────────────────────────────────────────────────────────────

function App() {
  const [phase,     setPhase]     = useState('idle');
  const [stage,     setStage]     = useState(0);
  const [progress,  setProgress]  = useState(0);
  const [logs,      setLogs]      = useState([]);
  const [results,   setResults]   = useState(null);
  const [subsample, setSubsample] = useState(2);
  const [error,     setError]     = useState(null);
  const [jobId,     setJobId]     = useState(null);
  const [devJobId,  setDevJobId]  = useState('0610cb8c');
  const [history,   setHistory]   = useState(getHistory);

  const isProcessing = phase === 'uploading' || phase === 'processing';

  // ── Core job loader ────────────────────────────────────────────────────────
  const loadJobById = useCallback(async (id) => {
    if (!id) return;
    const trimmed = id.trim();
    setError(null);
    try {
      const res = await fetch(`/dev/load/${trimmed}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResults(data);
      setPhase('completed');
      setJobId(data.job_id);
      setStage(3);
      setProgress(100);
      setLogs([`[dev] Loaded case ${data.job_id}`]);
      window.location.hash = '#' + data.job_id;
      setHistory(saveHistory({ id: data.job_id, filename: data.job_id, ts: Date.now() }));
    } catch (err) {
      setError(`Load failed: ${err.message}`);
    }
  }, []);

  // Auto-load from URL hash on mount
  useEffect(() => {
    const hashId = window.location.hash.slice(1);
    if (/^[0-9a-f]{8,}$/i.test(hashId)) loadJobById(hashId);
  }, []); // eslint-disable-line

  // ── Upload + pipeline ──────────────────────────────────────────────────────
  const startPipeline = useCallback(async (file) => {
    setPhase('uploading');
    setLogs([]); setResults(null); setError(null);
    setStage(1); setProgress(0);
    try {
      const form = new FormData();
      form.append('file', file);
      const upRes = await fetch('/upload', { method: 'POST', body: form });
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.status}`);
      const { job_id } = await upRes.json();
      setJobId(job_id);
      setLogs([`Uploaded: ${file.name}  (case ${job_id})`]);

      const runRes = await fetch(`/run/${job_id}?subsample=${subsample}`, { method: 'POST' });
      if (!runRes.ok) throw new Error(`Pipeline start failed: ${runRes.status}`);
      setPhase('processing');

      const sse = new EventSource(`/status/${job_id}`);
      sse.onmessage = e => {
        const state = JSON.parse(e.data);
        if (typeof state.stage    === 'number') setStage(state.stage);
        if (typeof state.progress === 'number') setProgress(state.progress);
        if (Array.isArray(state.logs))          setLogs(state.logs.slice(-100));

        if (state.status === 'completed') {
          sse.close();
          setPhase('completed');
          window.location.hash = '#' + job_id;
          setHistory(saveHistory({ id: job_id, filename: file.name, ts: Date.now() }));
          fetch(`/results/${job_id}`)
            .then(r => r.json())
            .then(setResults)
            .catch(err => setError('Could not fetch results: ' + err.message));
        } else if (state.status === 'failed') {
          sse.close();
          setPhase('failed');
          setError('Pipeline failed — see logs below.');
        }
      };
      sse.onerror = () => {
        sse.close();
        setPhase('failed');
        setError('Lost connection to server.');
      };
    } catch (err) {
      setPhase('failed');
      setError(err.message);
    }
  }, [subsample]);

  const loadDevJob = useCallback(() => loadJobById(devJobId.trim()), [devJobId, loadJobById]);

  const deleteJob = useCallback(async () => {
    if (!jobId) return;
    try { await fetch(`/jobs/${jobId}`, { method: 'DELETE' }); } catch (_) {}
    const h = getHistory().filter(e => e.id !== jobId);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
    setHistory(h);
    reset();
  }, [jobId]);

  const reset = () => {
    setPhase('idle'); setJobId(null); setStage(0);
    setProgress(0);   setLogs([]);    setResults(null); setError(null);
    window.location.hash = '';
  };

  const removeFromHistory = useCallback(id => {
    const h = getHistory().filter(e => e.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
    setHistory(h);
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────

  const mainCanvas = results
    ? <Viewer plyUrl={results.ply_url} jsonUrl={results.json_url} />
    : isProcessing
    ? <ProcessingScreen stage={stage} progress={progress} logs={logs} phase={phase} />
    : <UploadScreen
        onFile={startPipeline}
        subsample={subsample} setSubsample={setSubsample}
        devJobId={devJobId}   setDevJobId={setDevJobId}
        onDevLoad={loadDevJob}
        history={history}
        onHistoryLoad={loadJobById}
      />;

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* ── Main canvas — LEFT ─────────────────────────────────────────── */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {mainCanvas}
      </div>

      {/* ── Sidebar — RIGHT ────────────────────────────────────────────── */}
      <div style={{
        width: 280, minWidth: 280, background: C.sidebar,
        borderLeft: `1px solid ${C.border}`,
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>

        {/* Header */}
        <div style={{
          padding: '14px 16px 12px',
          borderBottom: `1px solid ${C.border}`,
        }}>
          <div style={{
            fontSize: 14, fontWeight: 700, color: C.textPrimary,
            letterSpacing: '0.16em', fontFamily: C.mono,
          }}>BODYPLOT</div>
          <div style={{
            fontSize: 8, color: C.amber, marginTop: 3,
            letterSpacing: '0.16em', fontFamily: C.mono,
          }}>── CASE FILE ANALYSIS ──</div>
        </div>

        {/* Upload (compact) */}
        {(phase === 'idle' || phase === 'completed' || phase === 'failed') && (
          <div style={{ padding: '12px 16px', borderBottom: `1px solid ${C.border}` }}>
            <div style={{ ...LABEL, marginBottom: 8 }}>Footage</div>
            <CompactDropZone onFile={startPipeline} disabled={isProcessing} />

            {(phase === 'completed' || phase === 'failed') && (
              <div style={{ display: 'flex', gap: 6, marginTop: 9 }}>
                <button onClick={reset} style={{
                  flex: 1, padding: '7px 0',
                  background: C.surface, border: `1px solid ${C.border}`,
                  color: C.textSecond, borderRadius: 5, cursor: 'pointer',
                  fontSize: 11, fontFamily: C.mono,
                }}>↺  New case</button>
                {phase === 'completed' && jobId && (
                  <button onClick={deleteJob} title="Delete case outputs" style={{
                    padding: '7px 9px', background: 'transparent',
                    border: `1px solid ${C.redDim}`, color: C.red,
                    borderRadius: 5, cursor: 'pointer', fontSize: 11,
                  }}>🗑</button>
                )}
              </div>
            )}

            {phase === 'idle' && (
              <div style={{ marginTop: 10 }}>
                <div style={{ ...LABEL, marginBottom: 5 }}>
                  Subsample — every {subsample} frame{subsample > 1 ? 's' : ''}
                </div>
                <div style={{ display: 'flex', gap: 3, marginBottom: 5 }}>
                  {[1, 2, 3, 4, 8].map(n => (
                    <button key={n} onClick={() => setSubsample(n)} style={{
                      flex: 1, padding: '3px 0', fontSize: 10, cursor: 'pointer', borderRadius: 4,
                      fontFamily: C.mono,
                      background: subsample === n ? C.amberDim : 'transparent',
                      border: `1px solid ${subsample === n ? C.amber : C.border}`,
                      color: subsample === n ? C.amberBright : C.textMuted,
                    }}>{n}</button>
                  ))}
                </div>
                <input type="range" min={1} max={8} value={subsample}
                  onChange={e => setSubsample(+e.target.value)}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        )}

        {/* Dev load (idle only) */}
        {phase === 'idle' && (
          <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.border}` }}>
            <div style={{ ...LABEL, marginBottom: 5 }}>Load Existing Case</div>
            <div style={{ display: 'flex', gap: 5 }}>
              <input
                type="text" value={devJobId}
                onChange={e => setDevJobId(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && loadDevJob()}
                placeholder="case id"
                style={{
                  flex: 1, padding: '5px 7px', fontSize: 10,
                  background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 4, color: C.textPrimary, outline: 'none',
                  fontFamily: C.mono,
                }}
              />
              <button onClick={loadDevJob} style={{
                padding: '5px 9px', fontSize: 10, fontWeight: 600,
                background: C.amberDim, border: `1px solid ${C.amber}`,
                borderRadius: 4, color: C.amber, cursor: 'pointer', fontFamily: C.mono,
              }}>Load</button>
            </div>
          </div>
        )}

        {/* Case history */}
        {phase === 'idle' && (
          <HistoryPanel history={history} onLoad={loadJobById} onRemove={removeFromHistory} />
        )}

        {/* Pipeline stages */}
        {phase !== 'idle' && (
          <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.border}` }}>
            <div style={LABEL}>Pipeline</div>
            {stage === 1 && progress > 0 && <ProgressBar value={progress} />}
            {PIPELINE_STAGES.map(s => (
              <StageRow key={s.id} stage={s} current={stage} status={phase} />
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            padding: '9px 16px', borderBottom: `1px solid ${C.border}`,
            fontSize: 11, color: C.red, lineHeight: 1.5, fontFamily: C.mono,
          }}>
            ✕  {error}
          </div>
        )}

        {/* Logs */}
        {logs.length > 0 && (
          <>
            <div style={{ padding: '8px 16px 0', ...LABEL }}>Case Log</div>
            <LogConsole logs={logs} />
          </>
        )}
      </div>
    </div>
  );
}

// ── Compact drop zone for sidebar ─────────────────────────────────────────────

function CompactDropZone({ onFile, disabled }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef(null);
  const handleDrop = e => {
    e.preventDefault(); setDrag(false);
    if (disabled) return;
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) onFile(f);
  };
  return (
    <div
      onClick={() => !disabled && ref.current?.click()}
      onDragOver={e => { e.preventDefault(); if (!disabled) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      style={{
        border: `1.5px dashed ${drag ? C.amber : C.border}`,
        borderRadius: 6, padding: '14px 10px', textAlign: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        background: drag ? C.amberGlow : 'transparent',
        transition: 'border-color .15s, background .15s',
        userSelect: 'none',
      }}
    >
      <div style={{ fontSize: 18, marginBottom: 4, opacity: disabled ? 0.3 : 0.65 }}>◎</div>
      <div style={{ fontSize: 10, color: disabled ? C.textFaint : C.textSecond }}>
        {disabled ? 'Processing…' : 'Drop or click to upload'}
      </div>
      <input ref={ref} type="file" accept="video/*" style={{ display: 'none' }}
        onChange={e => { const f = e.target.files[0]; if (f) onFile(f); }}
      />
    </div>
  );
}

const LABEL = {
  fontSize: 9, color: '#78716c',
  textTransform: 'uppercase', letterSpacing: '0.10em',
  fontFamily: '"JetBrains Mono", "Fira Code", monospace',
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
