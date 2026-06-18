// Viewer.jsx v13 — BodyPlot crime-board aesthetic
// Globals: THREE r134 (+ OrbitControls + PLYLoader), React 18

const { useEffect, useRef, useState, useCallback } = React;

// ── Coordinate helper (module-level — used by overlay effect AND animation loop)
const fy = ([x, y, z]) => new THREE.Vector3(x, -y, z);

// ── Design tokens (mirrors App.jsx) ──────────────────────────────────────────
const V = {
  bg:          '#09090b',
  surface:     '#111116',
  surface2:    '#18181b',
  border:      '#27272a',
  borderWarm:  '#44201a',
  textPrimary: '#fafaf9',
  textSecond:  '#a8a29e',
  textMuted:   '#78716c',
  textFaint:   '#44403c',
  amber:       '#d97706',
  amberBright: '#f59e0b',
  amberDim:    '#451a03',
  amberGlow:   'rgba(217,119,6,0.10)',
  red:         '#dc2626',
  green:       '#16a34a',
  mono:        '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
};

// ── Constants ─────────────────────────────────────────────────────────────────

// Per-subject marker colours — vivid, well-separated, cool-leaning categorical set
// (no amber/orange/terracotta, which read as "Claude" against the dark theme).
const PALETTE = [
  0x3b82f6, 0xec4899, 0x22c55e, 0x8b5cf6, 0x06b6d4,
  0xf43f5e, 0xeab308, 0xd946ef, 0x14b8a6, 0x818cf8,
];

const PALETTE_HEX = [
  '#3b82f6', '#ec4899', '#22c55e', '#8b5cf6', '#06b6d4',
  '#f43f5e', '#eab308', '#d946ef', '#14b8a6', '#818cf8',
];

const ACTION_COLORS = {
  stationary: '#78716c',
  walking:    '#3b82f6',
  running:    '#22c55e',
  sitting:    '#a78bfa',
  reaching:   '#f59e0b',
  bending:    '#ef4444',
};

// Hall (1966) proxemics zone colours — already warm/red, fit the crime-board perfectly
const ZONE_COLORS = {
  intimate: 0xdc2626,   // red   < 0.45 m
  personal: 0xf97316,   // orange 0.45–1.2 m
  social:   0xd97706,   // amber  1.2–3.7 m
  public:   0x44403c,   // stone  > 3.7 m (not rendered as edge)
};

const ZONE_HEX = {
  intimate: '#dc2626',
  personal: '#f97316',
  social:   '#d97706',
  public:   '#44403c',
};

const SMPLX_BONES = [
  [0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
  [16,18],[17,19],[18,20],[19,21],
];
const BONE_RADII = [
  0.022,0.022,0.022,0.016,0.016,0.022,0.013,0.013,0.022,
  0.009,0.009,0.020,0.015,0.015,0.020,0.015,0.015,0.013,0.013,0.009,0.009,
];
const JOINT_RADII = [
  0.040,0.030,0.030,0.028,0.026,0.026,0.028,0.020,0.020,0.028,
  0.016,0.016,0.028,0.020,0.020,0.050,0.026,0.026,0.022,0.022,0.016,0.016,
];
const TRAIL_LEN   = 30;
const SPEED_STEPS = [0.25, 0.5, 1, 2, 4];
const DEFAULT_LAYERS = {
  cloud: true, skeleton: true, mesh: true, gaze: true,
  trails: true, interactions: true, heatmap: true, grid: true,
};

const CAM_BTNS = [
  { id: 'persp',  label: 'Persp'  },
  { id: 'front',  label: 'Front'  },
  { id: 'follow', label: 'Follow' },
];

// ── Metric definitions ────────────────────────────────────────────────────────

const METRIC_DEFS = {
  social_engagement_pct: {
    label:       'Social Engagement',
    description: "Percentage of frames where at least one pair is within Hall's personal-distance zone (≤ 1.2 m), indicating active interpersonal interaction.",
    formula:     'frames_with_any_pair ≤ 1.2 m / total_frames × 100%',
    research:    'Hall (1966) proxemics — personal zone: 45–120 cm.',
    accent:      '#34d399',
  },
  avg_inter_human_distance: {
    label:       'Avg Inter-Subject Distance',
    description: 'Mean 3D Euclidean distance between all detected subjects, averaged across all pairs and all frames.',
    formula:     'mean(‖pos_i − pos_j‖₂)  ∀ pairs (i,j), ∀ frames',
    research:    'Spatial cohesion proxy (Cristani et al., 2011; Hall, 1966).',
    accent:      '#3b82f6',
  },
  scene_utilization_pct: {
    label:       'Scene Utilization',
    description: 'Percentage of 0.5 m³ voxels inside the scene bounding box that were occupied by at least one subject at any point in the footage.',
    formula:     'occupied_0.5 m_voxels / total_scene_voxels × 100%',
    research:    'Measures spatial coverage and mobility range within the scene.',
    accent:      '#818cf8',
  },
  gaze_convergence_events: {
    label:       'Gaze Convergence Events',
    description: 'Number of frames where at least one pair of subjects have gaze rays whose closest-approach distance is < 1 m, suggesting mutual visual attention.',
    formula:     'min‖closest_approach(ray_i, ray_j)‖ < 1.0 m, counted per frame',
    research:    'Mutual gaze: key indicator of social attention (Kendon, 1967; Argyle & Cook, 1976).',
    accent:      '#ec4899',
  },
  personal_space_pct: {
    label:       'Intimate Zone Violations',
    description: "Percentage of frames where any pair of subjects is within Hall's intimate-distance zone (< 0.45 m), indicating very close physical proximity or contact.",
    formula:     'frames_with_any_pair < 0.45 m / total_frames × 100%',
    research:    'Hall (1966) intimate zone: 0–45 cm. Indicates physical contact or crowding.',
    accent:      '#f43f5e',
  },
  avg_speed_mps: {
    label:       'Average Movement Speed',
    description: 'Mean movement speed of all tracked subjects in metres per second, normalised by the clip\'s effective frame rate (fps ÷ subsample), so it is correct at any subsample.',
    formula:     'mean(‖Δpelvis‖ × effective_fps)  [m/s]',
    research:    'Pelvis velocity is a standard locomotion proxy in SMPL-X-based tracking.',
    accent:      '#22d3ee',
  },
  approach_events: {
    label:       'Approach Events',
    description: 'Number of times any pair transitions into an approaching state (inter-subject distance derivative changes from positive or zero to negative).',
    formula:     'count of sign(Δdist) transitions: non-negative → negative, per pair',
    research:    'Approach–avoidance dynamics predict social engagement (Goffman, 1971).',
    accent:      '#14b8a6',
  },
  peak_occupancy: {
    label:       'Peak Occupancy',
    description: 'Maximum number of subjects simultaneously detected and tracked in any single video frame.',
    formula:     'max(|subjects_per_frame|) across all frames',
    research:    'Relevant for crowding analysis and group dynamics (Stokols, 1972).',
    accent:      '#eab308',
  },
  group_count: {
    label:       'Group Formations',
    description: 'Number of times a new conversational group (F-formation) forms — subjects close enough and oriented toward each other to share an interaction space.',
    formula:     'count of frames where #groups increases (connected components on the proximity+orientation graph)',
    research:    "Kendon (1990) F-formation / o-space; Cristani et al. (2011).",
    accent:      '#34d399',
  },
  event_count: {
    label:       'Detected Events',
    description: 'Total auto-detected highlights: meetings (pair enters personal zone), group formations, sitting transitions, and the single closest-contact moment.',
    formula:     'Σ (meetings + group_forms + sits + closest_contact)',
    research:    'Event segmentation of social interaction streams.',
    accent:      '#818cf8',
  },
};

// ── StatCard (exported to App.jsx) ────────────────────────────────────────────

function StatCard({ label, value, accent }) {
  return (
    <div style={{
      background: 'rgba(11,9,8,0.90)', border: `1px solid ${V.border}`,
      borderRadius: 7, padding: '10px 16px', backdropFilter: 'blur(6px)',
      minWidth: 100,
    }}>
      <div style={{
        fontSize: 20, fontWeight: 700, color: accent || V.amber,
        lineHeight: 1.15, fontVariantNumeric: 'tabular-nums', fontFamily: V.mono,
      }}>{value}</div>
      <div style={{
        fontSize: 9, color: V.textMuted, marginTop: 3,
        textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: V.mono,
      }}>{label}</div>
    </div>
  );
}
window.StatCard = StatCard;

// ── Metric info modal ─────────────────────────────────────────────────────────

function MetricModal({ def, value, onClose }) {
  const accent = def.accent || V.amber;
  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#0d0b09', border: `1px solid ${accent}55`,
          borderRadius: 10, padding: '22px 26px',
          minWidth: 320, maxWidth: 440,
          boxShadow: `0 8px 40px rgba(0,0,0,0.9), 0 0 0 1px ${accent}22`,
          fontSize: 12, color: V.textSecond, lineHeight: 1.55,
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
          <div>
            <div style={{
              fontSize: 22, fontWeight: 700, color: accent,
              fontVariantNumeric: 'tabular-nums', marginBottom: 2, fontFamily: V.mono,
            }}>{value}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: V.textPrimary }}>{def.label}</div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: V.textMuted, fontSize: 18, lineHeight: 1, padding: '0 2px', marginLeft: 16,
          }}>✕</button>
        </div>

        {/* Description */}
        <p style={{ marginBottom: 12, color: V.textSecond }}>{def.description}</p>

        {/* Formula */}
        {def.formula && (
          <>
            <div style={{
              fontSize: 9, color: V.textMuted, textTransform: 'uppercase',
              letterSpacing: '0.08em', marginBottom: 4, fontFamily: V.mono,
            }}>Formula</div>
            <div style={{
              background: 'rgba(69,26,3,0.5)', border: `1px solid ${V.borderWarm}`,
              borderRadius: 5, padding: '7px 11px', fontFamily: V.mono, fontSize: 11,
              color: V.amberBright, marginBottom: 12, wordBreak: 'break-all',
            }}>
              {def.formula}
            </div>
          </>
        )}

        {/* Research note */}
        {def.research && (
          <div style={{
            fontSize: 10, color: V.textFaint, fontStyle: 'italic',
            borderTop: `1px solid ${V.border}`, paddingTop: 8, fontFamily: V.mono,
          }}>
            {def.research}
          </div>
        )}
      </div>
    </div>
  );
}

// small helper — convert hex to rgb triplet for rgba()
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `${r},${g},${b}`;
}

// Desaturated tint of an accent — blends it toward a neutral grey so a value
// reads as restrained data (hue-linked to its dot) rather than a vivid label.
function mutedAccent(hex, t = 0.55) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  const GREY = 0xd4;  // zinc-300-ish neutral target
  const mix = (c) => Math.round(c * (1 - t) + GREY * t);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

// ── Analytics panel (floating, top-left) ──────────────────────────────────────
// Replaces the old scrolling stat-pill strip at the bottom. Shows a few primary
// metrics always; the rest expand inline (no horizontal scroll).

const PRIMARY_METRICS   = ['social_engagement_pct', 'avg_inter_human_distance',
                           'peak_occupancy', 'personal_space_pct'];
const SECONDARY_METRICS = ['scene_utilization_pct', 'gaze_convergence_events',
                           'avg_speed_mps', 'approach_events',
                           'group_count', 'event_count'];

function metricValue(key, summary) {
  if (!summary) return '—';
  switch (key) {
    case 'social_engagement_pct':   return `${summary.social_engagement_pct ?? '—'}%`;
    case 'avg_inter_human_distance':return `${summary.avg_inter_human_distance ?? '—'} m`;
    case 'scene_utilization_pct':   return `${summary.scene_utilization_pct ?? '—'}%`;
    case 'gaze_convergence_events': return `${summary.gaze_convergence_events ?? '—'}`;
    case 'personal_space_pct':      return `${summary.personal_space_pct ?? '—'}%`;
    case 'avg_speed_mps':           return summary.avg_speed_mps != null ? `${summary.avg_speed_mps} m/s` : '—';
    case 'approach_events':         return `${summary.approach_events ?? '—'}`;
    case 'peak_occupancy':          return `${summary.peak_occupancy ?? '—'}`;
    case 'group_count':             return `${summary.group_count ?? '—'}`;
    case 'event_count':             return `${summary.event_count ?? '—'}`;
    default:                        return '—';
  }
}

function MetricRow({ metricKey, value, onOpenModal }) {
  const [hover, setHover] = useState(false);
  const def    = METRIC_DEFS[metricKey] || {};
  const accent = def.accent || V.amber;
  return (
    <div
      onClick={() => def.description && onOpenModal && onOpenModal(metricKey, value)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '5px 6px',
        borderRadius: 5, cursor: def.description ? 'pointer' : 'default',
        background: hover ? `rgba(${hexToRgb(accent)},0.10)` : 'transparent',
        transition: 'background .12s',
      }}
    >
      <span style={{
        width: 7, height: 7, borderRadius: '50%', background: accent,
        flexShrink: 0, boxShadow: `0 0 6px ${accent}99`,
      }} />
      <span style={{
        flex: 1, fontSize: 10, color: hover ? V.textPrimary : V.textSecond,
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
      }}>{def.label || metricKey}</span>
      <span style={{
        fontSize: 12, fontWeight: 700, color: mutedAccent(accent), fontFamily: V.mono,
        fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap',
      }}>{value}</span>
    </div>
  );
}

function MetricsPanel({ summary, onOpenModal }) {
  const [open, setOpen]         = useState(true);
  const [expanded, setExpanded] = useState(false);
  if (!summary) return null;
  return (
    <div style={{
      position: 'absolute', top: 12, left: 12, zIndex: 10, width: 210,
      background: 'rgba(13,11,9,0.95)', border: `1px solid ${V.border}`,
      borderRadius: 8, backdropFilter: 'blur(8px)',
    }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'none', border: 'none',
        padding: '7px 12px', cursor: 'pointer', display: 'flex',
        justifyContent: 'space-between', fontSize: 9, color: V.textMuted,
        letterSpacing: '0.12em', textTransform: 'uppercase', fontFamily: V.mono,
      }}>
        Analytics <span>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{ padding: '0 8px 8px' }}>
          {PRIMARY_METRICS.map(k => (
            <MetricRow key={k} metricKey={k}
              value={metricValue(k, summary)} onOpenModal={onOpenModal} />
          ))}
          {expanded && SECONDARY_METRICS.map(k => (
            <MetricRow key={k} metricKey={k}
              value={metricValue(k, summary)} onOpenModal={onOpenModal} />
          ))}
          <button onClick={() => setExpanded(e => !e)} style={{
            width: '100%', marginTop: 4, padding: '5px 0', cursor: 'pointer',
            background: 'transparent', border: `1px solid ${V.border}`,
            borderRadius: 5, color: V.textMuted, fontSize: 9, fontFamily: V.mono,
            letterSpacing: '0.08em', textTransform: 'uppercase',
            transition: 'border-color .12s, color .12s',
          }}>
            {expanded ? '▴  Less' : `▾  ${SECONDARY_METRICS.length} more`}
          </button>
        </div>
      )}
    </div>
  );
}

// ── Layer / camera panel ──────────────────────────────────────────────────────

const LAYER_DEFS = [
  { key: 'cloud',        label: 'Scene Cloud'   },
  { key: 'skeleton',     label: 'Skeleton'      },
  { key: 'mesh',         label: 'SMPL-X Mesh'   },
  { key: 'gaze',         label: 'Gaze Arrows'   },
  { key: 'trails',       label: 'Trails'        },
  { key: 'interactions', label: 'Interactions'  },
  { key: 'heatmap',      label: 'Floor Heatmap' },
  { key: 'grid',         label: 'Grid'          },
];

function LayerPanel({ layers, onToggle, pointSize, onPointSize, camMode, onCam }) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{
      position: 'absolute', top: 12, right: 12, zIndex: 10,
      background: 'rgba(13,11,9,0.95)', border: `1px solid ${V.border}`,
      borderRadius: 8, backdropFilter: 'blur(8px)', minWidth: 170,
    }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'none', border: 'none',
        padding: '7px 12px', cursor: 'pointer', display: 'flex',
        justifyContent: 'space-between', fontSize: 9, color: V.textMuted,
        letterSpacing: '0.12em', textTransform: 'uppercase', fontFamily: V.mono,
      }}>
        Layers <span>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{ padding: '0 12px 12px' }}>
          {LAYER_DEFS.map(l => (
            <div key={l.key} onClick={() => onToggle(l.key)} style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '3px 0', cursor: 'pointer', userSelect: 'none',
            }}>
              <div style={{
                width: 12, height: 12, borderRadius: 2,
                border: `1px solid ${layers[l.key] ? V.amber : V.border}`,
                flexShrink: 0,
                background: layers[l.key] ? V.amber : 'transparent',
                transition: 'background .12s, border-color .12s',
              }} />
              <span style={{
                fontSize: 11, color: layers[l.key] ? V.textPrimary : V.textMuted,
              }}>
                {l.label}
              </span>
            </div>
          ))}

          <div style={{ marginTop: 10, borderTop: `1px solid ${V.border}`, paddingTop: 8 }}>
            <div style={{
              fontSize: 9, color: V.textMuted, marginBottom: 4,
              textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: V.mono,
            }}>Point Size</div>
            <input type="range" min={0.004} max={0.05} step={0.001}
              value={pointSize} onChange={e => onPointSize(+e.target.value)}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ marginTop: 10, borderTop: `1px solid ${V.border}`, paddingTop: 8 }}>
            <div style={{
              fontSize: 9, color: V.textMuted, marginBottom: 6,
              textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: V.mono,
            }}>Camera</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {CAM_BTNS.map(b => (
                <button key={b.id} onClick={() => onCam(b.id)} style={{
                  flex: 1, padding: '4px 0', fontSize: 9, cursor: 'pointer', borderRadius: 4,
                  fontFamily: V.mono,
                  background: camMode === b.id ? V.amberDim : 'transparent',
                  border: `1px solid ${camMode === b.id ? V.amber : V.border}`,
                  color: camMode === b.id ? V.amberBright : V.textMuted,
                  transition: 'all .12s',
                }}>{b.label}</button>
              ))}
            </div>
          </div>

          {/* Proxemics zone legend */}
          <div style={{ marginTop: 10, borderTop: `1px solid ${V.border}`, paddingTop: 8 }}>
            <div style={{
              fontSize: 9, color: V.textMuted, marginBottom: 5,
              textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: V.mono,
            }}>Proximity Zones</div>
            {[
              { zone: 'intimate', label: 'Intimate  < 0.45 m' },
              { zone: 'personal', label: 'Personal  < 1.2 m'  },
              { zone: 'social',   label: 'Social    < 3.7 m'  },
            ].map(({ zone, label }) => (
              <div key={zone} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: ZONE_HEX[zone] }} />
                <span style={{ fontSize: 9, color: V.textMuted, fontFamily: V.mono }}>{label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Playback bar ──────────────────────────────────────────────────────────────

const PB = {
  background: 'transparent', border: `1px solid ${V.border}`,
  borderRadius: 4, color: V.textSecond, fontSize: 13,
  padding: '3px 8px', cursor: 'pointer', lineHeight: 1,
};

function PlaybackBar({ frame, total, playing, speed, loop,
                       onPlay, onStep, onSpeed, onLoop, onShot, fpsRef }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 8 }}>
      <button onClick={() => onStep(-1)} style={PB}>‹</button>
      <button onClick={onPlay} style={{
        ...PB, color: V.amberBright, borderColor: V.amber, padding: '3px 12px',
      }}>{playing ? '⏸' : '▶'}</button>
      <button onClick={() => onStep(1)} style={PB}>›</button>

      <div style={{ display: 'flex', gap: 3, marginLeft: 6 }}>
        {SPEED_STEPS.map(s => (
          <button key={s} onClick={() => onSpeed(s)} style={{
            ...PB, fontSize: 9, padding: '2px 5px', fontFamily: V.mono,
            background:  speed === s ? V.amberDim : 'transparent',
            borderColor: speed === s ? V.amber : V.border,
            color:       speed === s ? V.amberBright : V.textMuted,
          }}>{s}×</button>
        ))}
      </div>

      <button onClick={onLoop} title="Loop" style={{
        ...PB, marginLeft: 2, fontSize: 12,
        background:  loop ? V.amberDim : 'transparent',
        borderColor: loop ? V.amber : V.border,
        color:       loop ? V.amberBright : V.textMuted,
      }}>↻</button>

      <span style={{ flex: 1 }} />
      <span ref={fpsRef} style={{
        fontSize: 10, color: V.green, fontVariantNumeric: 'tabular-nums',
        minWidth: 38, textAlign: 'right', fontFamily: V.mono,
      }}>-- fps</span>
      <span style={{
        fontSize: 10, color: V.textMuted, fontVariantNumeric: 'tabular-nums',
        marginLeft: 10, fontFamily: V.mono,
      }}>Frame {frame + 1} / {total}</span>
      <button onClick={onShot} title="Screenshot" style={{ ...PB, fontSize: 11, marginLeft: 4 }}>📷</button>
    </div>
  );
}

// ── Interaction timeline strip ─────────────────────────────────────────────────

function InteractionTimeline({ colors, frame, total, onSeek }) {
  const cvRef = useRef(null);

  useEffect(() => {
    const cv = cvRef.current;
    if (!cv || !colors.length) return;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const w = W / colors.length;
    colors.forEach((c, i) => {
      ctx.fillStyle =
        c === 'intimate'  ? ZONE_HEX.intimate :
        c === 'personal'  ? ZONE_HEX.personal  :
        c === 'social'    ? ZONE_HEX.social    :
        c === 'contact'   ? ZONE_HEX.intimate  :
        c === 'proximity' ? ZONE_HEX.personal  : '#18100a';
      ctx.fillRect(i * w, 0, w + 0.5, H);
    });
  }, [colors]);

  const pct = total > 1 ? frame / (total - 1) : 0;

  const handleClick = e => {
    if (!total || !cvRef.current) return;
    const r = cvRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    onSeek(Math.round(x * (total - 1)));
  };

  return (
    <div style={{ position: 'relative', height: 10, marginBottom: 3, cursor: 'crosshair' }}
      onClick={handleClick}>
      <canvas ref={cvRef} width={1200} height={10}
        style={{ width: '100%', height: '100%', borderRadius: 2, display: 'block' }} />
      <div style={{
        position: 'absolute', top: 0, bottom: 0, left: `${pct * 100}%`,
        width: 2, background: V.amberBright, opacity: 0.85,
        transform: 'translateX(-50%)', pointerEvents: 'none', borderRadius: 1,
      }} />
    </div>
  );
}

// ── Action display ────────────────────────────────────────────────────────────

function ActionDisplay({ frameData, personIds, frame, onSeek }) {
  const [expanded, setExpanded] = useState(false);
  const canvasesRef = useRef({});
  const total = frameData.length;

  const currentHumans = (frameData[frame] || {}).humans || [];

  useEffect(() => {
    if (!expanded || !total) return;
    personIds.forEach(pid => {
      const cv = canvasesRef.current[pid];
      if (!cv) return;
      const ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height;
      ctx.clearRect(0, 0, W, H);
      const w = W / total;
      frameData.forEach((fd, i) => {
        const h = (fd.humans || []).find(hu => hu.id === pid);
        ctx.fillStyle = h?.action ? (ACTION_COLORS[h.action] || '#18100a') : '#18100a';
        ctx.fillRect(i * w, 0, w + 0.5, H);
      });
    });
  }, [frameData, personIds, expanded]);

  if (!personIds.length || !total) return null;

  const pct = total > 1 ? frame / (total - 1) : 0;
  const handleSeekClick = e => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    onSeek(Math.round(x * (total - 1)));
  };

  return (
    <div style={{ marginBottom: 4 }}>
      {/* Row 1: per-subject action badge chips + expand toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
        {personIds.map(pid => {
          const h    = currentHumans.find(hu => hu.id === pid);
          const act  = h?.action || '—';
          const col  = ACTION_COLORS[act] || V.textMuted;
          const pCol = PALETTE_HEX[pid % PALETTE_HEX.length];
          return (
            <div key={pid} style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '2px 8px 2px 5px', borderRadius: 999,
              background: col + '18', border: `1px solid ${col}44`,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: pCol, flexShrink: 0,
              }} />
              <span style={{ fontSize: 9, color: pCol, fontWeight: 600, fontFamily: V.mono }}>
                S-{pid.toString().padStart(2, '0')}
              </span>
              <span style={{ fontSize: 9, color: col, fontFamily: V.mono }}>{act}</span>
            </div>
          );
        })}

        <button
          onClick={() => setExpanded(x => !x)}
          style={{
            marginLeft: 'auto', background: 'none',
            border: `1px solid ${V.border}`, borderRadius: 4,
            color: V.textMuted, fontSize: 9, cursor: 'pointer',
            padding: '2px 7px', letterSpacing: '0.06em',
            textTransform: 'uppercase', fontFamily: V.mono,
            transition: 'border-color .12s, color .12s',
          }}
          title={expanded ? 'Hide full action timeline' : 'Show full action timeline'}
        >
          {expanded ? '▲ hide' : '▼ expand'}
        </button>
      </div>

      {/* Row 2 (expanded): full per-subject timeline strips */}
      {expanded && (
        <div style={{ marginTop: 5, maxHeight: 110, overflowY: 'auto', paddingRight: 2 }}>
          {personIds.map(pid => (
            <div key={pid} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
              <span style={{
                fontSize: 8, color: PALETTE_HEX[pid % PALETTE_HEX.length],
                width: 28, flexShrink: 0, textAlign: 'right', fontFamily: V.mono,
              }}>S-{pid.toString().padStart(2,'0')}</span>
              <div
                style={{ flex: 1, position: 'relative', height: 9, cursor: 'crosshair' }}
                onClick={handleSeekClick}
              >
                <canvas
                  ref={el => { canvasesRef.current[pid] = el; }}
                  width={1200} height={9}
                  style={{ width: '100%', height: '100%', borderRadius: 2, display: 'block' }}
                />
                <div style={{
                  position: 'absolute', top: 0, bottom: 0, left: `${pct * 100}%`,
                  width: 2, background: V.amberBright, opacity: 0.65,
                  transform: 'translateX(-50%)', pointerEvents: 'none',
                }} />
              </div>
            </div>
          ))}

          {/* Legend */}
          <div style={{ display: 'flex', gap: 7, marginTop: 3, flexWrap: 'wrap', paddingLeft: 32 }}>
            {Object.entries(ACTION_COLORS).map(([action, color]) => (
              <div key={action} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ width: 6, height: 6, borderRadius: 1, background: color }} />
                <span style={{ fontSize: 7.5, color: V.textMuted, fontFamily: V.mono }}>{action}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Viewer ────────────────────────────────────────────────────────────────────

function Viewer({ plyUrl, jsonUrl }) {
  const mountRef        = useRef(null);
  const threeRef        = useRef({});
  const overlayRef      = useRef(null);
  const frameDataRef    = useRef([]);
  const facesRef        = useRef(null);
  const trailsRef       = useRef({});
  const fpsDisplayRef   = useRef(null);
  const playTimerRef    = useRef(null);
  const totalRef        = useRef(0);
  const loopRef         = useRef(true);
  const frameRef        = useRef(0);
  const labelsCanvasRef = useRef(null);

  const [frame,          setFrame]          = useState(0);
  const [total,          setTotal]          = useState(0);
  const [summary,        setSummary]        = useState(null);
  const [plyReady,       setPlyReady]       = useState(false);
  const [loadError,      setLoadError]      = useState(null);
  const [playing,        setPlaying]        = useState(false);
  const [playSpeed,      setPlaySpeed]      = useState(1);
  const [loop,           setLoop]           = useState(true);
  const [layers,         setLayers]         = useState(DEFAULT_LAYERS);
  const [pointSize,      setPointSize]      = useState(0.018);
  const [camMode,        setCamMode]        = useState('persp');
  const [timelineColors, setTimelineColors] = useState([]);
  const [heatmapMeta,    setHeatmapMeta]    = useState(null);
  const [personIds,      setPersonIds]      = useState([]);
  const [frameData,      setFrameData]      = useState([]);
  const [modalInfo,      setModalInfo]      = useState(null);

  useEffect(() => { totalRef.current = total; }, [total]);
  useEffect(() => { loopRef.current  = loop;  }, [loop]);
  useEffect(() => { frameRef.current = frame; }, [frame]);

  // ── Three.js init + PLY ───────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth || 800, H = mount.clientHeight || 600;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);

    if (labelsCanvasRef.current) {
      labelsCanvasRef.current.width  = W;
      labelsCanvasRef.current.height = H;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);   // warm dark
    scene.add(new THREE.AmbientLight(0xfff8f0, 0.6));  // warm ambient
    const dLight = new THREE.DirectionalLight(0xfff4e0, 0.9);
    dLight.position.set(5, 10, 5);
    scene.add(dLight);

    const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 200);
    camera.position.set(0, 2, -6);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance   = 0.3;
    controls.maxDistance   = 80;
    controls.update();

    // Warm dark amber grid (not cold gray)
    const grid = new THREE.GridHelper(40, 80, 0x1c0e00, 0x130900);
    scene.add(grid);

    threeRef.current = { renderer, scene, camera, controls, grid };

    let fCount = 0, fLast = performance.now();
    let raf;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);

      const lcv = labelsCanvasRef.current;
      if (lcv) {
        const ctx = lcv.getContext('2d');
        ctx.clearRect(0, 0, lcv.width, lcv.height);
        const fd = frameDataRef.current[frameRef.current];
        if (fd) {
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          (fd.humans || []).forEach(h => {
            if (!h.action) return;
            const headVec = (h.joints && h.joints.length > 15)
              ? fy(h.joints[15])
              : h.world_pos ? fy(h.world_pos) : null;
            if (!headVec) return;
            const v3 = headVec.project(camera);
            if (v3.z > 1.0) return;
            const sx = (v3.x *  0.5 + 0.5) * lcv.width;
            const sy = (v3.y * -0.5 + 0.5) * lcv.height - 24;
            const color = ACTION_COLORS[h.action] || V.textSecond;
            const pid = h.id;
            const label = `S-${pid.toString().padStart(2,'0')} · ${h.action}`;
            ctx.font = `500 10px ${V.mono}`;
            const tw = ctx.measureText(label).width;
            // pill background
            ctx.fillStyle = 'rgba(9,9,11,0.72)';
            roundRect(ctx, sx - tw/2 - 7, sy - 9, tw + 14, 18, 4);
            ctx.fill();
            // amber left border marker
            ctx.fillStyle = color;
            roundRect(ctx, sx - tw/2 - 7, sy - 9, 3, 18, 2);
            ctx.fill();
            // text
            ctx.fillStyle = color;
            ctx.fillText(label, sx + 1, sy + 1);
          });
        }
      }

      fCount++;
      const now = performance.now();
      if (now - fLast >= 1000) {
        const fps = Math.round(fCount * 1000 / (now - fLast));
        if (fpsDisplayRef.current) fpsDisplayRef.current.textContent = `${fps} fps`;
        fCount = 0; fLast = now;
      }
    };
    animate();

    const onResize = () => {
      const W = mount.clientWidth, H = mount.clientHeight;
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
      if (labelsCanvasRef.current) {
        labelsCanvasRef.current.width  = W;
        labelsCanvasRef.current.height = H;
      }
    };
    window.addEventListener('resize', onResize);

    // ── Load PLY ─────────────────────────────────────────────────────────
    const loader = new THREE.PLYLoader();
    loader.load(plyUrl, geo => {
      geo.computeBoundingBox();
      const hasColor = geo.hasAttribute('color');
      const mat = new THREE.PointsMaterial({
        size: 0.018, sizeAttenuation: true, vertexColors: hasColor,
      });
      if (!hasColor) mat.color.set(0x6b5a40);  // warm fallback
      const pts = new THREE.Points(geo, mat);
      pts.scale.y = -1;
      scene.add(pts);
      threeRef.current.cloudPts = pts;
      threeRef.current.cloudMat = mat;

      const box = geo.boundingBox;
      const center = new THREE.Vector3(), size = new THREE.Vector3();
      box.getCenter(center);
      center.y = -center.y;
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      threeRef.current.maxDim = maxDim;

      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(0, maxDim * 0.3, -maxDim * 1.2));
      camera.up.set(0, 1, 0);
      controls.update();

      const floorY = -box.max.y;
      grid.position.y = floorY;
      threeRef.current.floorY = floorY;

      setPlyReady(true);
    }, undefined, err => setLoadError('Could not load scene.ply: ' + (err.message || String(err))));

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [plyUrl]);

  // ── JSON load ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const ctrl = new AbortController();
    const sig  = ctrl.signal;

    setFrame(0); setTotal(0); setSummary(null);
    setTimelineColors([]); setLoadError(null);
    setHeatmapMeta(null); setPersonIds([]); setFrameData([]);
    facesRef.current  = null;
    trailsRef.current = {};

    const base    = jsonUrl.replace('enriched_data.json', '');
    const dashUrl = base + 'dashboard_data.json';
    const vidxUrl = base + 'verts_index.json';
    const vbinUrl = base + 'verts.bin';

    Promise.all([
      fetch(jsonUrl, { signal: sig }).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      }),
      fetch(dashUrl, { signal: sig }).then(r => r.ok ? r.json() : null).catch(() => null),
      // Verts side-car: index (small JSON) + raw Float32 binary. Loaded via
      // arrayBuffer so it dodges Chrome's ~512 MB max JS-string limit that makes
      // a full-resolution dashboard_data.json un-parseable on long clips.
      fetch(vidxUrl, { signal: sig }).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(vbinUrl, { signal: sig }).then(r => r.ok ? r.arrayBuffer() : null).catch(() => null),
    ])
    .then(([enriched, dash, vidx, vbin]) => {
      if (sig.aborted) return;
      const frames = enriched.frames || [];

      if (dash && dash.frames) {
        dash.frames.forEach((df, i) => {
          if (!frames[i]) return;
          (df.humans || []).forEach((dh, j) => {
            const eh = (frames[i].humans || [])[j];
            if (!eh) return;
            if (dh.verts)  eh.verts  = dh.verts;
            if (dh.joints) eh.joints = dh.joints;
          });
        });
      }

      // Attach a flat Float32 vertex buffer (h.vbuf) per human. Prefer the binary
      // side-car (block index → offset into the big buffer); else flatten any
      // array-of-[x,y,z] verts merged from a small dashboard_data.json.
      if (vidx && vbin) {
        const f32 = new Float32Array(vbin);
        const stride = vidx.stride;
        (vidx.frames || []).forEach((blocks, i) => {
          const humans = frames[i] && frames[i].humans;
          if (!humans) return;
          blocks.forEach((block, j) => {
            if (!humans[j]) return;
            const start = block * stride;
            humans[j].vbuf = f32.subarray(start, start + stride);
          });
        });
      }
      frames.forEach(fd => (fd.humans || []).forEach(h => {
        if (!h.vbuf && h.verts && h.verts.length) {
          const fb = new Float32Array(h.verts.length * 3);
          for (let k = 0; k < h.verts.length; k++) {
            fb[k*3] = h.verts[k][0]; fb[k*3+1] = h.verts[k][1]; fb[k*3+2] = h.verts[k][2];
          }
          h.vbuf = fb;
        }
      }));

      frameDataRef.current = frames;
      setFrameData(frames);
      setTotal(frames.length);
      setSummary(enriched.summary || null);

      const meta  = (dash && dash.metadata) || enriched.metadata || {};
      const faces = meta.smpl_faces;
      if (faces && faces.length > 0) facesRef.current = new Uint32Array(faces.flat());

      const enrichedMeta = enriched.metadata || {};
      if (enrichedMeta.heatmap) {
        setHeatmapMeta(enrichedMeta.heatmap);
      } else if (frames.length > 0) {
        const allPos = frames.flatMap(fd => (fd.humans || []).map(h => h.world_pos));
        if (allPos.length > 0) {
          const margin = 0.7;
          const xs = allPos.map(p => p[0]);
          const zs = allPos.map(p => p[2]);
          setHeatmapMeta({
            x_min: Math.min(...xs) - margin,
            x_max: Math.max(...xs) + margin,
            z_min: Math.min(...zs) - margin,
            z_max: Math.max(...zs) + margin,
          });
        }
      }

      const ids = new Set();
      frames.forEach(fd => (fd.humans || []).forEach(h => ids.add(h.id)));
      setPersonIds([...ids].sort((a, b) => a - b));

      const colors = frames.map(fd => {
        if (!fd.interactions || !fd.interactions.length) return 'none';
        const zones = fd.interactions.map(ix =>
          ix.zone || (ix.distance < 0.45 ? 'intimate' :
                      ix.distance < 1.2  ? 'personal' :
                      ix.distance < 3.7  ? 'social'   : 'none')
        );
        if (zones.includes('intimate')) return 'intimate';
        if (zones.includes('personal')) return 'personal';
        if (zones.includes('social'))   return 'social';
        return 'none';
      });
      setTimelineColors(colors);
    })
    .catch(err => {
      if (err.name !== 'AbortError') setLoadError('Could not load data: ' + err.message);
    });

    return () => ctrl.abort();
  }, [jsonUrl]);

  // ── Heatmap floor plane ───────────────────────────────────────────────────
  useEffect(() => {
    const { scene } = threeRef.current;
    if (!scene || !heatmapMeta || !plyReady) return;

    if (threeRef.current.heatmapPlane) {
      scene.remove(threeRef.current.heatmapPlane);
      threeRef.current.heatmapPlane.geometry.dispose();
      const hm = threeRef.current.heatmapPlane.material;
      if (hm.map) hm.map.dispose();
      hm.dispose();
      threeRef.current.heatmapPlane = null;
    }

    const { x_min, x_max, z_min, z_max } = heatmapMeta;
    const w  = x_max - x_min;
    const d  = z_max - z_min;
    const cx = (x_min + x_max) / 2;
    const cz = (z_min + z_max) / 2;
    const floorY = threeRef.current.floorY ?? 0;

    const mat = new THREE.MeshBasicMaterial({
      color: 0x1c0a00,
      transparent: true, opacity: 0.35,
      depthWrite: false, side: THREE.DoubleSide,
    });

    const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
    plane.rotation.x = Math.PI / 2;
    plane.position.set(cx, floorY + 0.01, cz);
    plane.visible = layers.heatmap;
    scene.add(plane);
    threeRef.current.heatmapPlane = plane;

    const hmUrl = jsonUrl.replace('enriched_data.json', 'heatmap.png');
    new THREE.TextureLoader().load(
      hmUrl,
      texture => {
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        mat.map     = texture;
        mat.color.set(0xffffff);
        mat.opacity = 0.60;
        mat.needsUpdate = true;
      },
      undefined,
      () => {}
    );
  }, [heatmapMeta, plyReady, jsonUrl]);

  // ── Playback timer ────────────────────────────────────────────────────────
  useEffect(() => {
    if (playTimerRef.current) { clearInterval(playTimerRef.current); playTimerRef.current = null; }
    if (!playing || total === 0) return;
    const ms = Math.max(16, Math.round(1000 / (30 * playSpeed)));
    playTimerRef.current = setInterval(() => {
      setFrame(f => {
        const next = f + 1;
        if (next >= totalRef.current) {
          if (loopRef.current) return 0;
          setPlaying(false);
          return f;
        }
        return next;
      });
    }, ms);
    return () => { clearInterval(playTimerRef.current); playTimerRef.current = null; };
  }, [playing, playSpeed, total]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === ' ')          { e.preventDefault(); setPlaying(p => !p); }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); setFrame(f => Math.max(0, f - 1)); }
      if (e.key === 'ArrowRight') { e.preventDefault(); setFrame(f => Math.min(totalRef.current - 1, f + 1)); }
      if (e.key === '1') setPlaySpeed(0.25);
      if (e.key === '2') setPlaySpeed(0.5);
      if (e.key === '3') setPlaySpeed(1);
      if (e.key === '4') setPlaySpeed(2);
      if (e.key === '5') setPlaySpeed(4);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // ── Direct visibility toggles ─────────────────────────────────────────────
  useEffect(() => {
    if (threeRef.current.cloudPts) threeRef.current.cloudPts.visible = layers.cloud;
  }, [layers.cloud]);

  useEffect(() => {
    if (threeRef.current.grid) threeRef.current.grid.visible = layers.grid;
  }, [layers.grid]);

  useEffect(() => {
    if (threeRef.current.heatmapPlane) threeRef.current.heatmapPlane.visible = layers.heatmap;
  }, [layers.heatmap]);

  useEffect(() => {
    if (threeRef.current.cloudMat) threeRef.current.cloudMat.size = pointSize;
  }, [pointSize]);

  // ── Per-frame overlay ─────────────────────────────────────────────────────
  useEffect(() => {
    const { scene } = threeRef.current;
    if (!scene) return;

    if (overlayRef.current) {
      scene.remove(overlayRef.current);
      overlayRef.current.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
      });
    }

    const group = new THREE.Group();
    overlayRef.current = group;
    scene.add(group);

    const fd = frameDataRef.current[frame];
    if (!fd) return;

    const byId = {};

    (fd.humans || []).forEach(h => {
      byId[h.id] = h;
      const color = PALETTE[h.id % PALETTE.length];
      const pos   = fy(h.world_pos);

      // SMPL-X surface mesh — built from the flat Float32 buffer (h.vbuf), which
      // comes from either verts.bin or a small dashboard_data.json (see loader).
      const vb = h.vbuf;
      if (layers.mesh) {
        if (vb && vb.length > 300 && facesRef.current) {
          const buf = new Float32Array(vb.length);   // copy + Y-flip (OpenCV→Three)
          for (let i = 0; i < vb.length; i += 3) {
            buf[i] = vb[i]; buf[i+1] = -vb[i+1]; buf[i+2] = vb[i+2];
          }
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
          geo.setIndex(new THREE.BufferAttribute(facesRef.current.slice(), 1));
          geo.computeVertexNormals();
          group.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
            color, opacity: 0.82, transparent: true,
            side: THREE.DoubleSide, shininess: 30,
          })));
        } else if (vb && vb.length > 0) {
          const buf = new Float32Array(vb.length);
          for (let i = 0; i < vb.length; i += 3) {
            buf[i] = vb[i]; buf[i+1] = -vb[i+1]; buf[i+2] = vb[i+2];
          }
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
          group.add(new THREE.Points(geo, new THREE.PointsMaterial({
            color, size: 0.025, sizeAttenuation: true,
          })));
        } else {
          const m = new THREE.Mesh(
            new THREE.SphereGeometry(0.07, 10, 8),
            new THREE.MeshBasicMaterial({ color })
          );
          m.position.copy(pos);
          group.add(m);
        }
      }

      // Skeleton
      if (layers.skeleton) {
        if (h.joints && h.joints.length >= 22) {
          const jPts   = h.joints.map(j => fy(j));
          const boneMat = new THREE.MeshPhongMaterial({
            color, opacity: 0.88, transparent: true, shininess: 25,
          });
          jPts.slice(0, 22).forEach((jp, idx) => {
            const s = new THREE.Mesh(
              new THREE.SphereGeometry(JOINT_RADII[idx] || 0.022, 8, 6), boneMat);
            s.position.copy(jp);
            group.add(s);
          });
          SMPLX_BONES.forEach(([a, b], bIdx) => {
            if (!jPts[a] || !jPts[b]) return;
            const p1 = jPts[a], p2 = jPts[b];
            const dir = new THREE.Vector3().subVectors(p2, p1);
            const len = dir.length();
            if (len < 0.005) return;
            const cyl = new THREE.Mesh(
              new THREE.CylinderGeometry(BONE_RADII[bIdx]||0.012, BONE_RADII[bIdx]||0.012, len, 6),
              boneMat
            );
            cyl.position.copy(p1).add(p2).multiplyScalar(0.5);
            cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dir.clone().normalize());
            group.add(cyl);
          });
        } else if (h.head_world) {
          const headPos = fy(h.head_world);
          const m = new THREE.Mesh(
            new THREE.SphereGeometry(0.045, 8, 6),
            new THREE.MeshBasicMaterial({ color, opacity: 0.7, transparent: true })
          );
          m.position.copy(headPos);
          group.add(m);
          const stemGeo = new THREE.BufferGeometry().setFromPoints([pos, headPos]);
          group.add(new THREE.Line(stemGeo,
            new THREE.LineBasicMaterial({ color, opacity: 0.35, transparent: true })));
        }
      }

      // Gaze arrow — red thread aesthetic for crime-board feel
      if (layers.gaze && h.gaze_vec) {
        const arrowOrigin = (h.joints && h.joints.length > 15)
          ? fy(h.joints[15])
          : fy(h.world_pos);
        const dir = fy(h.gaze_vec);
        if (dir.lengthSq() > 0.001) {
          dir.normalize();
          group.add(new THREE.ArrowHelper(dir, arrowOrigin, 1.0, color, 0.22, 0.10));
        }
      }

      // Trajectory trail
      if (layers.trails) {
        const trail = trailsRef.current[h.id] || [];
        trail.push(h.world_pos);
        if (trail.length > TRAIL_LEN) trail.shift();
        trailsRef.current[h.id] = trail;
        if (trail.length >= 2) {
          const pts = trail.map(p => fy(p));
          for (let t = 1; t < pts.length; t++) {
            const opacity = 1 - (pts.length - t) / pts.length;
            const geo = new THREE.BufferGeometry().setFromPoints([pts[t-1], pts[t]]);
            group.add(new THREE.Line(geo,
              new THREE.LineBasicMaterial({ color, opacity, transparent: true })));
          }
        }
      }
    });

    // Interaction edges — zone-coloured (already red/orange/amber = crime board)
    if (layers.interactions) {
      (fd.interactions || []).forEach(ix => {
        const h1 = byId[ix.source], h2 = byId[ix.target];
        if (!h1 || !h2) return;
        const zone = ix.zone || (ix.distance < 0.45 ? 'intimate' :
                                  ix.distance < 1.2  ? 'personal' :
                                  ix.distance < 3.7  ? 'social'   : 'public');
        if (zone === 'public') return;
        const zoneColor   = ZONE_COLORS[zone] || ZONE_COLORS.social;
        const zoneOpacity = zone === 'intimate' ? 0.95 :
                            zone === 'personal' ? 0.72 : 0.45;
        const geo = new THREE.BufferGeometry().setFromPoints([
          fy(h1.world_pos), fy(h2.world_pos),
        ]);
        group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
          color: zoneColor, transparent: true, opacity: zoneOpacity,
        })));
        if (ix.mutual_gaze) {
          const mid = new THREE.Vector3()
            .addVectors(fy(h1.world_pos), fy(h2.world_pos))
            .multiplyScalar(0.5);
          const mg = new THREE.Mesh(
            new THREE.SphereGeometry(0.04, 6, 4),
            new THREE.MeshBasicMaterial({ color: 0xfcd34d, opacity: 0.7, transparent: true })
          );
          mg.position.copy(mid);
          group.add(mg);
        }
      });
    }

    // Follow-camera
    if (camMode === 'follow' && fd.humans && fd.humans.length > 0) {
      const { controls, camera } = threeRef.current;
      if (controls && camera) {
        const p0     = fy(fd.humans[0].world_pos);
        const offset = camera.position.clone().sub(controls.target);
        controls.target.copy(p0);
        camera.position.copy(p0).add(offset);
        controls.update();
      }
    }

  }, [frame, total, layers, camMode]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const toggleLayer = useCallback(key =>
    setLayers(l => ({ ...l, [key]: !l[key] })), []);

  const handleStep = useCallback(delta =>
    setFrame(f => Math.max(0, Math.min(totalRef.current - 1, f + delta))), []);

  const handleCam = useCallback(mode => {
    setCamMode(mode);
    const { camera, controls } = threeRef.current;
    if (!camera || !controls || mode === 'follow') return;
    const t    = controls.target.clone();
    const dist = camera.position.distanceTo(t);
    if (mode === 'front') {
      camera.position.set(t.x, t.y + 0.5, t.z - Math.max(dist, 5));
    } else {
      camera.position.set(t.x - dist * 0.25, t.y + dist * 0.35, t.z - dist);
    }
    camera.up.set(0, 1, 0);
    controls.update();
  }, []);

  const handleOpenModal = useCallback((metricKey, value) => {
    setModalInfo({ metricKey, value });
  }, []);

  const handleScreenshot = useCallback(() => {
    const { renderer, scene, camera } = threeRef.current;
    if (!renderer) return;
    renderer.render(scene, camera);
    const a = document.createElement('a');
    a.href     = renderer.domElement.toDataURL('image/png');
    a.download = `bodyplot_frame_${frame + 1}.png`;
    a.click();
  }, [frame]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* 2D action-label overlay */}
      <canvas ref={labelsCanvasRef} style={{
        position: 'absolute', inset: 0, width: '100%', height: '100%',
        pointerEvents: 'none',
      }} />

      {/* Loading overlay */}
      {!plyReady && !loadError && (
        <div style={OVL}>
          <Spinner />
          <span style={{ marginTop: 10, color: V.textMuted, fontSize: 12, fontFamily: V.mono }}>
            Loading scene…
          </span>
        </div>
      )}

      {/* Error overlay */}
      {loadError && (
        <div style={{ ...OVL, color: V.red, fontSize: 12, padding: 24, textAlign: 'center', fontFamily: V.mono }}>
          {loadError}
        </div>
      )}

      {/* Metric info modal */}
      {modalInfo && (() => {
        const def = METRIC_DEFS[modalInfo.metricKey];
        if (!def) return null;
        return <MetricModal def={def} value={modalInfo.value} onClose={() => setModalInfo(null)} />;
      })()}

      {/* Analytics panel (top-left) */}
      <MetricsPanel
        summary={summary} onOpenModal={handleOpenModal}
      />

      {/* Layer + camera panel */}
      <LayerPanel
        layers={layers} onToggle={toggleLayer}
        pointSize={pointSize} onPointSize={setPointSize}
        camMode={camMode} onCam={handleCam}
      />

      {/* ── Bottom dashboard ─────────────────────────────────────────────── */}
      {total > 0 && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 3,
          background: 'linear-gradient(to top, rgba(9,9,11,0.98) 65%, rgba(9,9,11,0.80))',
          backdropFilter: 'blur(12px)',
          borderTop: `1px solid ${V.borderWarm}`,
          padding: '10px 18px 13px',
        }}>

          {/* Metrics moved to the floating Analytics panel (top-left) to free
              vertical space here and avoid a horizontal scroll strip. */}

          {/* Playback controls */}
          <PlaybackBar
            frame={frame} total={total} playing={playing}
            speed={playSpeed} loop={loop}
            onPlay={() => setPlaying(p => !p)}
            onStep={handleStep}
            onSpeed={setPlaySpeed}
            onLoop={() => setLoop(l => !l)}
            onShot={handleScreenshot}
            fpsRef={fpsDisplayRef}
          />

          {/* Subject action badges + optional full timeline */}
          {personIds.length > 0 && frameData.length > 0 && (
            <ActionDisplay
              frameData={frameData} personIds={personIds}
              frame={frame} onSeek={setFrame}
            />
          )}

          {/* Interaction timeline */}
          {timelineColors.length > 0 && (
            <InteractionTimeline
              colors={timelineColors} frame={frame} total={total}
              onSeek={setFrame}
            />
          )}

          {/* Scrubber */}
          <input
            type="range" min={0} max={total - 1} value={frame}
            onChange={e => setFrame(+e.target.value)}
            style={{ width: '100%', cursor: 'pointer', display: 'block' }}
          />
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontSize: 9, marginTop: 3, fontFamily: V.mono,
          }}>
            <span style={{ color: V.textFaint }}>0</span>
            <span style={{ color: V.textMuted }}>
              Space play/pause · ←/→ step · 1–5 speed · click strip to seek
            </span>
            <span style={{ color: V.textFaint }}>{total - 1}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const OVL = {
  position: 'absolute', inset: 0, pointerEvents: 'none',
  display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center',
};

function Spinner() {
  return (
    <div style={{
      width: 32, height: 32, borderRadius: '50%',
      border: `3px solid ${V.border}`, borderTopColor: V.amber,
      animation: 'spin 0.8s linear infinite',
    }} />
  );
}

// Canvas rounded-rect helper (CanvasRenderingContext2D.roundRect not available in all browsers)
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

(function injectCSS() {
  if (document.getElementById('sitl-spin-css')) return;
  const s = document.createElement('style');
  s.id = 'sitl-spin-css';
  s.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
  document.head.appendChild(s);
})();

window.Viewer = Viewer;
