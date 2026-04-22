/**
 * Decorative neural network animation for the landing hero.
 * Pure SVG + CSS — no external deps. Themed via design tokens.
 */
export function NeuralNetworkAnim() {
  // Layer x positions and node y positions
  const layers: { x: number; ys: number[] }[] = [
    { x: 60, ys: [80, 140, 200, 260, 320] },
    { x: 200, ys: [70, 130, 190, 250, 310, 370] },
    { x: 340, ys: [90, 160, 230, 300] },
    { x: 480, ys: [170, 230] },
  ];

  // Build edges between consecutive layers
  const edges: { x1: number; y1: number; x2: number; y2: number; key: string; delay: number }[] =
    [];
  for (let l = 0; l < layers.length - 1; l++) {
    const a = layers[l];
    const b = layers[l + 1];
    a.ys.forEach((y1, i) => {
      b.ys.forEach((y2, j) => {
        edges.push({
          x1: a.x,
          y1,
          x2: b.x,
          y2,
          key: `${l}-${i}-${j}`,
          delay: ((i + j) % 6) * 0.35 + l * 0.15,
        });
      });
    });
  }

  return (
    <div className="relative aspect-[5/4] w-full">
      {/* Glow backdrop */}
      <div
        aria-hidden
        className="absolute inset-0 -z-10 rounded-3xl opacity-60 blur-3xl"
        style={{ background: "var(--gradient-primary)" }}
      />

      <svg
        viewBox="0 0 540 440"
        className="h-full w-full"
        role="img"
        aria-label="Animated neural network detecting patterns in a chest X-ray"
      >
        <defs>
          <linearGradient id="edgeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.15" />
            <stop offset="50%" stopColor="var(--primary-glow)" stopOpacity="0.9" />
            <stop offset="100%" stopColor="var(--primary)" stopOpacity="0.15" />
          </linearGradient>
          <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--primary-glow)" />
            <stop offset="100%" stopColor="var(--primary)" />
          </radialGradient>
          <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Edges */}
        <g stroke="url(#edgeGrad)" strokeWidth="1" opacity="0.55">
          {edges.map((e) => (
            <line
              key={e.key}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              className="nn-edge"
              style={{ animationDelay: `${e.delay}s` }}
            />
          ))}
        </g>

        {/* Nodes */}
        <g filter="url(#softGlow)">
          {layers.map((layer, li) =>
            layer.ys.map((y, i) => (
              <circle
                key={`n-${li}-${i}`}
                cx={layer.x}
                cy={y}
                r={li === layers.length - 1 ? 10 : 7}
                fill="url(#nodeGrad)"
                className="nn-node"
                style={{ animationDelay: `${(li * 0.25 + i * 0.15) % 2.4}s` }}
              />
            )),
          )}
        </g>

        {/* Output labels */}
        <g
          fill="var(--primary-foreground)"
          fontSize="11"
          fontWeight="600"
          textAnchor="middle"
          fontFamily="ui-sans-serif, system-ui"
        >
          <text x="480" y="174">
            ✓
          </text>
          <text x="480" y="234">
            ✗
          </text>
        </g>

        {/* Traveling pulses on a few selected edges */}
        {edges
          .filter((_, idx) => idx % 7 === 0)
          .slice(0, 8)
          .map((e, i) => (
            <circle
              key={`p-${e.key}`}
              r="2.5"
              fill="var(--primary-glow)"
              className="nn-pulse"
              style={{ animationDelay: `${i * 0.4}s`, offsetPath: "none" }}
            >
              <animate
                attributeName="cx"
                from={e.x1}
                to={e.x2}
                dur="2.4s"
                begin={`${i * 0.4}s`}
                repeatCount="indefinite"
              />
              <animate
                attributeName="cy"
                from={e.y1}
                to={e.y2}
                dur="2.4s"
                begin={`${i * 0.4}s`}
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                dur="2.4s"
                begin={`${i * 0.4}s`}
                repeatCount="indefinite"
              />
            </circle>
          ))}
      </svg>

      <style>{`
        @keyframes nn-pulse-node {
          0%, 100% { opacity: 0.55; transform-origin: center; }
          50% { opacity: 1; }
        }
        @keyframes nn-flow {
          0%, 100% { stroke-opacity: 0.18; }
          50% { stroke-opacity: 0.85; }
        }
        .nn-node {
          animation: nn-pulse-node 2.6s ease-in-out infinite;
        }
        .nn-edge {
          animation: nn-flow 3.2s ease-in-out infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .nn-node, .nn-edge, .nn-pulse { animation: none !important; }
        }
      `}</style>
    </div>
  );
}
