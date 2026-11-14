// polymarket-faa-attribution-02 entry: "Adaptive Laplace Persistence" (Node 22, stdlib only).
//
// Forecast for each market: the median stays at the checkpoint price, and the
// spread widens like a random walk whose per-minute scale is re-estimated from
// that market's own recent history at every checkpoint. The spread uses a
// Laplace (double-exponential) shape, which fits these prices better than a
// Gaussian: most minutes barely move, but jumps happen, so the centre is
// peaked and the tails are heavier.
//
// Per-market scale, from the 1-minute moves of the case's own history:
//   - drop the market's first 120 minutes (the feed carries synchronised
//     one-minute blips and listing transients early in a market's life);
//   - clip each move to +/-0.02 so one freak jump cannot set the scale;
//   - for each window W in {120, 240, 1440} minutes, sigma_W =
//     mean(|move|) * sqrt(pi/2) over the last W moves (the mean absolute
//     deviation of a Laplace variable, converted to its scale);
//   - sigma = the mean of the available sigma_W, floored at 0.0002 and
//     capped at 0.03; a market with no usable moves falls back to the
//     baseline's 0.0025.
// Quantiles at horizon h (minutes after the checkpoint):
//   q_tau(h) = clamp(x + L_tau * sigma * 1.2 * sqrt(h), 0, 1), rounded to
//   6 decimals, where x is the checkpoint price and L_tau is the Laplace
//   quantile at level tau: ln(2*tau) below the median, -ln(2*(1-tau)) above.
// The 1.2 widens the raw Laplace scale to the width the practice data
// supports (practice loss 0.309 on p00-p04 vs the baseline's 0.393);
// sqrt(h) is the random-walk widening.
//
// Deterministic, no network, no clock: the same case always yields the same
// bytes. Reads every <id>.in in the input directory and writes <id>.out
// per case: {"forecasts":[{"id","q": 240 rows x 5 quantiles}]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The Laplace distribution's quantile at each level: ln(2*tau) / -ln(2*(1-tau)). */
const LAPLACE_Z = [-2.302585092994046, -0.6931471805599453, 0, 0.6931471805599453, 2.302585092994046];
/** Windows (minutes) over which the per-market scale is averaged. */
const SCALE_WINDOWS = [120, 240, 1440];
/** First minutes of a market's history ignored when estimating its scale. */
const BURN_IN = 120;
/** One move is clipped to this magnitude before it enters the scale. */
const MOVE_CLIP = 0.02;
/** Scale floor / cap, in price units per square-root minute. */
const SIGMA_FLOOR = 0.0002;
const SIGMA_CAP = 0.03;
/** Fallback scale when a market has no usable moves (baseline's value). */
const SIGMA_FALLBACK = 0.0025;
/** Extra widening of the Laplace scale, tuned on the practice cases. */
const SPREAD_K = 1.2;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/**
 * A market's per-minute Laplace scale from its own history. The moves after
 * the burn-in are clipped, then each window's mean absolute move is converted
 * to a Laplace scale (mean |X| = b for Laplace(0, b), and for a normal
 * variable mean |X| = sigma * sqrt(2/pi), so the same factor converts the
 * mean absolute move to the scale either way) and the windows are averaged.
 */
function estimateScale(prices: number[]): number {
  const moves: number[] = [];
  for (let i = BURN_IN; i < prices.length; i++) {
    const d = prices[i] - prices[i - 1];
    moves.push(Math.max(-MOVE_CLIP, Math.min(MOVE_CLIP, d)));
  }
  if (moves.length < 10) return SIGMA_FALLBACK;
  let sum = 0;
  let used = 0;
  for (const w of SCALE_WINDOWS) {
    const tail = moves.slice(Math.max(0, moves.length - w));
    if (tail.length < 10) continue;
    let abs = 0;
    for (const d of tail) abs += Math.abs(d);
    // Mean absolute move * sqrt(pi/2) is the Laplace scale of the moves.
    sum += (abs / tail.length) * Math.sqrt(Math.PI / 2);
    used++;
  }
  if (used === 0) return SIGMA_FALLBACK;
  const sigma = sum / used;
  return Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
}

/** The forecast of one market: horizon rows of the five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = estimateScale(m.prices);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * SPREAD_K * Math.sqrt(h);
    rows.push(LAPLACE_Z.map((z) => price6(x + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
