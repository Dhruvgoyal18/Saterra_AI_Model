// Diverging colormap for tile confidence: near 1 = confident artifact (red),
// near 0 = confident clean (blue), near 0.5 = uncertain (pale/white). This is
// what turns the raw confidence grid into "what is and is not an artifact,
// and how sure the model is" in one glance, rather than a binary mask.

const BLUE: [number, number, number] = [33, 102, 172]; // confidence -> 0 (confident clean)
const WHITE: [number, number, number] = [247, 247, 247]; // confidence -> 0.5 (uncertain)
const RED: [number, number, number] = [178, 24, 43]; // confidence -> 1 (confident artifact)

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(c1: [number, number, number], c2: [number, number, number], t: number): [number, number, number] {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}

export function confidenceToRgb(confidence: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, confidence));
  if (clamped <= 0.5) {
    return lerpColor(BLUE, WHITE, clamped / 0.5);
  }
  return lerpColor(WHITE, RED, (clamped - 0.5) / 0.5);
}

export function confidenceToRgba(confidence: number, alpha: number): string {
  const [r, g, b] = confidenceToRgb(confidence);
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

export const UNCERTAIN_LOW = 0.4;
export const UNCERTAIN_HIGH = 0.6;

export function isUncertain(confidence: number): boolean {
  return confidence >= UNCERTAIN_LOW && confidence <= UNCERTAIN_HIGH;
}
