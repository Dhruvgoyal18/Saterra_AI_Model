// Diverging colormap for tile confidence: near 1 = confident artifact (the
// chosen artifact color), near 0 = confident clean (blue), near 0.5 =
// uncertain (pale/white). This is what turns the raw confidence grid into
// "what is and is not an artifact, and how sure the model is" in one
// glance, rather than a binary mask.
//
// The "clean" anchor (blue) and "uncertain" midpoint (white) are fixed --
// only the "artifact" end of the gradient is chooseable, via the palette
// below. "blue" is deliberately excluded from the palette (backend's
// chat.py enforces the same exclusion for the same reason): allowing it as
// the artifact color too would collapse the diverging gradient into a
// single hue with no visual distinction between "confidently clean" and
// "confidently artifact".

const BLUE: [number, number, number] = [33, 102, 172]; // confidence -> 0 (confident clean), fixed
const WHITE: [number, number, number] = [247, 247, 247]; // confidence -> 0.5 (uncertain), fixed

export const OVERLAY_COLOR_PALETTE = {
  red: [178, 24, 43],
  green: [27, 120, 55],
  orange: [230, 126, 34],
  purple: [118, 42, 131],
  yellow: [202, 178, 30],
  pink: [214, 96, 172],
  teal: [26, 152, 152],
} as const satisfies Record<string, [number, number, number]>;

export type OverlayColorName = keyof typeof OVERLAY_COLOR_PALETTE;

// Matches the pre-existing hardcoded RED exactly, so leaving this unset
// anywhere reproduces today's behavior pixel-for-pixel.
export const DEFAULT_OVERLAY_COLOR: OverlayColorName = "red";

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(c1: [number, number, number], c2: [number, number, number], t: number): [number, number, number] {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}

export function confidenceToRgb(
  confidence: number,
  artifactColor: OverlayColorName = DEFAULT_OVERLAY_COLOR
): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, confidence));
  const red = OVERLAY_COLOR_PALETTE[artifactColor];
  if (clamped <= 0.5) {
    return lerpColor(BLUE, WHITE, clamped / 0.5);
  }
  return lerpColor(WHITE, red, (clamped - 0.5) / 0.5);
}

export function confidenceToRgba(
  confidence: number,
  alpha: number,
  artifactColor: OverlayColorName = DEFAULT_OVERLAY_COLOR
): string {
  const [r, g, b] = confidenceToRgb(confidence, artifactColor);
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

export const UNCERTAIN_LOW = 0.4;
export const UNCERTAIN_HIGH = 0.6;

export function isUncertain(confidence: number): boolean {
  return confidence >= UNCERTAIN_LOW && confidence <= UNCERTAIN_HIGH;
}
