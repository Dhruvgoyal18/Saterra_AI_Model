import type { DetectSummary } from "./types";

interface Props {
  summary: DetectSummary;
}

export default function SummaryPanel({ summary }: Props) {
  const pctOfTissue = (summary.total_artifact_area_frac_of_tissue * 100).toFixed(1);
  const tissueCoverage = ((summary.n_tiles_evaluated / summary.n_tiles_total) * 100).toFixed(0);

  return (
    <div className="summary-panel">
      <div className="stat">
        <div className="stat-value">{pctOfTissue}%</div>
        <div className="stat-label">of tissue flagged as artifact</div>
      </div>
      <div className="stat">
        <div className="stat-value">{summary.num_connected_regions.toLocaleString()}</div>
        <div className="stat-label">connected artifact regions</div>
      </div>
      <div className="stat">
        <div className="stat-value">{(summary.largest_region_area_px / 1_000_000).toFixed(2)}M px</div>
        <div className="stat-label">largest single region</div>
      </div>
      <div className="stat">
        <div className="stat-value">{tissueCoverage}%</div>
        <div className="stat-label">
          of tiles evaluated ({summary.n_tiles_evaluated.toLocaleString()} / {summary.n_tiles_total.toLocaleString()})
        </div>
      </div>
    </div>
  );
}
