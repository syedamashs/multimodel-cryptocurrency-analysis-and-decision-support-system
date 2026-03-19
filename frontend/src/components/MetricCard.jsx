export default function MetricCard({ title, value, subtitle }) {
  return (
    <article className="rounded-2xl border border-slate-200/70 bg-white p-4 shadow-soft">
      <p className="text-xs uppercase tracking-[0.16em] text-steel/80">{title}</p>
      <p className="mt-2 text-2xl font-display font-bold text-ink">{value}</p>
      {subtitle ? <p className="mt-2 text-xs text-slate-500">{subtitle}</p> : null}
    </article>
  );
}
