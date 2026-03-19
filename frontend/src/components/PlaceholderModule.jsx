export default function PlaceholderModule({ title }) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-soft">
      <p className="text-xs uppercase tracking-[0.16em] text-steel">Upcoming Module</p>
      <h2 className="mt-3 font-display text-3xl font-bold text-ink">{title}</h2>
      <p className="mt-4 max-w-2xl text-sm text-slate-600">
        This tab is scaffolded and ready. Once you say go, I will implement this module with dataset wiring,
        model training flow, visual metrics, and integration into the final decision engine.
      </p>
    </section>
  );
}
