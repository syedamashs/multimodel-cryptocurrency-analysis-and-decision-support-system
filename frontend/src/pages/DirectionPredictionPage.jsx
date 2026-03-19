import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchDirectionPrediction } from '../api';
import MetricCard from '../components/MetricCard';

const PIE_COLORS = ['#0e9f92', '#ef7e28'];

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

export default function DirectionPredictionPage() {
  const [lookback, setLookback] = useState(2000);
  const [horizon, setHorizon] = useState(7);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError('');
        const payload = await fetchDirectionPrediction({ lookback, threshold, horizon });
        setData(payload);
      } catch (err) {
        const message = err?.response?.data?.error || err.message || 'Failed to load direction prediction';
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [lookback, threshold, horizon]);

  const bestMetrics = useMemo(() => {
    if (!data?.modelMetrics?.length) return null;
    const target = data.bestModel;
    return data.modelMetrics.find((m) => m.model === target) || data.modelMetrics[0];
  }, [data]);

  if (loading) {
    return <section className="rounded-3xl bg-white p-8 shadow-soft">Training direction classifiers...</section>;
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-300 bg-red-50 p-8 text-red-800 shadow-soft">
        <h2 className="font-display text-xl font-bold">Direction module error</h2>
        <p className="mt-2 text-sm">{error}</p>
      </section>
    );
  }

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#1d4ed8] via-[#0f766e] to-[#0f172a] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-100">Module 3</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Direction Prediction (UP or DOWN)</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-100">
          Classification engine predicts next-day market direction from OHLCV and engineered momentum features.
        </p>
      </div>

      <div className="rounded-2xl bg-white p-4 shadow-soft">
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Lookback</p>
            <div className="mt-1 flex gap-2">
              {[1000, 2000, 4000].map((item) => (
                <button
                  key={item}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    lookback === item ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setLookback(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Threshold</p>
            <div className="mt-1 flex gap-2">
              {[0.45, 0.5, 0.55].map((item) => (
                <button
                  key={item}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    threshold === item ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setThreshold(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Signal Horizon</p>
            <div className="mt-1 flex gap-2">
              {[7, 14, 21].map((item) => (
                <button
                  key={item}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    horizon === item ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setHorizon(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Best Model" value={data.bestModel} />
        <MetricCard title="Latest Signal" value={data.latestSignal.direction} subtitle={`P(UP)=${pct(data.latestSignal.probabilityUp)}`} />
        <MetricCard title="Accuracy" value={pct(bestMetrics?.accuracy)} subtitle="Holdout" />
        <MetricCard title="F1 Score" value={fmt(bestMetrics?.f1, 4)} subtitle="Best model" />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Class Distribution</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.charts.classDistribution} dataKey="count" nameKey="label" outerRadius={112} label>
                  {data.charts.classDistribution.map((entry, index) => (
                    <Cell key={entry.label} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Confusion Matrix (Best Model)</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.charts.confusionMatrix}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="label" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#1d4ed8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Probability Curve (Test Period)</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.charts.probabilityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="Date" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="ProbabilityUp" stroke="#0e9f92" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Top Feature Importance</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.charts.featureImportance.slice(0, 8)} layout="vertical" margin={{ left: 70 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="importance" fill="#ef7e28" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Future Direction Signals</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                <th className="px-2 py-2">Date</th>
                <th className="px-2 py-2">Predicted Direction</th>
                <th className="px-2 py-2">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {data.charts.futureSignals.map((row) => (
                <tr key={row.Date} className="border-b border-slate-100 text-slate-700">
                  <td className="px-2 py-2">{row.Date}</td>
                  <td className="px-2 py-2 font-semibold text-ink">{row.Direction}</td>
                  <td className="px-2 py-2">{pct(row.Confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Model Metrics</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                <th className="px-2 py-2">Model</th>
                <th className="px-2 py-2">Accuracy</th>
                <th className="px-2 py-2">Precision</th>
                <th className="px-2 py-2">Recall</th>
                <th className="px-2 py-2">F1</th>
              </tr>
            </thead>
            <tbody>
              {data.modelMetrics.map((row) => (
                <tr key={row.model} className="border-b border-slate-100 text-slate-700">
                  <td className="px-2 py-2 font-semibold text-ink">{row.model}</td>
                  <td className="px-2 py-2">{pct(row.accuracy)}</td>
                  <td className="px-2 py-2">{pct(row.precision)}</td>
                  <td className="px-2 py-2">{pct(row.recall)}</td>
                  <td className="px-2 py-2">{fmt(row.f1, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
