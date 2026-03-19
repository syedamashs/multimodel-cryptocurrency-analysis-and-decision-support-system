import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchRiskClustering } from '../api';
import MetricCard from '../components/MetricCard';

const RISK_COLORS = {
  'Low Risk': '#16a34a',
  'Medium Risk': '#f59e0b',
  'High Risk': '#dc2626',
};

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value);
}

export default function RiskClusteringPage() {
  const [days, setDays] = useState(365);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError('');
        const payload = await fetchRiskClustering({ days, seed });
        setData(payload);
      } catch (err) {
        const message = err?.response?.data?.error || err.message || 'Failed to cluster risk';
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [days, seed]);

  const highRiskCount = useMemo(() => {
    if (!data?.coinAssignments) return 0;
    return data.coinAssignments.filter((item) => item.riskLevel === 'High Risk').length;
  }, [data]);

  const scatterData = useMemo(() => {
    if (!data?.charts?.scatter) return [];
    return data.charts.scatter.map((item) => ({ ...item, z: Math.max(item.meanVolume || 1, 1) }));
  }, [data]);

  if (loading) {
    return <section className="rounded-3xl bg-white p-8 shadow-soft">Running K-means risk clustering...</section>;
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-300 bg-red-50 p-8 text-red-800 shadow-soft">
        <h2 className="font-display text-xl font-bold">Risk clustering error</h2>
        <p className="mt-2 text-sm">{error}</p>
      </section>
    );
  }

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#7f1d1d] via-[#b91c1c] to-[#1e293b] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-red-100">Module 5</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Risk Clustering Engine</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-100">
          Multi-coin clustering groups assets into Low, Medium, and High risk profiles from volatility and drawdown behavior.
        </p>
      </div>

      <div className="rounded-2xl bg-white p-4 shadow-soft">
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Window</p>
            <div className="mt-1 flex gap-2">
              {[180, 365, 730, 1460].map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    days === option ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setDays(option)}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Seed</p>
            <div className="mt-1 flex gap-2">
              {[7, 42, 99].map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    seed === option ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setSeed(option)}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Coins" value={data.overview.coins} />
        <MetricCard title="Clusters" value={data.overview.clusters} />
        <MetricCard title="High Risk Coins" value={highRiskCount} />
        <MetricCard title="Safest Coin" value={data.insight.safestCoin} subtitle={`Risk=${fmt(data.insight.safestCoinRisk, 3)}`} />
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <p className="text-sm leading-6 text-slate-700">
          Riskiest coin is <span className="font-bold text-ink">{data.insight.riskiestCoin}</span> with risk index{' '}
          <span className="font-bold text-ink">{fmt(data.insight.riskiestCoinRisk, 3)}</span>. Clustering inertia is{' '}
          <span className="font-bold text-ink">{fmt(data.overview.inertia, 3)}</span>.
        </p>
      </article>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Volatility vs Drawdown Map</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="volatility" name="Volatility" />
                <YAxis dataKey="maxDrawdown" name="Max Drawdown" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                {['Low Risk', 'Medium Risk', 'High Risk'].map((risk) => (
                  <Scatter
                    key={risk}
                    name={risk}
                    data={scatterData.filter((row) => row.riskLevel === risk)}
                    fill={RISK_COLORS[risk]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Risk Level Distribution</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.charts.riskLevelCounts}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="riskLevel" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {data.charts.riskLevelCounts.map((entry) => (
                    <Cell key={entry.riskLevel} fill={RISK_COLORS[entry.riskLevel]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Cluster Risk Index Ranking</h3>
        <div className="mt-4 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.clusterSummary}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
              <XAxis dataKey="riskLevel" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="avgRiskIndex" fill="#b91c1c" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </article>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Coin Risk Assignment</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                <th className="px-2 py-2">Coin</th>
                <th className="px-2 py-2">Risk Level</th>
                <th className="px-2 py-2">Risk Index</th>
                <th className="px-2 py-2">Volatility</th>
                <th className="px-2 py-2">Max Drawdown</th>
                <th className="px-2 py-2">Volume CV</th>
              </tr>
            </thead>
            <tbody>
              {data.coinAssignments
                .slice()
                .sort((a, b) => b.riskIndex - a.riskIndex)
                .map((row) => (
                  <tr key={row.coin} className="border-b border-slate-100 text-slate-700">
                    <td className="px-2 py-2 font-semibold text-ink">{row.coin}</td>
                    <td className="px-2 py-2">
                      <span
                        className="rounded-full px-2 py-1 text-xs font-bold"
                        style={{
                          backgroundColor: `${RISK_COLORS[row.riskLevel]}20`,
                          color: RISK_COLORS[row.riskLevel],
                        }}
                      >
                        {row.riskLevel}
                      </span>
                    </td>
                    <td className="px-2 py-2">{fmt(row.riskIndex, 4)}</td>
                    <td className="px-2 py-2">{fmt(row.volatility, 4)}</td>
                    <td className="px-2 py-2">{fmt(row.maxDrawdown, 4)}</td>
                    <td className="px-2 py-2">{fmt(row.volumeCV, 4)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
