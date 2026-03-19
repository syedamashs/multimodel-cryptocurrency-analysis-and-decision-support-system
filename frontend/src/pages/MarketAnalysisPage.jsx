import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchMarketAnalysis } from '../api';
import MetricCard from '../components/MetricCard';

const COLORS = ['#0e9f92', '#ef7e28', '#2d4f60', '#0f766e', '#3b82f6', '#ec4899', '#8b5cf6'];

function formatNum(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(value);
}

export default function MarketAnalysisPage() {
  const [days, setDays] = useState(365);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError('');
        const payload = await fetchMarketAnalysis(days);
        setData(payload);
      } catch (err) {
        const message = err?.response?.data?.error || err.message || 'Failed to load module data';
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [days]);

  const trendChartData = useMemo(() => {
    if (!data?.charts?.trend) return [];

    const byDate = new Map();
    for (const point of data.charts.trend) {
      if (!byDate.has(point.Date)) {
        byDate.set(point.Date, { Date: point.Date });
      }
      byDate.get(point.Date)[point.Coin] = point.Close;
    }

    return Array.from(byDate.values());
  }, [data]);

  const trendCoins = useMemo(() => {
    if (!data?.coinMetrics) return [];
    return data.coinMetrics.map((item) => item.coin).slice(0, 8);
  }, [data]);

  if (loading) {
    return <section className="rounded-3xl bg-white p-8 shadow-soft">Loading market analysis...</section>;
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-300 bg-red-50 p-8 text-red-800 shadow-soft">
        <h2 className="font-display text-xl font-bold">Data loading error</h2>
        <p className="mt-2 text-sm">{error}</p>
      </section>
    );
  }

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#0f766e] via-[#155e75] to-[#1e293b] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-100">Module 1</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Market Analysis Intelligence</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-100">
          Statistical and behavioral understanding of your 23-coin market data from dataset-1.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 rounded-2xl bg-white p-4 shadow-soft">
        <label className="text-xs font-semibold uppercase tracking-[0.12em] text-steel">Time Window</label>
        {[180, 365, 730, 1095].map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setDays(option)}
            className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
              days === option
                ? 'bg-ink text-white'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            {option} days
          </button>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Coins" value={data.overview.coins} />
        <MetricCard title="Rows" value={formatNum(data.overview.rows)} />
        <MetricCard title="Date Start" value={data.overview.dateStart} />
        <MetricCard title="Date End" value={data.overview.dateEnd} />
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <p className="text-xs uppercase tracking-[0.14em] text-steel">System Insight</p>
        <p className="mt-3 text-sm leading-6 text-slate-700">
          Market condition is <span className="font-bold text-ink">{data.insight.marketVolatility}</span> with an
          overall <span className="font-bold text-ink">{data.insight.marketTrend}</span> trend. Strongest
          price-volume behavior is in <span className="font-bold text-ink">{data.insight.strongestPriceVolumeCoin}</span>
          {' '}with correlation{' '}
          <span className="font-bold text-ink">{formatNum(data.insight.strongestPriceVolumeCorrelation)}</span>.
        </p>
      </article>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Volatility Ranking</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.charts.volatility} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="coin" tick={{ fontSize: 11 }} interval={0} angle={-25} height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="annualizedVolatility" fill="#0e9f92" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Price-Volume Correlation</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.charts.correlation} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="coin" tick={{ fontSize: 11 }} interval={0} angle={-25} height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="priceVolumeCorrelation" fill="#ef7e28" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Closing Price Trend (Top 8 Coins by Volatility)</h3>
        <div className="mt-4 h-[26rem]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendChartData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
              <XAxis dataKey="Date" tick={{ fontSize: 11 }} minTickGap={32} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {trendCoins.map((coin, index) => (
                <Line
                  key={coin}
                  type="monotone"
                  dataKey={coin}
                  dot={false}
                  strokeWidth={2}
                  stroke={COLORS[index % COLORS.length]}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </article>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Coin Metrics Table</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                <th className="px-2 py-2">Coin</th>
                <th className="px-2 py-2">Mean Close</th>
                <th className="px-2 py-2">Variance Close</th>
                <th className="px-2 py-2">Annualized Volatility</th>
                <th className="px-2 py-2">Trend Slope</th>
                <th className="px-2 py-2">Price-Volume Corr</th>
              </tr>
            </thead>
            <tbody>
              {data.coinMetrics.map((row) => (
                <tr key={row.coin} className="border-b border-slate-100 text-slate-700">
                  <td className="px-2 py-2 font-semibold text-ink">{row.coin}</td>
                  <td className="px-2 py-2">{formatNum(row.meanClose)}</td>
                  <td className="px-2 py-2">{formatNum(row.varianceClose)}</td>
                  <td className="px-2 py-2">{formatNum(row.annualizedVolatility)}</td>
                  <td className="px-2 py-2">{formatNum(row.trendSlope)}</td>
                  <td className="px-2 py-2">{formatNum(row.priceVolumeCorrelation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
