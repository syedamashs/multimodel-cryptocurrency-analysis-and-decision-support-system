import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
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
import { fetchPricePrediction } from '../api';
import MetricCard from '../components/MetricCard';

function formatNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value);
}

export default function PricePredictionPage() {
  const [horizon, setHorizon] = useState(7);
  const [lookback, setLookback] = useState(365);
  const [frequency, setFrequency] = useState('D');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError('');
        const payload = await fetchPricePrediction({ horizon, lookback, frequency });
        setData(payload);
      } catch (err) {
        const message = err?.response?.data?.error || err.message || 'Failed to load predictions';
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [horizon, lookback, frequency]);

  const metricBars = useMemo(() => {
    if (!data?.modelMetrics) return [];
    return data.modelMetrics.map((item) => ({
      model: item.model,
      RMSE: item.rmse,
      MAE: item.mae,
      MAPE: item.mape,
    }));
  }, [data]);

  if (loading) {
    return <section className="rounded-3xl bg-white p-8 shadow-soft">Training and forecasting models...</section>;
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-300 bg-red-50 p-8 text-red-800 shadow-soft">
        <h2 className="font-display text-xl font-bold">Prediction module error</h2>
        <p className="mt-2 text-sm">{error}</p>
      </section>
    );
  }

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#92400e] via-[#b45309] to-[#0f172a] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-amber-100">Module 2</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Bitcoin Price Prediction Engine</h2>
        <p className="mt-2 max-w-3xl text-sm text-orange-50">
          Regression and forecasting models are trained and evaluated on historical Bitcoin data from dataset-2.
        </p>
      </div>

      <div className="rounded-2xl bg-white p-4 shadow-soft">
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Horizon</p>
            <div className="mt-1 flex gap-2">
              {[7, 14, 30].map((item) => (
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

          <div className="rounded-xl bg-slate-100 p-2">
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Lookback Points</p>
            <div className="mt-1 flex gap-2">
              {[365, 730, 1460].map((item) => (
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
            <p className="px-2 text-xs font-semibold uppercase tracking-[0.1em] text-steel">Frequency</p>
            <div className="mt-1 flex gap-2">
              {[
                { id: 'D', label: 'Daily' },
                { id: 'W', label: 'Weekly' },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`rounded-lg px-3 py-1 text-sm font-semibold ${
                    frequency === item.id ? 'bg-ink text-white' : 'bg-white text-slate-700'
                  }`}
                  onClick={() => setFrequency(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Best Model" value={data.bestModel} />
        <MetricCard title="Next Forecast" value={formatNum(data.nextForecast, 3)} subtitle="Predicted next close" />
        <MetricCard title="Data Points" value={formatNum(data.overview.points, 0)} subtitle={`${data.overview.frequency} series`} />
        <MetricCard title="Last Close" value={formatNum(data.overview.lastClose, 3)} />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Model Error Comparison (RMSE)</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricBars} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} interval={0} angle={-18} height={70} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="RMSE" fill="#b45309" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Holdout Fit (Actual vs Models)</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.charts.testComparison} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="Date" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Actual" stroke="#0f172a" dot={false} strokeWidth={2.5} />
                <Line type="monotone" dataKey="Linear Regression" stroke="#0e9f92" dot={false} strokeWidth={1.8} />
                <Line type="monotone" dataKey="Autoregressive (AR-7)" stroke="#b45309" dot={false} strokeWidth={1.8} />
                <Line type="monotone" dataKey="Holt Linear Trend" stroke="#3b82f6" dot={false} strokeWidth={1.8} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Historical Close Trend</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.charts.history} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="Date" tick={{ fontSize: 11 }} minTickGap={26} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="Close" stroke="#0e9f92" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="rounded-2xl bg-white p-5 shadow-soft">
          <h3 className="font-display text-lg font-bold text-ink">Future Forecast with Confidence Band</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.charts.futureForecast} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                <XAxis dataKey="Date" tick={{ fontSize: 11 }} minTickGap={20} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="UpperBand"
                  stackId="1"
                  stroke="none"
                  fill="#fde68a"
                  fillOpacity={0.55}
                />
                <Area
                  type="monotone"
                  dataKey="LowerBand"
                  stackId="1"
                  stroke="none"
                  fill="#ffffff"
                  fillOpacity={1}
                />
                <Line type="monotone" dataKey="PredictedClose" stroke="#b45309" dot strokeWidth={2.5} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Model Metrics Table</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                <th className="px-2 py-2">Model</th>
                <th className="px-2 py-2">MAE</th>
                <th className="px-2 py-2">RMSE</th>
                <th className="px-2 py-2">MAPE %</th>
              </tr>
            </thead>
            <tbody>
              {data.modelMetrics.map((row) => (
                <tr key={row.model} className="border-b border-slate-100 text-slate-700">
                  <td className="px-2 py-2 font-semibold text-ink">{row.model}</td>
                  <td className="px-2 py-2">{formatNum(row.mae, 4)}</td>
                  <td className="px-2 py-2">{formatNum(row.rmse, 4)}</td>
                  <td className="px-2 py-2">{formatNum(row.mape, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
