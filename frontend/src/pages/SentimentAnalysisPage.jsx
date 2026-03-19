import { useMemo, useState } from 'react';
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
import { analyzeSentimentText } from '../api';
import MetricCard from '../components/MetricCard';

const PIE_COLORS = ['#16a34a', '#64748b', '#dc2626'];

const SAMPLE_TEXTS = [
  'Bitcoin adoption is accelerating as more institutions add BTC to treasury, and market momentum remains bullish after ETF inflows.',
  'Crypto market shows uncertainty due to regulatory pressure and fear of liquidation, while bearish sentiment increases across traders.',
  'Bitcoin is moving sideways with mixed signals; adoption remains strong but short-term volatility and risk are still high.',
];

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

export default function SentimentAnalysisPage() {
  const [text, setText] = useState(SAMPLE_TEXTS[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  async function runAnalysis(payloadText) {
    try {
      setLoading(true);
      setError('');
      const response = await analyzeSentimentText(payloadText);
      setData(response);
    } catch (err) {
      const message = err?.response?.data?.error || err.message || 'Failed to analyze sentiment';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const resultToneClass = useMemo(() => {
    if (!data?.result?.label) return 'bg-slate-100 text-slate-700';
    if (data.result.label === 'Positive') return 'bg-green-100 text-green-800';
    if (data.result.label === 'Negative') return 'bg-red-100 text-red-800';
    return 'bg-slate-100 text-slate-700';
  }, [data]);

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#166534] via-[#0f766e] to-[#1e293b] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-green-100">Module 4</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Sentiment Analysis Intelligence</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-100">
          NLP mood detection for tweets, headlines, and user narratives. No external dataset required for baseline analysis.
        </p>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="font-display text-lg font-bold text-ink">Text Input</h3>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_TEXTS.map((sample, idx) => (
              <button
                key={sample}
                type="button"
                className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-steel hover:bg-slate-50"
                onClick={() => {
                  setText(sample);
                  runAnalysis(sample);
                }}
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="mt-4 h-36 w-full rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800 focus:border-ink focus:outline-none"
          placeholder="Paste crypto tweet/news/user text here..."
        />

        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => runAnalysis(text)}
            disabled={loading}
            className="rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-60"
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>
          <button
            type="button"
            onClick={() => {
              setText('');
              setData(null);
              setError('');
            }}
            className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            Clear
          </button>
        </div>

        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}
      </article>

      {data ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard title="Sentiment Label" value={data.result.label} subtitle="Overall mood" />
            <MetricCard title="Sentiment Score" value={fmt(data.result.score, 4)} subtitle="-1 (negative) to +1 (positive)" />
            <MetricCard title="Confidence" value={pct(data.result.confidence)} />
            <MetricCard title="Best Model" value={data.bestModel} subtitle={`Train rows: ${data.overview.trainingRows || 0}`} />
          </div>

          <article className="rounded-2xl bg-white p-5 shadow-soft">
            <h3 className="font-display text-lg font-bold text-ink">Sentiment Model Comparison</h3>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.charts.modelComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} interval={0} angle={-12} height={70} />
                  <YAxis domain={[-1, 1]} />
                  <Tooltip />
                  <Bar dataKey="score" fill="#166534" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                    <th className="px-2 py-2">Model</th>
                    <th className="px-2 py-2">Label</th>
                    <th className="px-2 py-2">Score</th>
                    <th className="px-2 py-2">Confidence</th>
                    <th className="px-2 py-2">Validation Acc</th>
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((row) => (
                    <tr key={row.model} className="border-b border-slate-100 text-slate-700">
                      <td className="px-2 py-2 font-semibold text-ink">{row.model}</td>
                      <td className="px-2 py-2">{row.label}</td>
                      <td className="px-2 py-2">{fmt(row.score, 4)}</td>
                      <td className="px-2 py-2">{pct(row.confidence)}</td>
                      <td className="px-2 py-2">{pct(row.validationAccuracy || 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="rounded-2xl bg-white p-5 shadow-soft">
            <div className="flex items-center gap-3">
              <p className="text-sm text-slate-600">Current Mood:</p>
              <span className={`rounded-full px-3 py-1 text-sm font-bold ${resultToneClass}`}>{data.result.label}</span>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">{data.result.recommendation}</p>
          </article>

          <div className="grid gap-6 xl:grid-cols-2">
            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Sentence Distribution</h3>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={data.charts.sentenceDistribution} dataKey="count" nameKey="label" label outerRadius={110}>
                      {data.charts.sentenceDistribution.map((entry, index) => (
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
              <h3 className="font-display text-lg font-bold text-ink">Aspect Sentiment</h3>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.charts.aspectSentiment}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                    <XAxis dataKey="aspect" tick={{ fontSize: 11 }} />
                    <YAxis domain={[-1, 1]} />
                    <Tooltip />
                    <Bar dataKey="score" fill="#0f766e" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </article>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Sentence Timeline</h3>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.charts.sentenceTimeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                    <XAxis dataKey="index" />
                    <YAxis domain={[-1, 1]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2.2} dot />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </article>

            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Top Keyword Impact</h3>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.charts.tokenImpact.slice(0, 10)} layout="vertical" margin={{ left: 70 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="token" width={110} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="impact" fill="#f59e0b" radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </article>
          </div>

          <article className="rounded-2xl bg-white p-5 shadow-soft">
            <h3 className="font-display text-lg font-bold text-ink">Sentence-Level Breakdown</h3>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-steel">
                    <th className="px-2 py-2">Sentence</th>
                    <th className="px-2 py-2">Label</th>
                    <th className="px-2 py-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.sentenceAnalysis.map((row) => (
                    <tr key={row.sentence} className="border-b border-slate-100 text-slate-700">
                      <td className="px-2 py-2">{row.sentence}</td>
                      <td className="px-2 py-2 font-semibold text-ink">{row.label}</td>
                      <td className="px-2 py-2">{fmt(row.score, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </>
      ) : null}
    </section>
  );
}
