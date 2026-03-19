import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { analyzeImageNews } from '../api';
import MetricCard from '../components/MetricCard';

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

export default function ImageAnalysisPage() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  async function handleAnalyze() {
    if (!imageFile) {
      setError('Please select an image first.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      const response = await analyzeImageNews({ image: imageFile, question });
      setData(response);
    } catch (err) {
      const message = err?.response?.data?.error || err.message || 'Image analysis failed';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="space-y-6 animate-rise">
      <div className="rounded-3xl bg-gradient-to-r from-[#6d28d9] via-[#0f766e] to-[#1e293b] p-6 text-white shadow-soft">
        <p className="text-xs uppercase tracking-[0.2em] text-violet-100">Module 6</p>
        <h2 className="mt-2 font-display text-3xl font-bold">Image Intelligence</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-100">
          Upload a crypto-news image, extract text via Ollama vision model, run sentiment analysis, and get a human-friendly explanation.
        </p>
      </div>

      <article className="rounded-2xl bg-white p-5 shadow-soft">
        <h3 className="font-display text-lg font-bold text-ink">Upload & Analyze</h3>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-4">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setImageFile(file);
                setData(null);
                setError('');
                if (file) {
                  setPreviewUrl(URL.createObjectURL(file));
                } else {
                  setPreviewUrl('');
                }
              }}
              className="text-sm"
            />

            {previewUrl ? (
              <img src={previewUrl} alt="preview" className="mt-4 max-h-72 w-full rounded-xl border border-slate-200 object-contain" />
            ) : (
              <p className="mt-4 text-sm text-slate-500">Select a news screenshot, chart snapshot, or article image.</p>
            )}
          </div>

          <div>
            <label className="text-xs font-semibold uppercase tracking-[0.1em] text-steel">Optional Question to Ollama</label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Example: Is this news bullish or risky for short-term traders?"
              className="mt-2 h-32 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-ink focus:outline-none"
            />

            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={loading}
                className="rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-60"
              >
                {loading ? 'Processing image...' : 'Run Image Analysis'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setImageFile(null);
                  setPreviewUrl('');
                  setQuestion('');
                  setData(null);
                  setError('');
                }}
                className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                Reset
              </button>
            </div>

            {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}
          </div>
        </div>
      </article>

      {data ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard title="File Name" value={data.file.name} />
            <MetricCard title="Extracted Words" value={data.sentiment.overview.words} />
            <MetricCard title="Sentiment" value={data.sentiment.result.label} />
            <MetricCard title="Confidence" value={pct(data.sentiment.result.confidence)} />
          </div>

          <article className="rounded-2xl bg-white p-5 shadow-soft">
            <h3 className="font-display text-lg font-bold text-ink">Ollama Human-Friendly Explanation</h3>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">{data.ollamaSummary}</p>
          </article>

          <div className="grid gap-6 xl:grid-cols-2">
            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Extracted Text (OCR)</h3>
              <textarea
                readOnly
                value={data.extractedText}
                className="mt-3 h-72 w-full rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800"
              />
            </article>

            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Sentiment Model Comparison</h3>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.sentiment.charts.modelComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                    <XAxis dataKey="model" tick={{ fontSize: 11 }} interval={0} angle={-12} height={70} />
                    <YAxis domain={[-1, 1]} />
                    <Tooltip />
                    <Bar dataKey="score" fill="#6d28d9" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-sm text-slate-600">Best model: <span className="font-semibold text-ink">{data.sentiment.bestModel}</span></p>
            </article>
          </div>

          {data.qa?.question ? (
            <article className="rounded-2xl bg-white p-5 shadow-soft">
              <h3 className="font-display text-lg font-bold text-ink">Answer to Your Question</h3>
              <p className="mt-2 text-sm text-slate-500">Q: {data.qa.question}</p>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">{data.qa.answer || 'No answer returned.'}</p>
            </article>
          ) : null}

          <article className="rounded-2xl bg-white p-5 shadow-soft">
            <h3 className="font-display text-lg font-bold text-ink">Sentence Sentiment Scores</h3>
            <div className="mt-4 h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.sentiment.charts.sentenceTimeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d7e2e7" />
                  <XAxis dataKey="index" />
                  <YAxis domain={[-1, 1]} />
                  <Tooltip />
                  <Bar dataKey="score" fill="#0f766e" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
        </>
      ) : null}
    </section>
  );
}
