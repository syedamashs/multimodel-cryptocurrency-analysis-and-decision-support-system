import { FiActivity, FiBarChart2, FiCompass, FiImage, FiMessageSquare, FiShield } from 'react-icons/fi';

const tabs = [
  { id: 'market-analysis', label: 'Market Analysis', icon: FiActivity, status: 'active' },
  { id: 'price-prediction', label: 'Price Prediction', icon: FiBarChart2, status: 'active' },
  { id: 'direction-prediction', label: 'Direction Prediction', icon: FiCompass, status: 'active' },
  { id: 'sentiment-analysis', label: 'Sentiment Analysis', icon: FiMessageSquare, status: 'active' },
  { id: 'risk-clustering', label: 'Risk Clustering', icon: FiShield, status: 'active' },
  { id: 'image-analysis', label: 'Image Intelligence', icon: FiImage, status: 'active' },
];

export default function Sidebar({ activeTab, onChangeTab }) {
  return (
    <aside className="self-start rounded-3xl bg-gradient-to-b from-[#0c2633] via-[#14394a] to-[#1f4f5a] p-4 shadow-soft md:p-6 lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)]">
      <div className="mb-10 rounded-2xl bg-white/10 p-5">
        <p className="text-xs uppercase tracking-[0.22em] text-slate-200">Crypto AI Lab</p>
        <h1 className="mt-2 text-xl font-display font-bold text-white">Decision Support</h1>
      </div>

      <nav className="space-y-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = tab.id === activeTab;
          const isUpcoming = tab.status === 'upcoming';

          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onChangeTab(tab.id)}
              className={`w-full rounded-2xl px-4 py-4 text-left transition ${
                isActive
                  ? 'bg-white text-ink'
                  : 'bg-white/5 text-slate-200 hover:bg-white/15'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <Icon className="text-xl" />
                  <span className="text-[15px] font-semibold">{tab.label}</span>
                </div>
                {isUpcoming ? (
                  <span className="rounded-full bg-accent/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-accent">
                    Soon
                  </span>
                ) : null}
              </div>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
