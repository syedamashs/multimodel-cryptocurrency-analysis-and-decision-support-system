import { useMemo, useRef, useState } from 'react';
import { FiCpu, FiMessageCircle, FiSend, FiX } from 'react-icons/fi';
import { askChatbot } from '../api';

const QUICK_PROMPTS = [
  'Summarize current dashboard insights in simple terms.',
  'Based on modules, what should I check before BUY?',
  'Explain risk clustering output for a beginner.',
  'What does sentiment confidence mean?'
];

export default function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hi, I am CryptoPilot. Ask me about your market, price prediction, direction, sentiment, or risk clustering results.'
    }
  ]);
  const scrollRef = useRef(null);

  const recentHistory = useMemo(() => messages.slice(-8), [messages]);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  }

  async function handleSend(customText) {
    const text = (customText ?? input).trim();
    if (!text || isLoading) return;

    const nextMessages = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setInput('');
    setIsLoading(true);
    scrollToBottom();

    try {
      const payloadHistory = recentHistory.map((m) => ({ role: m.role, content: m.content }));
      const response = await askChatbot({ message: text, history: payloadHistory });
      setMessages((prev) => [...prev, { role: 'assistant', content: response.answer }]);
    } catch (err) {
      const message = err?.response?.data?.error || err.message || 'Chatbot request failed.';
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${message}` }]);
    } finally {
      setIsLoading(false);
      scrollToBottom();
    }
  }

  return (
    <>
      {isOpen ? (
        <div className="fixed bottom-24 right-4 z-50 flex h-[min(84vh,620px)] w-[min(96vw,430px)] flex-col overflow-hidden animate-rise rounded-3xl border border-slate-200 bg-white shadow-[0_22px_60px_rgba(15,23,42,0.28)] md:right-6">
          <div className="flex items-center justify-between rounded-t-3xl bg-gradient-to-r from-[#0f766e] via-[#1d4ed8] to-[#0f172a] px-4 py-3 text-white">
            <div className="flex items-center gap-2">
              <FiCpu className="text-lg" />
              <div>
                <p className="text-sm font-bold">CryptoPilot Chat</p>
                <p className="text-[11px] text-slate-200">Powered by Ollama Llama 3</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="rounded-full p-2 text-white/90 transition hover:bg-white/20"
            >
              <FiX />
            </button>
          </div>

          <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            <div className="space-y-3">
              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-6 ${
                    message.role === 'user'
                      ? 'ml-auto bg-ink text-white'
                      : 'bg-slate-100 text-slate-800'
                  }`}
                >
                  {message.content}
                </div>
              ))}
              {isLoading ? (
                <div className="max-w-[85%] rounded-2xl bg-slate-100 px-3 py-2 text-sm text-slate-600">
                  Thinking...
                </div>
              ) : null}
            </div>
          </div>

          <div className="mt-auto border-t border-slate-200 bg-white px-3 py-2">
            <div className="mb-2 flex flex-wrap gap-2">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => handleSend(prompt)}
                  className="rounded-full border border-slate-200 px-2 py-1 text-[11px] font-semibold text-steel hover:bg-slate-50"
                >
                  {prompt}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask anything about your crypto dashboard..."
                className="h-10 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-ink focus:outline-none"
              />
              <button
                type="button"
                onClick={() => handleSend()}
                disabled={isLoading}
                className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink text-white transition hover:opacity-90 disabled:opacity-60"
                aria-label="Send message"
              >
                <FiSend />
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed bottom-5 right-4 z-50 flex items-center gap-2 rounded-full bg-gradient-to-r from-[#0f766e] via-[#1d4ed8] to-[#0f172a] px-4 py-3 text-sm font-bold text-white shadow-[0_12px_32px_rgba(15,23,42,0.32)] transition hover:scale-[1.03] md:right-6"
      >
        <FiMessageCircle className="text-lg" />
        Chatbot
      </button>
    </>
  );
}
