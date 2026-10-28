import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  CheckCircle2,
  ChevronDown,
  Clock,
  CreditCard,
  Database,
  Download,
  Fingerprint,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TrendingUp,
  Wallet,
  X,
  Zap
} from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const initialTransaction = {
  Time: '0',
  V1: '0',
  V2: '0',
  V3: '0',
  V4: '0',
  V5: '0',
  V6: '0',
  V7: '0',
  V8: '0',
  V9: '0',
  V10: '0',
  V11: '0',
  V12: '0',
  V13: '0',
  V14: '0',
  V15: '0',
  V16: '0',
  V17: '0',
  V18: '0',
  V19: '0',
  V20: '0',
  V21: '0',
  V22: '0',
  V23: '0',
  V24: '0',
  V25: '0',
  V26: '0',
  V27: '0',
  V28: '0',
  Amount: '100'
}

const initialActivity = [
  {
    id: 'TXN-92841',
    merchant: 'Online payment',
    amount: 2450.0,
    time: '10:42:18',
    status: 'Legitimate',
    score: 3
  },
  {
    id: 'TXN-92840',
    merchant: 'Card transaction',
    amount: 18750.0,
    time: '10:40:52',
    status: 'Review',
    score: 67
  },
  {
    id: 'TXN-92839',
    merchant: 'Retail purchase',
    amount: 890.0,
    time: '10:38:14',
    status: 'Legitimate',
    score: 8
  },
  {
    id: 'TXN-92838',
    merchant: 'Digital transfer',
    amount: 52600.0,
    time: '10:35:09',
    status: 'Review',
    score: 74
  },
  {
    id: 'TXN-92837',
    merchant: 'Online subscription',
    amount: 1299.0,
    time: '10:31:45',
    status: 'Legitimate',
    score: 2
  }
]

const chartData = [
  { time: '08:00', legitimate: 320, fraud: 12 },
  { time: '09:00', legitimate: 460, fraud: 18 },
  { time: '10:00', legitimate: 390, fraud: 15 },
  { time: '11:00', legitimate: 570, fraud: 28 },
  { time: '12:00', legitimate: 510, fraud: 21 },
  { time: '13:00', legitimate: 680, fraud: 34 },
  { time: '14:00', legitimate: 590, fraud: 25 },
  { time: '15:00', legitimate: 760, fraud: 39 },
  { time: '16:00', legitimate: 650, fraud: 30 },
  { time: '17:00', legitimate: 820, fraud: 45 },
  { time: '18:00', legitimate: 730, fraud: 33 },
  { time: '19:00', legitimate: 900, fraud: 51 }
]

const initialDistribution = [
  { name: 'Legitimate', value: 96, color: '#16b894' },
  { name: 'Fraudulent', value: 4, color: '#f05268' }
]

function formatCurrency(value) {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2
  }).format(value)
}

function StatCard({ title, value, subtitle, icon: Icon, trend, color }) {
  return (
    <article className="stat-card">
      <div className="stat-top">
        <span className="stat-label">{title}</span>
        <span className={`stat-icon ${color}`}>
          <Icon size={20} />
        </span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-bottom">
        <span className={`stat-trend ${trend < 0 ? 'negative' : ''}`}>
          {trend < 0 ? <ArrowDownRight size={15} /> : <ArrowUpRight size={15} />}
          {Math.abs(trend)}%
        </span>
        <span className="stat-subtitle">{subtitle}</span>
      </div>
    </article>
  )
}

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [mobileMenu, setMobileMenu] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [modelLoaded, setModelLoaded] = useState(false)
  const [checkingApi, setCheckingApi] = useState(false)
  const [transaction, setTransaction] = useState(initialTransaction)
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('All')
  const [activity, setActivity] = useState(initialActivity)
  const [notifications, setNotifications] = useState(false)
  const [showAllFeatures, setShowAllFeatures] = useState(false)

  const navigation = [
    { label: 'Overview', icon: LayoutDashboard },
    { label: 'Transactions', icon: CreditCard },
    { label: 'Fraud Detection', icon: ShieldAlert },
    { label: 'Analytics', icon: BarChart3 }
  ]

  const checkApi = async () => {
    setCheckingApi(true)

    try {
      const response = await fetch(`${API_URL}/health`)

      if (!response.ok) {
        throw new Error('API health check failed')
      }

      const data = await response.json()
      setApiStatus('online')
      setModelLoaded(Boolean(data.model_loaded))
    } catch {
      setApiStatus('offline')
      setModelLoaded(false)
    } finally {
      setCheckingApi(false)
    }
  }

  useEffect(() => {
    checkApi()
  }, [])

  const filteredActivity = useMemo(() => {
    return activity.filter((item) => {
      const matchesSearch =
        item.id.toLowerCase().includes(search.toLowerCase()) ||
        item.merchant.toLowerCase().includes(search.toLowerCase())

      const matchesFilter = filter === 'All' || item.status === filter

      return matchesSearch && matchesFilter
    })
  }, [activity, search, filter])

  const handleInputChange = (event) => {
    const { name, value } = event.target

    setTransaction((current) => ({
      ...current,
      [name]: value
    }))
  }

  const handlePredict = async (event) => {
    event.preventDefault()
    setError('')
    setPrediction(null)
    setPredicting(true)

    if (apiStatus !== 'online') {
      setError('The backend API is offline. Start your FastAPI server and try again.')
      setPredicting(false)
      return
    }

    try {
      const payload = {}

      for (const [key, value] of Object.entries(transaction)) {
        if (value.trim() === '') {
          throw new Error(`Please enter a value for ${key}.`)
        }

        const number = Number(value)

        if (!Number.isFinite(number)) {
          throw new Error(`${key} must be a valid number.`)
        }

        payload[key] = number
      }

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify(payload)
      })

      const data = await response.json()

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(', ')
          : data.detail || data.message

        throw new Error(detail || 'The prediction request failed.')
      }

      const result = {
        ...data,
        prediction: Number(data.prediction),
        fraud_probability: Number(data.fraud_probability),
        threshold: Number(data.threshold)
      }

      setPrediction(result)

      const riskScore = Math.round(result.fraud_probability * 100)

      setActivity((current) => [
        {
          id: `TXN-${Date.now().toString().slice(-8)}`,
          merchant: 'Manual risk assessment',
          amount: Number(transaction.Amount),
          time: new Date().toLocaleTimeString('en-IN'),
          status:
            result.prediction === 1
              ? 'Fraudulent'
              : riskScore >= 40
                ? 'Review'
                : 'Legitimate',
          score: riskScore
        },
        ...current
      ].slice(0, 20))

    } catch (err) {
      setError(err.message || 'Unable to predict this transaction.')
    } finally {
      setPredicting(false)
    }
  }

  const resetTransaction = () => {
    setTransaction(initialTransaction)
    setPrediction(null)
    setError('')
  }

  const exportTransactions = () => {
    const headers = ['Transaction ID', 'Merchant', 'Amount', 'Time', 'Status', 'Risk Score']
    const rows = filteredActivity.map((item) => [
      item.id,
      item.merchant,
      item.amount,
      item.time,
      item.status,
      item.score
    ])

    const csv = [headers, ...rows]
      .map((row) =>
        row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')
      )
      .join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = 'fraud-detection-transactions.csv'
    link.click()

    URL.revokeObjectURL(url)
  }

  const goToPage = (page) => {
    setActivePage(page)
    setMobileMenu(false)

    if (page === 'Overview') {
      document.getElementById('overview')?.scrollIntoView({ behavior: 'smooth' })
    } else if (page === 'Transactions') {
      document.getElementById('transactions')?.scrollIntoView({ behavior: 'smooth' })
    } else if (page === 'Fraud Detection') {
      document.getElementById('predictor')?.scrollIntoView({ behavior: 'smooth' })
    } else {
      document.getElementById('analytics')?.scrollIntoView({ behavior: 'smooth' })
    }
  }

  const legitimateCount = activity.filter(
    (item) => item.status === 'Legitimate'
  ).length

  const fraudCount = activity.filter(
    (item) => item.status === 'Fraudulent'
  ).length

  const reviewCount = activity.filter(
    (item) => item.status === 'Review'
  ).length

  const distribution = [
    {
      name: 'Legitimate',
      value: legitimateCount || 1,
      color: '#16b894'
    },
    {
      name: 'Fraudulent',
      value: fraudCount || 0,
      color: '#f05268'
    },
    {
      name: 'Review',
      value: reviewCount || 0,
      color: '#f5a623'
    }
  ]

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-symbol">
            <Shield size={24} />
            <span className="brand-spark">✦</span>
          </div>
          <div>
            <div className="brand-name">FraudGuard<span> AI</span></div>
            <div className="brand-caption">INTELLIGENCE PLATFORM</div>
          </div>
          <button
            className="icon-button sidebar-close"
            onClick={() => setMobileMenu(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${activePage === label ? 'nav-active' : ''}`}
              onClick={() => goToPage(label)}
            >
              <Icon size={19} />
              <span>{label}</span>
              {label === 'Fraud Detection' && (
                <span className="nav-badge">AI</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-divider" />

        <div className="workspace-label">SYSTEM</div>

        <button
          className="nav-item"
          onClick={checkApi}
          disabled={checkingApi}
        >
          {checkingApi ? (
            <LoaderCircle className="spin" size={19} />
          ) : (
            <Activity size={19} />
          )}
          <span>API Health</span>
        </button>

        <button
          className="nav-item"
          onClick={() => setShowAllFeatures((value) => !value)}
        >
          <Database size={19} />
          <span>Model Information</span>
          <ChevronDown
            size={15}
            className={showAllFeatures ? 'rotate-icon' : ''}
          />
        </button>

        {showAllFeatures && (
          <div className="model-info">
            <p>Algorithm</p>
            <strong>XGBoost Classifier</strong>
            <p>Training dataset</p>
            <strong>Credit Card Fraud</strong>
            <p>Model status</p>
            <strong>{modelLoaded ? 'Loaded' : 'Unavailable'}</strong>
          </div>
        )}

        <div className="sidebar-bottom">
          <div className="sidebar-security">
            <div className="security-icon">
              <Fingerprint size={22} />
            </div>
            <strong>Secure by design</strong>
            <p>AI-powered transaction risk analysis.</p>
            <span><ShieldCheck size={14} /> Fraud monitoring</span>
          </div>

          <div className="profile">
            <div className="avatar">FG</div>
            <div className="profile-text">
              <strong>Fraud Analyst</strong>
              <span>Administrator</span>
            </div>
            <button
              className="icon-button"
              onClick={() => setNotifications((value) => !value)}
              aria-label="Show notifications"
            >
              <Bell size={18} />
            </button>
          </div>
        </div>
      </aside>

      {mobileMenu && (
        <button
          className="mobile-overlay"
          aria-label="Close menu"
          onClick={() => setMobileMenu(false)}
        />
      )}

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button mobile-menu-button"
              onClick={() => setMobileMenu(true)}
              aria-label="Open menu"
            >
              <Menu size={21} />
            </button>
            <div>
              <div className="breadcrumb">Workspace <span>/</span> {activePage}</div>
              <h1>{activePage}</h1>
            </div>
          </div>

          <div className="topbar-actions">
<button
              className="icon-button refresh-button"
              onClick={checkApi}
              disabled={checkingApi}
              aria-label="Refresh API status"
              title="Refresh API status"
            >
              <RefreshCw size={18} className={checkingApi ? 'spin' : ''} />
            </button>

            <div className="notification-wrap">
              <button
                className="icon-button notification-button"
                onClick={() => setNotifications((value) => !value)}
                aria-label="Notifications"
              >
                <Bell size={19} />
                <span className="notification-dot" />
              </button>

              {notifications && (
                <div className="notification-popover">
                  <div className="notification-heading">
                    <strong>Notifications</strong>
                    <button
                      className="icon-button"
                      onClick={() => setNotifications(false)}
                      aria-label="Close notifications"
                    >
                      <X size={16} />
                    </button>
                  </div>
                  <p>
                    <Shield size={16} />
                    {modelLoaded
                      ? 'Fraud detection model is loaded.'
                      : 'Model availability needs verification.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-banner">
            <div className="welcome-copy">
              <div className="eyebrow"><Sparkles size={14} /> AI-POWERED MONITORING</div>
              <h2>Financial security,<br /> powered by intelligence.</h2>
              <p>
                Monitor transaction risk, analyze fraud signals, and evaluate
                transactions using your machine learning model.
              </p>
              <button
                className="primary-button"
                onClick={() => goToPage('Fraud Detection')}
              >
                <Zap size={17} /> Analyze a transaction
              </button>
            </div>
            <div className="welcome-art" aria-hidden="true">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="orbit orbit-three" />
              <div className="shield-glow">
                <ShieldCheck size={82} strokeWidth={1.2} />
              </div>
              <div className="floating-card card-one">
                <CheckCircle2 size={16} /> Risk monitoring
              </div>
              <div className="floating-card card-two">
                <Activity size={16} /> XGBoost AI
              </div>
              <div className="floating-card card-three">
                <Fingerprint size={16} /> Transaction analysis
              </div>
            </div>
          </section>

          <section id="overview" className="stats-grid">
            <StatCard
              title="Transactions in sample"
              value="56,962"
              subtitle="Model test transactions"
              icon={CreditCard}
              trend={12.4}
              color="purple"
            />
            <StatCard
              title="Fraud cases in sample"
              value="98"
              subtitle="Positive test examples"
              icon={ShieldAlert}
              trend={8.2}
              color="red"
            />
            <StatCard
              title="Fraud recall"
              value="86.7%"
              subtitle="Training evaluation result"
              icon={TargetIcon}
              trend={4.8}
              color="green"
            />
            <StatCard
              title="Model ROC-AUC"
              value="0.9808"
              subtitle="Recorded evaluation metric"
              icon={TrendingUp}
              trend={2.1}
              color="blue"
            />
          </section>

          <section id="analytics" className="analytics-grid">
            <article className="panel chart-panel">
              <div className="panel-heading">
                <div>
                  <h3>Transaction activity</h3>
                  <p>Illustrative hourly transaction volumes</p>
                </div>
                <span className="chart-tag"><Activity size={14} /> Sample data</span>
              </div>

              <div className="chart-legend">
                <span><i className="legend-dot legitimate-dot" /> Legitimate</span>
                <span><i className="legend-dot fraud-dot" /> Potential fraud</span>
              </div>

              <div className="main-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={chartData}
                    margin={{ top: 12, right: 8, left: -16, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="legitimateFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#7566ed" stopOpacity={0.25} />
                        <stop offset="100%" stopColor="#7566ed" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f05268" stopOpacity={0.22} />
                        <stop offset="100%" stopColor="#f05268" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#e9edf5" strokeDasharray="4 5" vertical={false} />
                    <XAxis
                      dataKey="time"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: '#8490a5', fontSize: 11 }}
                      dy={8}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: '#8490a5', fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 12,
                        border: '1px solid #e6eaf2',
                        boxShadow: '0 8px 30px #1d294010'
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="legitimate"
                      name="Legitimate"
                      stroke="#7566ed"
                      strokeWidth={2.5}
                      fill="url(#legitimateFill)"
                    />
                    <Area
                      type="monotone"
                      dataKey="fraud"
                      name="Potential fraud"
                      stroke="#f05268"
                      strokeWidth={2}
                      fill="url(#fraudFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </article>

            <article className="panel distribution-panel">
              <div className="panel-heading">
                <div>
                  <h3>Risk distribution</h3>
                  <p>Current local activity sample</p>
                </div>
                <button
                  className="icon-button"
                  onClick={() => setShowAllFeatures((value) => !value)}
                  aria-label="Show model information"
                >
                  <SlidersHorizontal size={18} />
                </button>
              </div>

              <div className="donut-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={distribution}
                      dataKey="value"
                      nameKey="name"
                      innerRadius="66%"
                      outerRadius="88%"
                      paddingAngle={4}
                      stroke="none"
                    >
                      {distribution.map((item) => (
                        <Cell key={item.name} fill={item.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center">
                  <ShieldCheck size={22} />
                  <strong>{activity.length}</strong>
                  <span>Local records</span>
                </div>
              </div>

              <div className="distribution-legend">
                {distribution.map((item) => (
                  <div className="distribution-item" key={item.name}>
                    <span className="distribution-label">
                      <i style={{ background: item.color }} />
                      {item.name}
                    </span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
              </div>
              <p className="chart-disclaimer">
                Illustrative initial records. This chart updates with predictions
                made in this browser session.
              </p>
            </article>
          </section>

          <section id="predictor" className="panel predictor-panel">
            <div className="panel-heading predictor-heading">
              <div>
                <div className="eyebrow"><Sparkles size={13} /> LIVE MODEL INFERENCE</div>
                <h3>Transaction risk assessment</h3>
                <p>
                  Submit all 30 numeric features required by your trained model.
                </p>
              </div>
              <span className={`model-badge ${modelLoaded ? 'model-ready' : ''}`}>
                <span className="status-dot" />
                {modelLoaded ? 'Model loaded' : 'Model status unconfirmed'}
              </span>
            </div>

            <form onSubmit={handlePredict}>
              <div className="transaction-summary">
                <div className="input-field amount-field">
                  <label htmlFor="Amount">Transaction amount</label>
                  <div className="amount-input-wrap">
                    <Wallet size={19} />
                    <span className="currency-symbol">₹</span>
                    <input
                      id="Amount"
                      name="Amount"
                      type="number"
                      min="0"
                      step="any"
                      value={transaction.Amount}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                </div>

                <div className="input-field">
                  <label htmlFor="Time">Transaction time (seconds)</label>
                  <input
                    id="Time"
                    name="Time"
                    type="number"
                    step="any"
                    value={transaction.Time}
                    onChange={handleInputChange}
                    required
                  />
                </div>
              </div>

              <div className="features-heading">
                <div>
                  <h4>Model feature inputs</h4>
                  <p>Enter the actual feature values for the transaction.</p>
                </div>
                <span>28 additional features</span>
              </div>

              <div className="features-grid">
                {Array.from({ length: 28 }, (_, index) => `V${index + 1}`).map(
                  (feature) => (
                    <div className="input-field" key={feature}>
                      <label htmlFor={feature}>{feature}</label>
                      <input
                        id={feature}
                        name={feature}
                        type="number"
                        step="any"
                        value={transaction[feature]}
                        onChange={handleInputChange}
                        required
                      />
                    </div>
                  )
                )}
              </div>

              {error && (
                <div className="error-message" role="alert">
                  <AlertTriangle size={18} />
                  <span>{error}</span>
                </div>
              )}

              {prediction && (
                <div
                  className={`prediction-result ${
                    prediction.prediction === 1 ? 'prediction-fraud' : 'prediction-safe'
                  }`}
                  role="status"
                >
                  <div className="prediction-result-icon">
                    {prediction.prediction === 1 ? (
                      <ShieldAlert size={27} />
                    ) : (
                      <ShieldCheck size={27} />
                    )}
                  </div>
                  <div className="prediction-result-copy">
                    <span>MODEL PREDICTION</span>
                    <h4>{prediction.label || (prediction.prediction === 1 ? 'Fraudulent' : 'Legitimate')}</h4>
                    <p>
                      Fraud probability: {
                        (prediction.fraud_probability * 100).toFixed(2)
                      }%
                    </p>
                  </div>
                  <div className="prediction-result-score">
                    <strong>
                      {(prediction.fraud_probability * 100).toFixed(1)}%
                    </strong>
                    <span>Fraud probability</span>
                  </div>
                </div>
              )}

              <div className="form-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={resetTransaction}
                >
                  Reset values
                </button>
                <button
                  type="submit"
                  className="primary-button"
                  disabled={predicting}
                >
                  {predicting ? (
                    <LoaderCircle size={18} className="spin" />
                  ) : (
                    <Zap size={18} />
                  )}
                  {predicting ? 'Analyzing transaction...' : 'Run fraud prediction'}
                </button>
              </div>
            </form>
          </section>

          <section id="transactions" className="panel transactions-panel">
            <div className="panel-heading transactions-heading">
              <div>
                <h3>Recent transaction assessments</h3>
                <p>Local demonstration records and predictions made in this session</p>
              </div>
              <button className="secondary-button export-button" onClick={exportTransactions}>
                <Download size={16} /> Export CSV
              </button>
            </div>

            <div className="table-controls">
              <div className="search-box">
                <Search size={17} />
                <input
                  type="search"
                  placeholder="Search transaction ID or type..."
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </div>

              <select
                aria-label="Filter transactions by status"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                <option value="All">All statuses</option>
                <option value="Legitimate">Legitimate</option>
                <option value="Review">Review</option>
                <option value="Fraudulent">Fraudulent</option>
              </select>
            </div>

            <div className="table-scroll">
              <table className="transaction-table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Transaction type</th>
                    <th>Amount</th>
                    <th>Time</th>
                    <th>Risk score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredActivity.map((item) => (
                    <tr key={item.id}>
                      <td><strong>{item.id}</strong></td>
                      <td>
                        <span className="transaction-type-icon">
                          <CreditCard size={16} />
                        </span>
                        {item.merchant}
                      </td>
                      <td className="amount-cell">{formatCurrency(item.amount)}</td>
                      <td><span className="time-cell"><Clock size={14} /> {item.time}</span></td>
                      <td>
                        <div className="risk-cell">
                          <div className="risk-bar">
                            <span
                              style={{
                                width: `${Math.min(item.score, 100)}%`,
                                background:
                                  item.score >= 70
                                    ? '#f05268'
                                    : item.score >= 40
                                      ? '#f5a623'
                                      : '#16b894'
                              }}
                            />
                          </div>
                          <span>{item.score}/100</span>
                        </div>
                      </td>
                      <td>
                        <span className={`status-badge ${item.status.toLowerCase()}`}>
                          <span />
                          {item.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {filteredActivity.length === 0 && (
                <div className="empty-state">
                  <Search size={24} />
                  <strong>No matching transactions</strong>
                  <span>Try a different search or status filter.</span>
                </div>
              )}
            </div>
          </section>

          <footer className="page-footer">
            <div><ShieldCheck size={16} /> FraudGuard AI <span>·</span> XGBoost fraud analysis</div>
            <span>Demo interface · Not a payment authorization system</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

function TargetIcon(props) {
  return <Fingerprint {...props} />
}

export default App
