import React, { useState, useMemo, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const DISEASES = [
  { id: 'diabetes', name: 'Diabetes' },
  { id: 'heart', name: 'Heart Disease' },
  { id: 'kidney', name: 'Kidney Disease' },
  { id: 'parkinsons', name: "Parkinson's" }
];

const PARAMETER_DESCRIPTIONS = {
  // Diabetes
  Pregnancies: "Number of times pregnant.",
  Glucose: "Plasma glucose concentration a 2 hours in an oral glucose tolerance test.",
  BloodPressure: "Diastolic blood pressure (mm Hg).",
  SkinThickness: "Triceps skin fold thickness (mm).",
  Insulin: "2-Hour serum insulin (mu U/ml).",
  BMI: "Body mass index (weight in kg/(height in m)^2).",
  DiabetesPedigreeFunction: "Diabetes pedigree function (scores likelihood based on family history).",
  Age: "Patient's age in years.",
  // Heart Disease
  age: "Patient's age in years.",
  sex: "Sex (1 = male; 0 = female).",
  cp: "Chest pain type (0-3).",
  trestbps: "Resting blood pressure (mm Hg).",
  chol: "Serum cholesterol in mg/dl.",
  fbs: "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).",
  restecg: "Resting electrocardiographic results (0-2).",
  thalach: "Maximum heart rate achieved.",
  exang: "Exercise induced angina (1 = yes; 0 = no).",
  oldpeak: "ST depression induced by exercise relative to rest.",
  slope: "The slope of the peak exercise ST segment.",
  ca: "Number of major vessels (0-3) colored by flourosopy.",
  thal: "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect).",
  // Kidney Disease
  bp_diastolic: "Diastolic blood pressure.",
  bp_limit: "Blood pressure limit indicator.",
  sg: "Specific gravity of urine.",
  al: "Albumin level in urine.",
  rbc: "Red blood cells (1=Normal, 0=Abnormal).",
  su: "Sugar level in urine.",
  pc: "Pus cell level.",
  pcc: "Pus cell clumps presence.",
  ba: "Bacteria presence.",
  bgr: "Blood glucose random.",
  bu: "Blood urea level.",
  sod: "Sodium level.",
  sc: "Serum creatinine level.",
  pot: "Potassium level.",
  hemo: "Hemoglobin level.",
  pcv: "Packed cell volume.",
  rbcc: "Red blood cell count.",
  wbcc: "White blood cell count.",
  htn: "Hypertension presence (1=Yes, 0=No).",
  dm: "Diabetes Mellitus presence.",
  cad: "Coronary Artery Disease presence.",
  appet: "Appetite level (1=Good, 0=Poor).",
  pe: "Pedal Edema presence.",
  ane: "Anemia presence.",
  grf: "Glomerular Filtration Rate level.",
  stage: "Chronic Kidney Disease stage (1-5).",
  affected: "Affected side indicator.",
  // Parkinson's
  fo: "Average vocal fundamental frequency (Hz).",
  fhi: "Maximum vocal fundamental frequency (Hz).",
  flo: "Minimum vocal fundamental frequency (Hz).",
  jitter_percent: "MDVP jitter as a percentage.",
  jitter_abs: "MDVP absolute jitter in microseconds.",
  rap: "MDVP relative average perturbation.",
  ppq: "MDVP five-point period perturbation quotient.",
  ddp: "Average absolute difference of differences between jitter cycles.",
  shimmer: "MDVP local shimmer.",
  shimmer_db: "MDVP local shimmer in dB.",
  apq3: "Three-point amplitude perturbation quotient.",
  apq5: "Five-point amplitude perturbation quotient.",
  apq: "MDVP 11-point amplitude perturbation quotient.",
  dda: "Average absolute difference between consecutive shimmer cycles.",
  nhr: "Ratio of noise to tonal components in the voice.",
  hnr: "Ratio of harmonic to noise components in the voice.",
  rpde: "Recurrence period density entropy.",
  dfa: "Signal fractal scaling exponent.",
  spread1: "Nonlinear measure of fundamental frequency variation.",
  spread2: "Nonlinear measure of fundamental frequency variation.",
  d2: "Correlation dimension.",
  ppe: "Pitch period entropy."
};

const DISEASE_FIELDS = {
  diabetes: [
    { name: 'Pregnancies', label: 'Pregnancies', def: 1 },
    { name: 'Glucose', label: 'Glucose (mg/dL)', def: 120 },
    { name: 'BloodPressure', label: 'Blood Pressure (mmHg)', def: 70 },
    { name: 'SkinThickness', label: 'Skin Thickness (mm)', def: 20 },
    { name: 'Insulin', label: 'Insulin (mu U/ml)', def: 80 },
    { name: 'BMI', label: 'BMI', def: 25.0 },
    { name: 'DiabetesPedigreeFunction', label: 'Pedigree Function', def: 0.5 },
    { name: 'Age', label: 'Age', def: 30 }
  ],
  heart: [
    { name: 'age', label: 'Age', def: 50 },
    { name: 'sex', label: 'Sex (1=M, 0=F)', def: 1 },
    { name: 'cp', label: 'Chest Pain (0-3)', def: 0 },
    { name: 'trestbps', label: 'Resting BP', def: 120 },
    { name: 'chol', label: 'Cholesterol', def: 200 },
    { name: 'fbs', label: 'Fasting Sugar (1/0)', def: 0 },
    { name: 'restecg', label: 'Rest ECG (0-2)', def: 0 },
    { name: 'thalach', label: 'Max Heart Rate', def: 150 },
    { name: 'exang', label: 'Exercise Angina (1/0)', def: 0 },
    { name: 'oldpeak', label: 'ST Depression', def: 1.0 },
    { name: 'slope', label: 'Slope', def: 1 },
    { name: 'ca', label: 'Major Vessels (0-3)', def: 0 },
    { name: 'thal', label: 'Thalassemia', def: 1 }
  ],
  kidney: [
    { name: 'bp_diastolic', label: 'BP Diastolic', def: 80 },
    { name: 'bp_limit', label: 'BP Limit', def: 0 },
    { name: 'sg', label: 'Specific Gravity', def: 1.020 },
    { name: 'al', label: 'Albumin', def: 0 },
    { name: 'rbc', label: 'RBC (1=Normal, 0=Abnormal)', def: 1 },
    { name: 'su', label: 'Sugar', def: 0 },
    { name: 'pc', label: 'Pus Cell', def: 1 },
    { name: 'pcc', label: 'Pus Cell Clumps', def: 0 },
    { name: 'ba', label: 'Bacteria', def: 0 },
    { name: 'bgr', label: 'Blood Glucose Rand', def: 120 },
    { name: 'bu', label: 'Blood Urea', def: 40 },
    { name: 'sod', label: 'Sodium', def: 138 },
    { name: 'sc', label: 'Serum Creatinine', def: 1.2 },
    { name: 'pot', label: 'Potassium', def: 4.5 },
    { name: 'hemo', label: 'Hemoglobin', def: 13 },
    { name: 'pcv', label: 'Packed Cell Vol', def: 40 },
    { name: 'rbcc', label: 'RBC Count', def: 5.2 },
    { name: 'wbcc', label: 'WBC Count', def: 8000 },
    { name: 'htn', label: 'Hypertension (1/0)', def: 0 },
    { name: 'dm', label: 'Diabetes Mel (1/0)', def: 0 },
    { name: 'cad', label: 'Coronary Art (1/0)', def: 0 },
    { name: 'appet', label: 'Appetite (1=Good, 0=Poor)', def: 1 },
    { name: 'pe', label: 'Pedal Edema (1/0)', def: 0 },
    { name: 'ane', label: 'Anemia (1/0)', def: 0 },
    { name: 'grf', label: 'GFR Level', def: 60 },
    { name: 'stage', label: 'CKD Stage', def: 1 },
    { name: 'affected', label: 'Affected Side', def: 0 },
    { name: 'age', label: 'Age', def: 50 }
  ],
  parkinsons: [
    { name: 'fo', label: 'MDVP:Fo(Hz)', def: 120 },
    { name: 'fhi', label: 'MDVP:Fhi(Hz)', def: 150 },
    { name: 'flo', label: 'MDVP:Flo(Hz)', def: 100 },
    { name: 'jitter_percent', label: 'Jitter (%)', def: 0.005 },
    { name: 'jitter_abs', label: 'Jitter (Abs)', def: 0.00003 },
    { name: 'rap', label: 'MDVP:RAP', def: 0.003 },
    { name: 'ppq', label: 'MDVP:PPQ', def: 0.003 },
    { name: 'ddp', label: 'Jitter:DDP', def: 0.008 },
    { name: 'shimmer', label: 'Shimmer', def: 0.03 },
    { name: 'shimmer_db', label: 'Shimmer(dB)', def: 0.2 },
    { name: 'apq3', label: 'Shimmer:APQ3', def: 0.015 },
    { name: 'apq5', label: 'Shimmer:APQ5', def: 0.02 },
    { name: 'apq', label: 'MDVP:APQ', def: 0.025 },
    { name: 'dda', label: 'Shimmer:DDA', def: 0.045 },
    { name: 'nhr', label: 'NHR', def: 0.02 },
    { name: 'hnr', label: 'HNR', def: 20 },
    { name: 'rpde', label: 'RPDE', def: 0.4 },
    { name: 'dfa', label: 'DFA', def: 0.7 },
    { name: 'spread1', label: 'Spread1', def: -5.0 },
    { name: 'spread2', label: 'Spread2', def: 0.2 },
    { name: 'd2', label: 'D2', def: 2.1 },
    { name: 'ppe', label: 'PPE', def: 0.2 }
  ]
};

const App = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = useCallback((e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  }, []);

  const handleTabChange = useCallback((diseaseId) => {
    setActiveTab(diseaseId);
    setResult(null);
    if (diseaseId === 'home') {
      setFormData({});
    } else {
      const fields = DISEASE_FIELDS[diseaseId] || [];
      const defaults = {};
      fields.forEach(f => defaults[f.name] = f.def);
      setFormData(defaults);
    }
  }, []);

  const calculateRisk = async () => {
    setLoading(true);
    setResult(null);
    const startTime = performance.now();
    try {
      const response = await fetch(`${API_BASE}/predict/${activeTab}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      const endTime = performance.now();
      const networkLatency = endTime - startTime;
      
      if (response.ok) {
        setResult({ ...data, network_latency_ms: networkLatency });
      } else {
        console.error("Diagnostic Error:", data.detail);
        alert(`Diagnostic analysis failed: ${data.detail || 'Internal Server Error'}`);
      }
    } catch (error) {
      console.error("Network Error:", error);
      alert("Failed to connect to the diagnostic engine. Please ensure the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  const renderGauge = useMemo(() => (probability) => {
    const safeProb = (typeof probability === 'number' && !isNaN(probability)) ? probability : 0;
    const percentage = Math.round(safeProb * 100);
    const color = safeProb > 0.5 ? 'var(--vaen-red)' : 'var(--vaen-green)';
    const radius = 40;
    const centerX = 50;
    const centerY = 45;
    const angle = (safeProb * 180) - 180;
    const angleRad = (angle * Math.PI) / 180;
    const endX = centerX + radius * Math.cos(angleRad);
    const endY = centerY + radius * Math.sin(angleRad);
    
    return (
      <div className="gauge-container">
        <svg viewBox="0 0 100 60" width="100%">
          <path d="M 10 45 A 40 40 0 0 1 90 45" fill="none" stroke="#eee" strokeWidth="8" strokeLinecap="round" />
          <path d={`M 10 45 A 40 40 0 0 1 ${endX} ${endY}`} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round" style={{ transition: 'stroke-dasharray 0.5s ease' }} />
          <text x="50" y="40" textAnchor="middle" fontSize="12" fontWeight="800" fill="var(--text-primary)">{percentage}%</text>
          <text x="50" y="55" textAnchor="middle" fontSize="6" fontWeight="600" fill="var(--text-secondary)">RISK LEVEL</text>
        </svg>
      </div>
    );
  }, []);

  const renderHome = () => (
    <div className="container">
      <h1>AI Disease Predictor</h1>
      <p className="subtitle">Enterprise-grade diagnostic insights powered by machine learning.</p>
      <div className="card">
        <div className="card-title">Welcome to Vaen Health AI</div>
        <p>This platform uses advanced Random Forest and SVM models to evaluate clinical parameters and predict the risk of chronic diseases. Select a specialized diagnostic module from the navigation bar to begin.</p>
      </div>
      <div className="form-grid">
        <div className="card">
          <div className="card-title">Precision Metrics</div>
          <p>Our Kidney Disease model achieves **95.5%** accuracy, providing reliable insights for early detection.</p>
        </div>
        <div className="card">
          <div className="card-title">High Performance</div>
          <p>Optimized diagnostic engine delivers predictions in milliseconds, ensuring a seamless clinical experience.</p>
        </div>
      </div>
    </div>
  );

  const renderForm = (disease) => {
    const fields = DISEASE_FIELDS[disease] || [];
    return (
      <div className="container">
        <h1>{DISEASES.find(d => d.id === disease)?.name} Analysis</h1>
        <p className="subtitle">Fill in the clinical parameters to calculate the risk score.</p>
        <div className="form-grid">
          <div className="card" style={{ gridColumn: 'span 2' }}>
            <div className="card-title">Patient Parameters</div>
            <div className="form-grid">
              {fields.map(field => (
                <div className="input-group" key={field.name}>
                  <div className="label-container">
                    <label>{field.label}</label>
                    <div className="info-btn"><span>i</span></div>
                  </div>
                  <input type="number" name={field.name} value={formData[field.name] !== undefined ? formData[field.name] : ''} placeholder={field.def} onChange={handleInputChange} step="any" />
                  <div className="tooltip">
                    {PARAMETER_DESCRIPTIONS[field.name] || "Clinical parameter for diagnostic evaluation."}
                  </div>
                </div>
              ))}
            </div>
            <button className="calculate-btn" onClick={calculateRisk} disabled={loading}>{loading ? 'Analyzing...' : 'Predict'}</button>
          </div>
          <div className="card" style={{ gridColumn: 'span 2' }}>
            <div className="card-title">Diagnostic Result</div>
            {result ? (
              <div className="result-section">
                {renderGauge(result.risk_probability)}
                {result.prediction === 1 ? (
                  <div className="status-msg status-high">HIGH RISK DETECTED: Further clinical evaluation recommended.</div>
                ) : (
                  <div className="status-msg status-low">LOW RISK DETECTED: Parameters within normal range.</div>
                )}
                <div className="latency-info" style={{ marginTop: '1rem', fontSize: '0.7rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
                    Engine: {result.latency_ms?.toFixed(2)}ms | Network: {result.network_latency_ms?.toFixed(2)}ms
                </div>
              </div>
            ) : (
              <p style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>Please enter clinical data and click calculate.</p>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-logo" onClick={() => handleTabChange('home')}>
          <span className="logo-text">Vaen Health</span>
        </div>
        <div className="nav-links">
          {DISEASES.map(d => (
            <div 
              key={d.id} 
              className={`nav-link ${activeTab === d.id ? 'active' : ''}`}
              onClick={() => handleTabChange(d.id)}
            >
              <span className="nav-text">{d.name}</span>
            </div>
          ))}
        </div>
      </nav>
      <main className="main-content">
        {activeTab === 'home' ? renderHome() : renderForm(activeTab)}
      </main>
      <footer className="footer">
        <div className="footer-info">System Version: 2.1.1-optimized | Diagnostics Engine: AI-v4 | © 2026 Vaen Health</div>
      </footer>
    </div>
  );
};

export default App;
