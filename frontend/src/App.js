import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Activity, PieChart as ChartIcon, Search, BarChart2, Percent } from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  
  // Predict States 
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('Analyzing...');
  const [mainType, setMainType] = useState('Civil'); 
  const [subType, setSubType] = useState('Traffic Accident');   
  const [days, setDays] = useState(1);              
  const [rounds, setRounds] = useState(1);   
  const [people, setPeople] = useState(2);   

  const [stats, setStats] = useState({ total_cases: 0, success_rate: 0, avg_days: 0, chart_data: [] });
  const [cases, setCases] = useState([]);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [filterPrediction, setFilterPrediction] = useState('');
  const [filterMainType, setFilterMainType] = useState('');
  const [filterSubType, setFilterSubType] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  
  const [chartMode, setChartMode] = useState('count');

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetch('/api/stats')
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(err => console.error("Fetch stats failed", err));
      handleSearch();
    }
  }, [activeTab]);

  const handlePredict = async () => {
    if (!inputText.trim()) return alert("Please enter case details.");
    setLoading(true); setLoadingMsg('AI is reasoning...'); setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText, main_type: mainType, sub_type: subType,
          days: Number(days), rounds: Number(rounds), people: Number(people)
        }),
      });

      if (!response.ok) throw new Error(`Server Error (${response.status})`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Connection or reasoning failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    setIsSearching(true);
    try {
      const queryParams = new URLSearchParams({
        keyword: searchKeyword, prediction: filterPrediction,
        main_type: filterMainType, sub_type: filterSubType
      });
      const response = await fetch(`/api/cases?${queryParams}`);
      const data = await response.json();
      setCases(data);
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setIsSearching(false);
    }
  };

  const processChartData = () => {
    return (stats.chart_data || []).map(item => ({
      name: item.name,
      "Total Cases": item["Total Cases"] || 0,
      "Success Cases": item["Success Cases"] || 0,
      "Success Rate (%)": item["Total Cases"] > 0 ? Number(((item["Success Cases"] / item["Total Cases"]) * 100).toFixed(1)) : 0
    }));
  };
  const chartData = processChartData();
  const safeCases = Array.isArray(cases) ? cases : [];

  return (
    <>
      <div className="header-section">
        <h1>⚖️ AI Mediation Analysis System</h1>
        <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', gap: '15px' }}>
          <button onClick={() => setActiveTab('predict')} style={{ padding: '10px 20px', backgroundColor: activeTab === 'predict' ? '#007bff' : '#fff', color: activeTab === 'predict' ? '#fff' : '#007bff', border: '1px solid #007bff', borderRadius: '5px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18}/> AI Prediction
          </button>
          <button onClick={() => setActiveTab('dashboard')} style={{ padding: '10px 20px', backgroundColor: activeTab === 'dashboard' ? '#007bff' : '#fff', color: activeTab === 'dashboard' ? '#fff' : '#007bff', border: '1px solid #007bff', borderRadius: '5px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ChartIcon size={18}/> Data Dashboard
          </button>
        </div>
      </div>

      <div className="app-container">
        {activeTab === 'predict' && (
          <div>
            <div className="input-card">
              <div className="feature-form" style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', marginBottom: '20px', padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid #dee2e6' }}>
                
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Main Category</label>
                  <select value={mainType} onChange={(e) => setMainType(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ced4da' }}>
                    <option value="Civil">Civil</option>
                    <option value="Criminal">Criminal</option>
                  </select>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Sub Category</label>
                  <select value={subType} onChange={(e) => setSubType(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ced4da', width: '160px' }}>
                    <option value="Land & Building">Land & Building</option>
                    <option value="Disturbance of Peace">Disturbance of Peace</option>
                    <option value="Traffic Accident">Traffic Accident</option>
                    <option value="Burglary">Burglary</option>
                    <option value="Infringement">Infringement</option>
                    <option value="Labor Dispute">Labor Dispute</option>
                    <option value="Intellectual Property">Intellectual Property</option>
                    <option value="Fraud">Fraud</option>
                    <option value="Sales & Lease">Sales & Lease</option>
                    <option value="Defamation">Defamation</option>
                    <option value="Debt">Debt</option>
                    <option value="Inheritance">Inheritance</option>
                    <option value="Injury">Injury</option>
                    <option value="Medical Dispute">Medical Dispute</option>
                    <option value="Relationship Dispute">Relationship Dispute</option>
                    <option value="Alimony">Alimony</option>
                    <option value="Property Damage">Property Damage</option>
                    <option value="Harassment">Harassment</option>
                  </select>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Duration (Days)</label>
                  <input type="number" value={days} onChange={(e) => setDays(e.target.value)} min="0" style={{ padding: '8px', width: '80px', borderRadius: '4px', border: '1px solid #ced4da' }} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Rounds</label>
                  <input type="number" value={rounds} onChange={(e) => setRounds(e.target.value)} min="1" style={{ padding: '8px', width: '80px', borderRadius: '4px', border: '1px solid #ced4da' }} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Participants</label>
                  <input type="number" value={people} onChange={(e) => setPeople(e.target.value)} min="2" style={{ padding: '8px', width: '80px', borderRadius: '4px', border: '1px solid #ced4da' }} />
                </div>
              </div>

              <textarea className="case-textarea" value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="Please describe the details of the case..." />
              <button className="predict-btn" onClick={handlePredict} disabled={loading}>{loading ? loadingMsg : 'Analyze Now'}</button>
            </div>

            {result && (
              <div className="result-card">
                <div className="status-badge">🤖 AI Ensemble Reasoning (Saved to DB: ID {result.db_id})</div>
                <div className="main-info">
                  <div className="info-box"><label>Prediction</label><div className={`prediction ${result.prediction === 'Successful' ? 'status-ok' : 'status-no'}`}>{result.prediction}</div></div>
                  <div className="info-box"><label>Keywords</label><p>{Array.isArray(result.keywords) ? result.keywords.join(', ') : result.keywords}</p></div>
                </div>
                <hr />
                <div className="path-section"><h4>🛤️ Knowledge Graph Path</h4><div className="path-display">{result.paths && result.paths.length > 0 ? result.paths.map((p, i) => <div key={i} className="path-line">{p}</div>) : "None"}</div></div>
                <hr />
                <div className="explanation-section"><h4>💡 AI Explanation</h4><p className="explanation-text">{result.explanation}</p></div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div className="result-card" style={{ marginTop: '0', padding: '30px' }}>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '40px', gap: '20px' }}>
              <div style={{ flex: 1, textAlign: 'center', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: 0, color: '#6c757d', fontSize: '16px' }}>Total Cases</h3>
                <p style={{ margin: '10px 0 0', fontSize: '32px', fontWeight: 'bold', color: '#007bff' }}>{stats.total_cases}</p>
              </div>
              <div style={{ flex: 1, textAlign: 'center', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: 0, color: '#6c757d', fontSize: '16px' }}>Overall Success Rate</h3>
                <p style={{ margin: '10px 0 0', fontSize: '32px', fontWeight: 'bold', color: '#28a745' }}>{stats.success_rate} <span style={{fontSize:'14px'}}>%</span></p>
              </div>
              <div style={{ flex: 1, textAlign: 'center', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: 0, color: '#6c757d', fontSize: '16px' }}>Average Duration</h3>
                <p style={{ margin: '10px 0 0', fontSize: '32px', fontWeight: 'bold', color: '#ffc107' }}>{stats.avg_days} <span style={{fontSize:'14px'}}>Days</span></p>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '2px solid #eee', paddingBottom: '10px' }}>
              <h3 style={{ margin: 0 }}>📊 Category Statistics</h3>
              <div style={{ display: 'flex', gap: '5px', backgroundColor: '#e9ecef', padding: '4px', borderRadius: '6px' }}>
                <button onClick={() => setChartMode('count')} style={{ border: 'none', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', backgroundColor: chartMode === 'count' ? '#fff' : 'transparent', boxShadow: chartMode === 'count' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none', color: chartMode === 'count' ? '#333' : '#6c757d' }}>
                  <BarChart2 size={16}/> Count
                </button>
                <button onClick={() => setChartMode('percent')} style={{ border: 'none', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', backgroundColor: chartMode === 'percent' ? '#fff' : 'transparent', boxShadow: chartMode === 'percent' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none', color: chartMode === 'percent' ? '#333' : '#6c757d' }}>
                  <Percent size={16}/> Rate
                </button>
              </div>
            </div>
            
            <div style={{ width: '100%', height: 320, marginBottom: '40px' }}>
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{fill: '#666'}} />
                  <YAxis tick={{fill: '#666'}} />
                  <Tooltip cursor={{fill: '#f4f4f4'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
                  <Legend wrapperStyle={{paddingTop: '20px'}} />
                  
                  {chartMode === 'count' ? (
                    <>
                      <Bar dataKey="Total Cases" fill="#8884d8" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Success Cases" fill="#82ca9d" radius={[4, 4, 0, 0]} />
                    </>
                  ) : (
                    <Bar dataKey="Success Rate (%)" fill="#ffc658" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry["Success Rate (%)"] >= 50 ? "#82ca9d" : "#ffc658"} />
                      ))}
                    </Bar>
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <h3 style={{ marginBottom: '20px', borderBottom: '2px solid #eee', paddingBottom: '10px' }}>🔍 Case Database Search</h3>
            
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
              <input type="text" placeholder="Keywords..." value={searchKeyword} onChange={(e) => setSearchKeyword(e.target.value)} style={{ flex: '1 1 200px', padding: '10px', borderRadius: '4px', border: '1px solid #ced4da' }} />
              
              <select value={filterPrediction} onChange={(e) => setFilterPrediction(e.target.value)} style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ced4da', minWidth: '120px' }}>
                <option value="">All Results</option>
                <option value="Successful">Successful</option>
                <option value="Unsuccessful">Unsuccessful</option>
              </select>

              <select value={filterMainType} onChange={(e) => setFilterMainType(e.target.value)} style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ced4da', minWidth: '120px' }}>
                <option value="">All Main Cat.</option>
                <option value="Civil">Civil</option>
                <option value="Criminal">Criminal</option>
              </select>

              <select value={filterSubType} onChange={(e) => setFilterSubType(e.target.value)} style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ced4da', minWidth: '150px' }}>
                <option value="">All Sub Cat.</option>
                {/* 自動從統計資料庫中抽出有的分類 */}
                {(stats.chart_data || []).map((item, idx) => (
                  <option key={idx} value={item.name}>{item.name}</option>
                ))}
              </select>
              
              <button onClick={handleSearch} style={{ padding: '10px 20px', backgroundColor: '#343a40', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Search size={16}/> Search
              </button>
            </div>

            <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid #dee2e6' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                <thead style={{ backgroundColor: '#f8f9fa' }}>
                  <tr>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '60px' }}>ID</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '80px' }}>Main</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '120px' }}>Sub</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '50px' }}>Days</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '50px' }}>Rnds</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '50px' }}>Ppl</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '80px' }}>Result</th>
                    <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6' }}>Case Details & AI Inference</th>
                  </tr>
                </thead>
                <tbody>
                  {isSearching ? (
                    <tr><td colSpan="8" style={{ textAlign: 'center', padding: '20px' }}>Loading...</td></tr>
                  ) : safeCases.length === 0 ? (
                    <tr><td colSpan="8" style={{ textAlign: 'center', padding: '20px', color: '#6c757d' }}>No cases found.</td></tr>
                  ) : (
                    safeCases.map((c) => (
                      <tr key={c.id} style={{ borderBottom: '1px solid #eee' }}>
                        <td style={{ padding: '12px' }}>#{c.id}</td>
                        <td style={{ padding: '12px', fontWeight: 'bold', color: c.main_type === 'Criminal' ? '#d9534f' : '#5bc0de' }}>{c.main_type}</td>
                        <td style={{ padding: '12px' }}>{c.sub_type}</td>
                        <td style={{ padding: '12px' }}>{c.days}</td>
                        <td style={{ padding: '12px' }}>{c.rounds}</td>
                        <td style={{ padding: '12px' }}>{c.people}</td>
                        <td style={{ padding: '12px' }}>
                          <span style={{ padding: '4px 8px', borderRadius: '12px', fontSize: '12px', backgroundColor: c.prediction === 'Successful' ? '#d4edda' : '#f8d7da', color: c.prediction === 'Successful' ? '#155724' : '#721c24' }}>
                            {c.prediction}
                          </span>
                        </td>
                        <td style={{ padding: '12px', color: '#495057' }}>
                          <div style={{ marginBottom: '4px' }}>{c.content}</div>
                          <div style={{ fontSize: '12px', color: '#888' }}>🗝️ Keywords: {c.keywords}</div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

          </div>
        )}

      </div>
    </>
  );
}

export default App;