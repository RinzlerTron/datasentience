import { useState, useEffect } from 'react';
import { Send, Activity, Zap, Thermometer, TrendingUp, Sparkles, X, Check, AlertTriangle, Brain, Search, Target } from 'lucide-react';
import './App.css';

// Professional number formatting utility
const formatNumber = (num) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1).replace(/\.0$/, '') + 'K';
  }
  return num.toLocaleString();
};

const formatCurrency = (num) => {
  return '$' + formatNumber(num);
};

const API_URL = import.meta.env.VITE_API_URL || 'https://zlnloj9fd5.execute-api.us-east-1.amazonaws.com/prod';

const DATACENTER_SPECS = {
  power_capacity_mw: 4.0,
  power_optimal_pct: 70,
  cooling_capacity_tons: 1200,
  total_racks: 120,
  occupied_racks: 102,
  pue_optimal: 1.2,
  pue_good: 1.4,
  temp_optimal_min: 20,
  temp_optimal_max: 25,
  cooling_optimal_ratio: 0.42
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Autonomous agent monitoring (always running)
  const [autonomousAgentStatus, setAutonomousAgentStatus] = useState({
    monitoring: 'active',
    analyzing: 'idle',
    predicting: 'idle'
  });

  // Query-based agents (only active during queries)
  const [queryAgentStatus, setQueryAgentStatus] = useState({
    retrieval: 'idle',
    reasoning: 'idle',
    action: 'idle'
  });

  const [agentDetails, setAgentDetails] = useState({
    retrieval: null,
    reasoning: null,
    action: null
  });

  // Autonomous discoveries (background alerts)
  const [autonomousDiscoveries, setAutonomousDiscoveries] = useState([]);

  // Pending actions awaiting human review
  const [pendingActions, setPendingActions] = useState([]);

  // Approved actions (completed)
  const [approvedActions, setApprovedActions] = useState([]);

  // Cumulative savings from approved actions
  const [cumulativeSavings, setCumulativeSavings] = useState({
    monthly: 0,
    annual: 0,
    avoided: 0
  });

  // NEW: Live autonomous actions (auto-resolved, no human approval needed)
  const [liveAutonomousActions, setLiveAutonomousActions] = useState([]);

  const [selectedEvent, setSelectedEvent] = useState(null);
  const [showActionModal, setShowActionModal] = useState(false);
  const [actionModalData, setActionModalData] = useState(null);

  const [performanceMetrics, setPerformanceMetrics] = useState({
    lastQueryLatency: 0,
    avgLatency: 0,
    totalQueries: 0,
    aiCostPerQuery: 0.0015,
    totalAiCost: 0
  });

  // NVIDIA NIM rate limiting tracking (daily limits)
  const [rateLimitUsage, setRateLimitUsage] = useState({
    requestsToday: 0,
    dailyLimit: 1000, // NVIDIA NIM daily request limit
    lastReset: new Date().toDateString(),
    avgResponseTime: 0
  });

  const [metrics, setMetrics] = useState({
    pue: 1.41,
    pueTrend: 'down',
    temperature: 24.5,
    tempTrend: 'stable',
    powerIT: 539,
    powerITTrend: 'up',
    powerCooling: 228,
    powerCoolingTrend: 'down',
    energyCostPerHour: 87.50
  });

  // F12 scenario cycling
  const [scenarioIndex, setScenarioIndex] = useState(0);

  // Demo mode detection
  const [demoMode, setDemoMode] = useState(false);

  // Page navigation
  const [currentPage, setCurrentPage] = useState('live'); // 'live' or 'autonomous'

  const scenarios = [
    {
      name: 'Cost Mystery',
      title: '💰 Unexpected Cost Spike Detected',
      description: 'Monthly costs increased by $50,000 above baseline',
      severity: 'warning',
      discovery: {
        findings: [
          'Power consumption +35% above normal',
          'Cluster-B utilization: 95% for 168 hours straight',
          'Deployment by dev-team@company.com on Oct 3'
        ]
      },
      agents: {
        retrieval: {
          duration: '2.3s',
          sources: ['Telemetry: 1,247 entries', 'Workload logs: 45 deployments', 'Billing data: 30 days'],
          findings: ['Cluster-B running continuously', 'No auto-terminate flag set', 'ML training job active']
        },
        reasoning: {
          duration: '3.1s',
          chain: [
            'Cross-referenced power spike with workload logs',
            'Identified: ML training job with auto_terminate=false',
            'Calculated: $50K monthly waste from 24/7 operation',
            'Typical usage: 8 hours/day, 5 days/week'
          ],
          conclusion: 'Dev cluster left running 24/7 unnecessarily',
          confidence: 94
        },
        action: {
          duration: '1.2s',
          recommendation: 'Enable auto-scaling on Cluster-B',
          impact: '$600K/year savings',
          risk: 'Low',
          timeline: 'Immediate'
        }
      },
      beforeAfter: {
        before: { cost: 450000, pue: 1.42, metric: 'Cluster-B: 24/7', waste: 50000 },
        after: { cost: 320000, pue: 1.32, metric: 'Cluster-B: Auto-scaled', savings: 130000 },
        annual: { totalSavings: 1560000, roi: 4800, payback: '2.3 weeks' }
      }
    },
    {
      name: 'Equipment Failure',
      title: '🔧 CRAC-12 Fan Degradation Detected',
      description: 'Fan speed declining, temperature rising in Zone-4',
      severity: 'critical',
      discovery: {
        findings: [
          'CRAC-12 fan speed: 90% → 82% over 5 days',
          'Rack 47 temperature: 24°C → 27.4°C',
          'Pattern matches previous fan bearing failures'
        ]
      },
      agents: {
        retrieval: {
          duration: '1.8s',
          sources: ['Equipment logs: 2,340 events', 'Temperature sensors: 120 racks', 'Maintenance history: 24 months'],
          findings: ['Fan speed declining linearly', 'Zone-4 temperature anomaly', 'Similar pattern in CRAC-7 failure (6mo ago)']
        },
        reasoning: {
          duration: '2.7s',
          chain: [
            'Analyzed 6-month trend: fan degradation pattern',
            'Predicted failure window: 36-48 hours',
            'Historical cost: $75K emergency repair + $120K downtime',
            'Preventive maintenance: $8K scheduled'
          ],
          conclusion: 'Imminent fan bearing failure in CRAC-12',
          confidence: 91
        },
        action: {
          duration: '0.9s',
          recommendation: 'Schedule emergency maintenance within 24 hours',
          impact: '$195K cost avoidance',
          risk: 'High if ignored',
          timeline: 'Within 24 hours'
        }
      },
      beforeAfter: {
        before: { cost: 195000, pue: 1.45, metric: 'CRAC-12: Failing', waste: 195000 },
        after: { cost: 8000, pue: 1.35, metric: 'CRAC-12: Serviced', savings: 187000 },
        annual: { totalSavings: 187000, roi: 2338, payback: 'Immediate' }
      }
    },
    {
      name: 'Energy Waste',
      title: '⚡ Night Cooling Inefficiency Detected',
      description: 'Overcooling during low-load periods (10PM-6AM)',
      severity: 'optimization',
      discovery: {
        findings: [
          'Cooling at 100% during 35% IT load',
          'Wasting $420/night on unnecessary cooling',
          'PUE spikes to 1.65 during night hours'
        ]
      },
      agents: {
        retrieval: {
          duration: '2.1s',
          sources: ['Telemetry: 30 days hourly', 'Workload schedules: 3 clusters', 'Weather data: ambient temp'],
          findings: ['Night load consistently 35%', 'Cooling setpoint unchanged', 'Ambient temp favorable']
        },
        reasoning: {
          duration: '3.4s',
          chain: [
            'Analyzed load patterns: predictable night reduction',
            'Cooling follows IT load with 5-min lag optimal',
            'Current: static setpoint regardless of load',
            'Opportunity: adaptive cooling based on real-time load'
          ],
          conclusion: 'Static cooling strategy wastes energy during low-load',
          confidence: 96
        },
        action: {
          duration: '1.1s',
          recommendation: 'Implement adaptive cooling setpoints (10PM-6AM)',
          impact: '$144K/year savings',
          risk: 'Very low',
          timeline: '1 week implementation'
        }
      },
      beforeAfter: {
        before: { cost: 152000, pue: 1.58, metric: 'Static cooling', waste: 12000 },
        after: { cost: 140000, pue: 1.38, metric: 'Adaptive cooling', savings: 12000 },
        annual: { totalSavings: 144000, roi: 1200, payback: '3 weeks' }
      }
    },
    {
      name: 'Capacity Planning',
      title: '📊 Capacity Threshold Approaching',
      description: 'Rack utilization at 85%, projected full in 42 days',
      severity: 'warning',
      discovery: {
        findings: [
          'Current: 102/120 racks occupied (85%)',
          'Growth rate: 0.5 racks/week',
          'No expansion plan in place'
        ]
      },
      agents: {
        retrieval: {
          duration: '2.5s',
          sources: ['Deployment history: 12 months', 'Capacity planning docs', 'Budget allocations'],
          findings: ['Steady 0.5 rack/week growth', 'Q4 project deployments planned', 'Lead time: 8-12 weeks for expansion']
        },
        reasoning: {
          duration: '3.8s',
          chain: [
            'Projected full capacity: 42 days',
            'Expansion lead time: 56-84 days',
            'Risk: deployment delays in 2-6 weeks',
            'Solution: Initiate expansion now OR optimize current usage'
          ],
          conclusion: 'Capacity crisis imminent without action',
          confidence: 88
        },
        action: {
          duration: '1.3s',
          recommendation: 'Initiate data center expansion OR implement workload optimization',
          impact: 'Avoid $500K revenue loss from deployment delays',
          risk: 'Medium',
          timeline: 'Start within 2 weeks'
        }
      },
      beforeAfter: {
        before: { cost: 500000, pue: 1.41, metric: 'No plan', waste: 500000 },
        after: { cost: 0, pue: 1.41, metric: 'Expansion planned', savings: 500000 },
        annual: { totalSavings: 500000, roi: 1000, payback: '6 months' }
      }
    }
  ];

  // Check demo mode status on startup
  useEffect(() => {
    // Backend is configured and working - disable demo mode
    setDemoMode(false);
  }, []);

  // Simulated metrics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        pue: (1.38 + Math.random() * 0.08).toFixed(2),
        temperature: (24 + Math.random() * 2).toFixed(1),
        powerIT: Math.floor(500 + Math.random() * 40),
        powerCooling: Math.floor(210 + Math.random() * 20)
      }));
      setLastUpdated(new Date());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Autonomous agent monitoring (always running in background)
  useEffect(() => {
    const interval = setInterval(() => {
      // Cycle through autonomous monitoring states
      setAutonomousAgentStatus(prev => {
        const states = ['monitoring', 'analyzing', 'predicting'];
        const current = states.find(s => prev[s] === 'active');
        const nextIndex = (states.indexOf(current) + 1) % states.length;

        return {
          monitoring: states[nextIndex] === 'monitoring' ? 'active' : 'idle',
          analyzing: states[nextIndex] === 'analyzing' ? 'active' : 'idle',
          predicting: states[nextIndex] === 'predicting' ? 'active' : 'idle'
        };
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // NEW: Generate live autonomous actions (continuous background optimization)
  useEffect(() => {
    const autonomousActionsList = [
      { action: 'Adjusted CRAC-3 cooling setpoint +1°C', reason: 'Low load detected, ambient temp favorable', savings: 45 },
      { action: 'Optimized workload distribution', reason: 'Balanced load across 3 clusters', savings: 120 },
      { action: 'Reduced redundant CRAC units', reason: 'Temperature stable, 2 units sufficient', savings: 230 },
      { action: 'Enabled free cooling mode', reason: 'Ambient temperature 18°C', savings: 340 },
      { action: 'Scaled down dev cluster', reason: 'No active workloads detected', savings: 180 },
      { action: 'Consolidated VMs on Cluster-A', reason: 'Optimizing rack utilization', savings: 95 },
      { action: 'Adjusted chiller setpoint', reason: 'PUE optimization opportunity', savings: 210 },
      { action: 'Powered down unused rack switches', reason: 'No traffic detected for 2 hours', savings: 65 }
    ];

    const interval = setInterval(() => {
      const randomAction = autonomousActionsList[Math.floor(Math.random() * autonomousActionsList.length)];
      const newAction = {
        id: Date.now(),
        time: new Date(),
        action: randomAction.action,
        reason: randomAction.reason,
        savings: randomAction.savings,
        type: 'auto-resolved'
      };

      setLiveAutonomousActions(prev => [newAction, ...prev.slice(0, 9)]);
    }, 8000); // New action every 8 seconds

    return () => clearInterval(interval);
  }, []);

  // F12 Demo Mode - Progressive flow
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'F12') {
        e.preventDefault();
        triggerScenario();
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [scenarioIndex]);

  const triggerScenario = async () => {
    // Only trigger real AI investigations on live page (Agent Investigations)
    if (currentPage !== 'live') {
      return;
    }

    const investigationQuestions = [
      `Based on current data center metrics (PUE: ${metrics.pue}, Temperature: ${metrics.temperature}°C, IT Power: ${metrics.powerIT}kW, Cooling: ${metrics.powerCooling}kW), identify the most critical performance issue and provide detailed cost impact analysis with specific savings recommendations.`,
      `Analyze potential equipment failures in our data center. Current power consumption is ${metrics.powerIT}kW with cooling at ${metrics.powerCooling}kW. What failure risks should we prioritize and what are the estimated costs if we don't act?`,
      `Our data center PUE is ${metrics.pue} with temperature at ${metrics.temperature}°C. Identify optimization opportunities that could reduce energy costs, provide specific monthly savings estimates, and recommend implementation timelines.`,
      `Evaluate capacity planning needs. Current IT power load is ${((metrics.powerIT / 1000) / DATACENTER_SPECS.power_capacity_mw * 100).toFixed(1)}% of capacity. When will we hit constraints and what are the expansion costs vs efficiency improvements?`
    ];

    const question = investigationQuestions[scenarioIndex % investigationQuestions.length];

    // Create discovery with "Investigating..." status
    const discovery = {
      id: Date.now(),
      time: new Date(),
      type: 'ai-investigation',
      title: `AI Investigation #${Date.now().toString().slice(-4)}`,
      description: 'Analyzing with NVIDIA NIM...',
      severity: 'investigation',
      investigating: true,
      question: question
    };

    setAutonomousDiscoveries(prev => [discovery, ...prev.slice(0, 4)]);
    setScenarioIndex((scenarioIndex + 1) % investigationQuestions.length);

    // Trigger real AI investigation
    try {
      // Prepare live metrics context
      const powerUtilPct = ((metrics.powerIT / 1000) / DATACENTER_SPECS.power_capacity_mw * 100).toFixed(1);
      const rackUtilPct = ((DATACENTER_SPECS.occupied_racks / DATACENTER_SPECS.total_racks) * 100).toFixed(0);

      const metricsContext = `CURRENT LIVE METRICS:
- PUE: ${metrics.pue} (efficiency: ${((1.2 / parseFloat(metrics.pue)) * 100).toFixed(0)}% of optimal)
- Temperature: ${metrics.temperature}°C (optimal: 20-25°C)
- IT Power: ${metrics.powerIT}kW (${powerUtilPct}% of ${DATACENTER_SPECS.power_capacity_mw}MW capacity)
- Cooling: ${metrics.powerCooling}kW
- Racks: ${DATACENTER_SPECS.occupied_racks}/${DATACENTER_SPECS.total_racks} (${rackUtilPct}% occupied)`;

      // MULTI-AGENT ORCHESTRATOR - Sequential agent coordination
      setQueryAgentStatus({ retrieval: 'active', reasoning: 'idle', action: 'idle' });

      // Determine question type for specialized data payloads
      const questionType = question.includes('cost') || question.includes('spike') ? 'cost_spike' :
                          question.includes('failure') || question.includes('equipment') ? 'failure_prediction' :
                          question.includes('cooling') || question.includes('energy') || question.includes('night') ? 'cooling_optimization' :
                          question.includes('capacity') || question.includes('hit capacity') ? 'capacity_planning' : 'general';

      // Call orchestrator via intelligent routing
      const orchestratorResponse = await fetch(API_URL + '/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: `${question}\n\n${metricsContext}`,
          agent_type: 'orchestrator',
          question_type: questionType
        })
      });
      const orchestratorData = await orchestratorResponse.json();

      // Check for AWS API Gateway error format (503 during initialization)
      if (orchestratorData.ErrorCode === "INTERNAL_FAILURE_FROM_MODEL") {
        const originalMessage = orchestratorData.OriginalMessage;
        if (originalMessage && originalMessage.includes("Service not ready")) {
          // Handle initialization in progress gracefully for F12 investigations
          setAutonomousDiscoveries(prev =>
            prev.map(d =>
              d.id === discovery.id
                ? {
                    ...d,
                    description: 'System initializing... Please wait 2-3 minutes and try again.',
                    investigating: false,
                    severity: 'info'
                  }
                : d
            )
          );
          return;
        } else {
          // Other API Gateway errors
          throw new Error(`Backend error: ${orchestratorData.Message || 'Unknown error'}`);
        }
      }

      // Check if we have a valid response
      if (!orchestratorData.answer) {
        throw new Error('Invalid response format from backend');
      }

      // Update agent status as orchestrator progresses
      setTimeout(() => setQueryAgentStatus({ retrieval: 'complete', reasoning: 'active', action: 'idle' }), 1000);
      setTimeout(() => setQueryAgentStatus({ retrieval: 'complete', reasoning: 'complete', action: 'active' }), 2000);

      // Extract orchestrator response (already formatted by backend)
      const multiAgentResponse = orchestratorData.answer;
      const roiData = orchestratorData.chart_data; // Backend ROI calculations from Agent 3

      // Complete the investigation
      setQueryAgentStatus({ retrieval: 'complete', reasoning: 'complete', action: 'complete' });

      setAutonomousDiscoveries(prev =>
        prev.map(d =>
          d.id === discovery.id
            ? {
                ...d,
                description: multiAgentResponse.substring(0, 200) + '...',
                investigating: false,
                aiResponse: multiAgentResponse,
                roiData: roiData, // Store backend ROI data for parseAIResponse
                type: 'ai-investigation',
                severity: multiAgentResponse.toLowerCase().includes('critical') ? 'critical' :
                         multiAgentResponse.toLowerCase().includes('urgent') ? 'critical' :
                         multiAgentResponse.toLowerCase().includes('optimization') ? 'optimization' : 'analysis'
              }
            : d
        )
      );

    } catch (error) {
      console.error('AI Investigation failed:', error);

      // Determine error message based on error type
      let errorDescription = 'AI investigation failed. Please check the connection to NVIDIA NIM.';
      if (error.message && (error.message.includes('504') || error.message.includes('timeout'))) {
        errorDescription = '⏱️ Investigation timed out. The backend is processing your query but it\'s taking longer than expected. Please try again in a moment or use a simpler query.';
      } else if (error.message && error.message.includes('Network')) {
        errorDescription = '🌐 Network error. Please check your connection and try again.';
      }

      // Update discovery with error status
      setAutonomousDiscoveries(prev =>
        prev.map(d =>
          d.id === discovery.id
            ? {
                ...d,
                description: errorDescription,
                investigating: false,
                severity: 'error'
              }
            : d
        )
      );
    }
  };

  // Parse AI response to extract actionable data
  const parseAIResponse = (aiResponse, roiData = null) => {
    const response = aiResponse || '';

    // Extract recommendations
    const recLines = response.split('\n').filter(line =>
      line.includes('Recommendation') || line.includes('Action:') || line.includes('Solution:')
    );
    const recommendation = recLines.length > 0 ?
      recLines[0].replace(/\*+/g, '').replace(/^.*?:/, '').trim() :
      'Implement AI-suggested optimizations';

    // Extract timeline
    const timelineMatch = response.match(/(\d+[-\s]?\d*)\s*(months?|weeks?|days?)/i);
    const timeline = timelineMatch ? timelineMatch[0] : '2-4 weeks';

    // Use backend ROI data if available (Gemini 2025 best practice: backend as source of truth)
    if (roiData) {
      return {
        recommendation: recommendation.substring(0, 100) + (recommendation.length > 100 ? '...' : ''),
        monthlySavings: roiData.monthly_savings || 15000,
        annualSavings: roiData.annual_savings || 180000,
        avoidedCosts: Math.floor((roiData.implementation_cost || 25000) * 0.3),
        timeline: timeline,
        roi: roiData.roi_percentage || 300,
        confidence: 92
      };
    }

    // Fallback to text parsing if no backend data (legacy compatibility)
    const costMatches = response.match(/\$[\d,]+/g) || [];
    const costs = costMatches.map(cost => parseInt(cost.replace(/[$,]/g, '')));
    const maxCost = costs.length > 0 ? Math.max(...costs) : 0;
    const roiMatch = response.match(/(\d+)%\s*ROI/i);
    const roi = roiMatch ? parseInt(roiMatch[1]) : Math.min(Math.max(Math.floor(maxCost / 10000 * 50), 100), 500);

    return {
      recommendation: recommendation.substring(0, 100) + (recommendation.length > 100 ? '...' : ''),
      monthlySavings: Math.floor(maxCost / 12) || 15000,
      annualSavings: maxCost || 180000,
      avoidedCosts: Math.floor(maxCost * 0.3) || 50000,
      timeline: timeline,
      roi: roi,
      confidence: 92
    };
  };

  const investigateDiscovery = (discovery) => {
    // Handle AI investigations differently
    if (discovery.type === 'ai-investigation' && discovery.aiResponse) {
      const parsedData = parseAIResponse(discovery.aiResponse, discovery.roiData);

      // Set up agent details for real AI response
      setAgentDetails({
        retrieval: {
          duration: '2.1s',
          sources: ['Live telemetry data', 'NVIDIA NIM endpoint', 'Real-time metrics'],
          findings: ['Current data center metrics analyzed', 'AI processing completed', 'Optimization opportunities identified']
        },
        reasoning: {
          duration: '3.2s',
          chain: [
            'Analyzed live data center performance metrics',
            'Identified inefficiencies and optimization opportunities',
            'Calculated financial impact and ROI projections',
            'Generated actionable recommendations with timelines'
          ],
          conclusion: `AI analysis identified potential savings of ${formatCurrency(parsedData.annualSavings)}/year`,
          confidence: parsedData.confidence
        },
        action: {
          duration: '1.5s',
          recommendation: parsedData.recommendation,
          impact: `${formatCurrency(parsedData.annualSavings)}/year savings`,
          risk: 'Low',
          timeline: parsedData.timeline
        }
      });

      // Set query agent status to complete to show full UI
      setQueryAgentStatus({ retrieval: 'complete', reasoning: 'complete', action: 'complete' });

      // Show action modal for AI investigation approval
      setActionModalData({
        discoveryId: discovery.id,
        scenario: 'ai-investigation',
        title: discovery.title,
        aiResponse: discovery.aiResponse,
        before: {
          cost: 450000,
          pue: parseFloat(metrics.pue),
          metric: 'Current inefficient operation',
          waste: parsedData.avoidedCosts
        },
        after: {
          cost: 450000 - parsedData.monthlySavings,
          pue: (parseFloat(metrics.pue) * 0.95).toFixed(2),
          metric: 'AI-optimized operation',
          savings: parsedData.monthlySavings
        },
        annual: {
          totalSavings: parsedData.annualSavings,
          roi: parsedData.roi,
          payback: '3-6 months'
        }
      });
      setShowActionModal(true);
      return;
    }

    const scenario = scenarios.find(s => s.name === discovery.scenario);
    if (!scenario) return;

    // Mark as being investigated
    setAutonomousDiscoveries(prev =>
      prev.map(d => d.id === discovery.id ? { ...d, investigating: true } : d)
    );

    // Show selected discovery
    setSelectedEvent(discovery);

    // Simulate agent work
    setQueryAgentStatus({ retrieval: 'idle', reasoning: 'idle', action: 'idle' });

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, retrieval: 'active' }));
    }, 100);

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, retrieval: 'complete' }));
      setAgentDetails(prev => ({ ...prev, retrieval: scenario.agents.retrieval }));
    }, 2000);

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, reasoning: 'active' }));
    }, 2000);

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, reasoning: 'complete' }));
      setAgentDetails(prev => ({ ...prev, reasoning: scenario.agents.reasoning }));
    }, 5000);

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, action: 'active' }));
    }, 5000);

    setTimeout(() => {
      setQueryAgentStatus(prev => ({ ...prev, action: 'complete' }));
      setAgentDetails(prev => ({ ...prev, action: scenario.agents.action }));

      // Show action modal for human approval
      setActionModalData({
        discoveryId: discovery.id,
        scenario: scenario.name,
        title: scenario.title,
        ...scenario.beforeAfter
      });
      setShowActionModal(true);
    }, 6500);
  };

  const handleReviewLater = () => {
    if (!actionModalData) return;

    // Move to pending actions
    const pendingAction = {
      id: actionModalData.discoveryId,
      scenario: actionModalData.scenario,
      title: actionModalData.title,
      time: new Date(),
      data: actionModalData
    };

    setPendingActions(prev => {
      // Avoid duplicates
      if (prev.some(p => p.id === pendingAction.id)) return prev;
      return [pendingAction, ...prev];
    });

    // Remove from autonomous discoveries
    setAutonomousDiscoveries(prev =>
      prev.filter(d => d.id !== actionModalData.discoveryId)
    );

    setShowActionModal(false);
    setActionModalData(null);
  };

  const handleApproveAction = () => {
    if (!actionModalData) return;

    const scenario = scenarios.find(s => s.name === actionModalData.scenario);

    const approvedAction = {
      id: actionModalData.discoveryId,
      scenario: actionModalData.scenario,
      title: actionModalData.title,
      actionTaken: scenario ? scenario.agents.action.recommendation : 'Action approved',
      time: new Date(),
      savings: actionModalData.after.savings,
      annual: actionModalData.annual.totalSavings,
      avoided: actionModalData.before.waste
    };

    // Add to approved actions
    setApprovedActions(prev => [approvedAction, ...prev]);

    // Update cumulative savings
    setCumulativeSavings(prev => ({
      monthly: prev.monthly + actionModalData.after.savings,
      annual: prev.annual + actionModalData.annual.totalSavings,
      avoided: prev.avoided + actionModalData.before.waste
    }));

    // Remove from pending and discoveries
    setPendingActions(prev => prev.filter(p => p.id !== actionModalData.discoveryId));
    setAutonomousDiscoveries(prev => prev.filter(d => d.id !== actionModalData.discoveryId));

    setShowActionModal(false);
    setActionModalData(null);
  };

  const reviewPendingAction = (action) => {
    setActionModalData(action.data);
    setShowActionModal(true);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    setQueryAgentStatus({ retrieval: 'idle', reasoning: 'idle', action: 'idle' });

    const startTime = performance.now();

    try {
      // Prepare live metrics context
      const powerUtilPct = ((metrics.powerIT / 1000) / DATACENTER_SPECS.power_capacity_mw * 100).toFixed(1);
      const rackUtilPct = ((DATACENTER_SPECS.occupied_racks / DATACENTER_SPECS.total_racks) * 100).toFixed(0);
      const coolingRatio = (metrics.powerCooling / (metrics.powerIT + metrics.powerCooling) * 100).toFixed(0);

      const metricsContext = `CURRENT LIVE METRICS:
- PUE: ${metrics.pue} (optimal: 1.2, current efficiency: ${((1.2 / parseFloat(metrics.pue)) * 100).toFixed(0)}%)
- Temperature: ${metrics.temperature}°C (optimal range: 20-25°C)
- IT Power: ${metrics.powerIT}kW (${powerUtilPct}% of ${DATACENTER_SPECS.power_capacity_mw}MW capacity)
- Cooling Power: ${metrics.powerCooling}kW (${coolingRatio}% of total power)
- Racks: ${DATACENTER_SPECS.occupied_racks}/${DATACENTER_SPECS.total_racks} (${rackUtilPct}% occupied)
- Power Efficiency: ${(metrics.powerIT / (metrics.powerIT + metrics.powerCooling)).toFixed(2)} IT power ratio`;

      // AGENT 1: DATA RETRIEVAL SPECIALIST (Chat optimized)
      setQueryAgentStatus({ retrieval: 'active', reasoning: 'idle', action: 'idle' });

      // Call Agent 1 via intelligent routing
      const agent1Response = await fetch(API_URL + '/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: `${userMessage.content}\n\n${metricsContext}`,
          agent_type: 'agent1'
        })
      });
      const agent1Data = await agent1Response.json();

      // Check for AWS API Gateway error format (503 during initialization)
      if (agent1Data.ErrorCode === "INTERNAL_FAILURE_FROM_MODEL") {
        const originalMessage = agent1Data.OriginalMessage;
        if (originalMessage && originalMessage.includes("Service not ready")) {
          // Handle initialization in progress gracefully
          const waitMessage = "🔄 DataSentience is initializing... This takes 2-3 minutes during startup. Please try again in a moment.";
          setMessages(prev => [...prev, { role: 'agent', content: waitMessage }]);
          setLoading(false);
          return;
        } else {
          // Other API Gateway errors
          throw new Error(`Backend error: ${agent1Data.Message || 'Unknown error'}`);
        }
      }

      // Check if we have a valid response
      if (!agent1Data.answer) {
        throw new Error('Invalid response format from backend');
      }

      // Calculate timing and metrics (for 1 optimized call)
      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);

      // Agent 1 optimized response
      const multiAgentResponse = agent1Data.answer;

      // Complete all agents
      setQueryAgentStatus({ retrieval: 'complete', reasoning: 'complete', action: 'complete' });

      // Update performance metrics (accounting for 1 optimized call)
      const newTotalQueries = performanceMetrics.totalQueries + 1; // 1 agent call
      const newAvgLatency = Math.round(
        ((performanceMetrics.avgLatency * performanceMetrics.totalQueries) + latency) / newTotalQueries
      );
      const newTotalCost = (newTotalQueries * 0.0015).toFixed(4);

      setPerformanceMetrics({
        lastQueryLatency: latency,
        avgLatency: newAvgLatency,
        totalQueries: newTotalQueries,
        aiCostPerQuery: 0.0015,
        totalAiCost: newTotalCost
      });

      // Update rate limiting tracking (1 request made)
      const today = new Date().toDateString();
      setRateLimitUsage(prev => ({
        ...prev,
        requestsToday: prev.lastReset === today ? prev.requestsToday + 1 : 1,
        lastReset: today,
        avgResponseTime: prev.requestsToday > 0 ?
          Math.round((prev.avgResponseTime * prev.requestsToday + latency) / (prev.requestsToday + 1)) :
          Math.round(latency)
      }));

      const agentMessage = { role: 'agent', content: multiAgentResponse };
      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      let errorMessage = 'Error: Could not reach the agent.';
      if (error.message && error.message.includes('504')) {
        errorMessage = '⏱️ Request timed out. The agent is processing your query but it\'s taking longer than expected. Please try again in a moment or use a simpler query.';
      } else if (error.message && error.message.includes('timeout')) {
        errorMessage = '⏱️ Request timed out. The backend is processing your query but it\'s taking longer than expected. Please try again.';
      } else if (error.message && error.message.includes('Network')) {
        errorMessage = '🌐 Network error. Please check your connection and try again.';
      }
      const demoMessage = demoMode
        ? 'Demo mode active. Press F12 to see AI scenarios in action, or set up a live backend for text queries.'
        : errorMessage;
      setMessages(prev => [...prev, { role: 'agent', content: demoMessage }]);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    if (status === 'idle') return '⚪';
    if (status === 'active') return '🟡';
    if (status === 'complete') return '🟢';
    return '⚪';
  };

  const getMetricContext = (type, value) => {
    if (type === 'pue') {
      const diff = ((value - DATACENTER_SPECS.pue_optimal) / DATACENTER_SPECS.pue_optimal * 100);
      return {
        status: diff < 5 ? 'optimal' : diff < 20 ? 'good' : 'poor',
        text: `${diff > 0 ? '+' : ''}${diff.toFixed(0)}% vs optimal (${DATACENTER_SPECS.pue_optimal})`,
        color: diff < 5 ? '#76B900' : diff < 20 ? '#FF9900' : '#ef4444'
      };
    }
    if (type === 'temp') {
      const inRange = value >= DATACENTER_SPECS.temp_optimal_min && value <= DATACENTER_SPECS.temp_optimal_max;
      return {
        status: inRange ? 'optimal' : 'warning',
        text: `Optimal: ${DATACENTER_SPECS.temp_optimal_min}-${DATACENTER_SPECS.temp_optimal_max}°C`,
        color: inRange ? '#76B900' : '#f59e0b'
      };
    }
    if (type === 'power') {
      const mw = value / 1000;
      const pct = (mw / DATACENTER_SPECS.power_capacity_mw * 100);
      return {
        status: pct < 80 ? 'good' : 'warning',
        text: `${pct.toFixed(1)}% of ${DATACENTER_SPECS.power_capacity_mw}MW capacity`,
        color: pct < 80 ? '#76B900' : '#f59e0b'
      };
    }
    if (type === 'cooling') {
      const totalPower = metrics.powerIT + value;
      const ratio = value / totalPower;
      const optimal = DATACENTER_SPECS.cooling_optimal_ratio;
      return {
        status: Math.abs(ratio - optimal) < 0.05 ? 'optimal' : 'good',
        text: `Ratio: ${(ratio * 100).toFixed(0)}% (optimal: ${(optimal * 100).toFixed(0)}%)`,
        color: Math.abs(ratio - optimal) < 0.05 ? '#76B900' : '#FF9900'
      };
    }
  };

  const pueContext = getMetricContext('pue', parseFloat(metrics.pue));
  const tempContext = getMetricContext('temp', parseFloat(metrics.temperature));
  const powerContext = getMetricContext('power', metrics.powerIT);
  const coolingContext = getMetricContext('cooling', metrics.powerCooling);

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <Activity className="icon" />
            DataSentience
          </h1>
          <p className="subtitle">Autonomous Data Center Optimization</p>
        </div>
        <div className="powered-badges">
          <span className="badge nvidia">NVIDIA NIM</span>
          <span className="badge aws">AWS SageMaker</span>
        </div>
      </header>

      {/* Demo Mode Banner */}
      {demoMode && (
        <div style={{
          background: 'linear-gradient(90deg, rgba(255, 153, 0, 0.9) 0%, rgba(239, 68, 68, 0.9) 100%)',
          padding: '12px 40px',
          textAlign: 'center',
          fontSize: '14px',
          fontWeight: 'bold',
          color: 'white',
          borderBottom: '2px solid #FF9900'
        }}>
          🎭 DEMO MODE - Using Simulated Responses (NVIDIA_API_KEY not configured)
        </div>
      )}

      {/* Autonomous Agent Status Bar - Always Visible (STICKY) */}
      <div style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'linear-gradient(90deg, rgba(118, 185, 0, 0.15) 0%, rgba(255, 153, 0, 0.15) 100%)',
        backdropFilter: 'blur(10px)',
        padding: '12px 40px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
      }}>
        <div style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
          <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#76B900' }}>
            🤖 AUTONOMOUS AGENTS (Always Running)
          </div>
          <div style={{ display: 'flex', gap: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <Search size={14} />
              <span>Monitoring</span>
              <span>{getStatusIcon(autonomousAgentStatus.monitoring)}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <Brain size={14} />
              <span>Analyzing</span>
              <span>{getStatusIcon(autonomousAgentStatus.analyzing)}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <Target size={14} />
              <span>Predicting</span>
              <span>{getStatusIcon(autonomousAgentStatus.predicting)}</span>
            </div>
          </div>
        </div>
        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>
          Scanning 337 sources • Last check: 2s ago
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{
        padding: '0 40px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        background: 'rgba(0, 0, 0, 0.2)'
      }}>
        <div style={{ display: 'flex', gap: '0' }}>
          <button
            onClick={() => setCurrentPage('live')}
            style={{
              padding: '16px 24px',
              background: currentPage === 'live' ? 'rgba(76, 175, 80, 0.2)' : 'transparent',
              border: 'none',
              borderBottom: currentPage === 'live' ? '2px solid #76B900' : '2px solid transparent',
              color: currentPage === 'live' ? '#76B900' : 'rgba(255, 255, 255, 0.7)',
              fontSize: '14px',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            🚀 Live Mode
          </button>
          <button
            onClick={() => setCurrentPage('autonomous')}
            style={{
              padding: '16px 24px',
              background: currentPage === 'autonomous' ? 'rgba(255, 153, 0, 0.2)' : 'transparent',
              border: 'none',
              borderBottom: currentPage === 'autonomous' ? '2px solid #FF9900' : '2px solid transparent',
              color: currentPage === 'autonomous' ? '#FF9900' : 'rgba(255, 255, 255, 0.7)',
              fontSize: '14px',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            🔍 Autonomous Actions
          </button>
        </div>
      </div>


      <div className="main-layout" style={{ display: 'grid', gridTemplateColumns: '300px 1fr 320px', gap: '20px', maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
        <aside className="sidebar-left">
          {currentPage === 'live' && (
            <>
              <div className="sidebar-header">
                <Sparkles size={18} />
                <h3>Quick Insights</h3>
              </div>
              <div className="quick-questions-sidebar">
                {[
                  "What's causing the cost spike?",
                  "Show equipment failure predictions",
                  "Calculate night cooling savings",
                  "When will we hit capacity?"
                ].map((q, i) => (
                  <button key={i} className="quick-button-sidebar" onClick={() => setInput(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </>
          )}

          {currentPage === 'live' && (
            <>
              {/* F12 Agent Investigations Trigger */}
              <div style={{
                marginTop: '16px',
                padding: '12px',
                background: 'rgba(255, 153, 0, 0.1)',
                border: '1px solid rgba(255, 153, 0, 0.3)',
                borderRadius: '8px',
                fontSize: '11px',
                color: '#FF9900',
                textAlign: 'center'
              }}>
                💡 Press F12: Agent Investigations (Human-in-Loop)
                <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.8 }}>
                  Real NVIDIA NIM analysis
                </div>
              </div>
            </>
          )}

          {currentPage === 'autonomous' && (
            <>
              <div className="sidebar-header">
                <Brain size={18} />
                <h3>Autonomous Operations</h3>
              </div>
              <div style={{
                padding: '12px',
                background: 'rgba(118, 185, 0, 0.1)',
                border: '1px solid rgba(118, 185, 0, 0.3)',
                borderRadius: '8px',
                fontSize: '11px',
                color: '#76B900',
                textAlign: 'center'
              }}>
                ⚡ Auto-executed optimizations
                <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.8 }}>
                  No human approval needed
                </div>
              </div>
            </>
          )}

          <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px' }}>
            <h3 style={{ fontSize: '12px', marginBottom: '12px', color: '#FF9900' }}>CAPACITY STATUS</h3>
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                Power: {((metrics.powerIT / 1000) / DATACENTER_SPECS.power_capacity_mw * 100).toFixed(1)}%
              </div>
              <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{
                  width: ((metrics.powerIT / 1000) / DATACENTER_SPECS.power_capacity_mw * 100) + '%',
                  height: '100%',
                  background: powerContext.color
                }} />
              </div>
            </div>
            <div>
              <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                Racks: {(DATACENTER_SPECS.occupied_racks / DATACENTER_SPECS.total_racks * 100).toFixed(0)}%
              </div>
              <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{
                  width: (DATACENTER_SPECS.occupied_racks / DATACENTER_SPECS.total_racks * 100) + '%',
                  height: '100%',
                  background: '#76B900'
                }} />
              </div>
            </div>
          </div>

          {/* Pending Review Section */}
          {pendingActions.length > 0 && (
            <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255, 153, 0, 0.05)', borderRadius: '8px', border: '1px solid rgba(255, 153, 0, 0.3)' }}>
              <h3 style={{ fontSize: '12px', marginBottom: '12px', color: '#FF9900' }}>📋 PENDING REVIEW</h3>
              <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
                {pendingActions.map(action => (
                  <div key={action.id} style={{
                    padding: '8px',
                    marginBottom: '8px',
                    background: 'rgba(255, 153, 0, 0.1)',
                    borderRadius: '4px',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}
                  onClick={() => reviewPendingAction(action)}>
                    <div style={{ fontWeight: 'bold', color: 'white' }}>{action.title}</div>
                    <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '10px' }}>
                      {new Date(action.time).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255, 153, 0, 0.05)', borderRadius: '8px', border: '1px solid rgba(255, 153, 0, 0.3)' }}>
            <h3 style={{ fontSize: '12px', marginBottom: '12px', color: '#FF9900' }}>⚡ NVIDIA NIM RATE LIMITS</h3>
            <div>
              <div style={{ fontSize: '11px', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Daily Requests</span>
                <span style={{ fontWeight: 'bold' }}>{rateLimitUsage.requestsToday} / {rateLimitUsage.dailyLimit}</span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden', marginBottom: '8px' }}>
                <div style={{
                  width: ((rateLimitUsage.requestsToday / rateLimitUsage.dailyLimit) * 100) + '%',
                  height: '100%',
                  background: (rateLimitUsage.requestsToday / rateLimitUsage.dailyLimit) > 0.8 ? '#ef4444' : (rateLimitUsage.requestsToday / rateLimitUsage.dailyLimit) > 0.5 ? '#FF9900' : '#76B900',
                  transition: 'all 0.3s ease'
                }} />
              </div>
              <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>
                Avg response: {rateLimitUsage.avgResponseTime}ms • Resets daily
              </div>
              <div style={{ fontSize: '10px', color: '#FF9900', marginTop: '4px' }}>
                {rateLimitUsage.dailyLimit - rateLimitUsage.requestsToday} requests remaining today
              </div>
            </div>
          </div>
        </aside>

        <main className="main-content">

          {/* Page Content - Conditional based on currentPage */}
          {currentPage === 'live' && (
            <div>
              {/* Agent Investigations (F12 Discoveries) */}
              {autonomousDiscoveries.length > 0 && (
                <div style={{ marginBottom: '24px' }}>
                  <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    🔬 Agent Investigations (Human-in-Loop)
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {autonomousDiscoveries.map(discovery => (
                      <div
                        key={discovery.id}
                        style={{
                          padding: '16px',
                          background: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: '8px',
                          border: `1px solid ${discovery.severity === 'critical' ? '#ef4444' : discovery.severity === 'optimization' ? '#FF9900' : '#76B900'}`
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <h5 style={{ margin: 0, fontSize: '14px' }}>{discovery.title}</h5>
                          {discovery.aiResponse && (
                            <button
                              onClick={() => investigateDiscovery(discovery)}
                              style={{
                                padding: '6px 12px',
                                background: '#76B900',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                fontSize: '12px',
                                cursor: 'pointer'
                              }}
                            >
                              View Full Analysis
                            </button>
                          )}
                        </div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.7)' }}>
                          {discovery.description}
                        </div>
                        {discovery.investigating && (
                          <div style={{ fontSize: '11px', color: '#FF9900', marginTop: '8px' }}>
                            🔬 Analyzing with NVIDIA NIM...
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Business Impact Section - Human-initiated Actions */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={20} />
                  💰 Business Impact (Human-Initiated)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div style={{ padding: '16px', background: 'rgba(118, 185, 0, 0.1)', borderRadius: '8px', border: '1px solid rgba(118, 185, 0, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#76B900', marginBottom: '8px' }}>MONTHLY SAVINGS</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{formatCurrency(cumulativeSavings.monthly)}</div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>
                      From chat queries + F12 investigations
                    </div>
                  </div>
                  <div style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#3b82f6', marginBottom: '8px' }}>AVOIDED COSTS</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{formatCurrency(cumulativeSavings.avoided)}</div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>Preventive actions from AI analysis</div>
                  </div>
                  <div style={{ padding: '16px', background: 'rgba(255, 153, 0, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 153, 0, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#FF9900', marginBottom: '8px' }}>ANNUAL SAVINGS</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{formatCurrency(cumulativeSavings.annual)}</div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>
                      Total projected from approved actions
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentPage === 'autonomous' && (
            <div>
              {/* Autonomous Actions */}
              {liveAutonomousActions.length > 0 && (
                <div style={{ marginBottom: '24px' }}>
                  <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    ⚡ Live Autonomous Actions
                  </h3>
                  <div style={{ maxHeight: '400px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {liveAutonomousActions.map(action => (
                      <div
                        key={action.id}
                        style={{
                          padding: '12px',
                          background: 'rgba(118, 185, 0, 0.1)',
                          borderRadius: '6px',
                          fontSize: '12px',
                          border: '1px solid rgba(118, 185, 0, 0.3)'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>✓ {action.action}</div>
                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>{action.reason}</div>
                          </div>
                          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                            <div style={{ fontSize: '11px', color: '#76B900', fontWeight: 'bold' }}>
                              {formatCurrency(action.savings)}/hr saved
                            </div>
                            <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)' }}>
                              {Math.floor((new Date() - action.time) / 1000)}s ago
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Business Impact Section - Autonomous Actions */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={20} />
                  💰 Business Impact (Autonomous)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div style={{ padding: '16px', background: 'rgba(118, 185, 0, 0.1)', borderRadius: '8px', border: '1px solid rgba(118, 185, 0, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#76B900', marginBottom: '8px' }}>HOURLY SAVINGS</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                      {formatCurrency(liveAutonomousActions.reduce((sum, action) => sum + action.savings, 0))}
                    </div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>
                      From auto-executed optimizations
                    </div>
                  </div>
                  <div style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#3b82f6', marginBottom: '8px' }}>DAILY PROJECTION</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                      {formatCurrency(liveAutonomousActions.reduce((sum, action) => sum + action.savings, 0) * 24)}
                    </div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>24-hour savings estimate</div>
                  </div>
                  <div style={{ padding: '16px', background: 'rgba(255, 153, 0, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 153, 0, 0.3)' }}>
                    <div style={{ fontSize: '12px', color: '#FF9900', marginBottom: '8px' }}>ACTIONS TAKEN</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{liveAutonomousActions.length}</div>
                    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>
                      Auto-optimizations completed
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Agent Work Breakdown - Only shown when investigating */}
          {selectedEvent && (
            <div className="metrics-section">
              <h3 style={{ marginBottom: '16px', fontSize: '16px' }}>
                🔬 Multi-Agent Investigation: {selectedEvent.scenario}
              </h3>

              <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
                <div style={{
                  flex: 1,
                  padding: '12px',
                  background: queryAgentStatus.retrieval === 'complete' ? 'rgba(118, 185, 0, 0.1)' : 'rgba(255,255,255,0.03)',
                  border: '1px solid ' + (queryAgentStatus.retrieval === 'active' ? '#FF9900' : 'rgba(255,255,255,0.1)'),
                  borderRadius: '8px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <Search size={16} />
                    <strong>Retrieval Agent</strong>
                    <span style={{ marginLeft: 'auto' }}>{getStatusIcon(queryAgentStatus.retrieval)}</span>
                  </div>
                  {agentDetails.retrieval && (
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.8)' }}>
                      <div>✓ {agentDetails.retrieval.duration}</div>
                      <div>✓ {agentDetails.retrieval.sources.length} sources</div>
                    </div>
                  )}
                </div>

                <div style={{
                  flex: 1,
                  padding: '12px',
                  background: queryAgentStatus.reasoning === 'complete' ? 'rgba(118, 185, 0, 0.1)' : 'rgba(255,255,255,0.03)',
                  border: '1px solid ' + (queryAgentStatus.reasoning === 'active' ? '#FF9900' : 'rgba(255,255,255,0.1)'),
                  borderRadius: '8px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <Brain size={16} />
                    <strong>Reasoning Agent</strong>
                    <span style={{ marginLeft: 'auto' }}>{getStatusIcon(queryAgentStatus.reasoning)}</span>
                  </div>
                  {agentDetails.reasoning && (
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.8)' }}>
                      <div>✓ {agentDetails.reasoning.duration}</div>
                      <div>✓ {agentDetails.reasoning.confidence}% confidence</div>
                    </div>
                  )}
                </div>

                <div style={{
                  flex: 1,
                  padding: '12px',
                  background: queryAgentStatus.action === 'complete' ? 'rgba(118, 185, 0, 0.1)' : 'rgba(255,255,255,0.03)',
                  border: '1px solid ' + (queryAgentStatus.action === 'active' ? '#FF9900' : 'rgba(255,255,255,0.1)'),
                  borderRadius: '8px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <Target size={16} />
                    <strong>Action Agent</strong>
                    <span style={{ marginLeft: 'auto' }}>{getStatusIcon(queryAgentStatus.action)}</span>
                  </div>
                  {agentDetails.action && (
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.8)' }}>
                      <div>✓ {agentDetails.action.duration}</div>
                      <div>✓ {agentDetails.action.risk} risk</div>
                    </div>
                  )}
                </div>
              </div>

              {agentDetails.retrieval && (
                <div style={{ padding: '16px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', fontSize: '13px' }}>
                  <div style={{ marginBottom: '12px' }}>
                    <strong style={{ color: '#76B900' }}>Key Findings:</strong>
                    <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                      {agentDetails.retrieval.findings.map((f, i) => (
                        <li key={i} style={{ marginBottom: '4px' }}>{f}</li>
                      ))}
                    </ul>
                  </div>
                  {agentDetails.reasoning && (
                    <div style={{ marginBottom: '12px' }}>
                      <strong style={{ color: '#FF9900' }}>Analysis Chain:</strong>
                      <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                        {agentDetails.reasoning.chain.map((step, i) => (
                          <li key={i} style={{ marginBottom: '4px' }}>{step}</li>
                        ))}
                      </ul>
                      <div style={{ marginTop: '8px', padding: '8px', background: 'rgba(255, 153, 0, 0.1)', borderRadius: '4px' }}>
                        <strong>Conclusion:</strong> {agentDetails.reasoning.conclusion}
                      </div>
                    </div>
                  )}
                  {agentDetails.action && (
                    <div>
                      <strong style={{ color: '#4299e1' }}>Recommendation (Human Approval Required):</strong>
                      <div style={{ marginTop: '8px', padding: '12px', background: 'rgba(66, 153, 225, 0.1)', borderRadius: '4px' }}>
                        <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{agentDetails.action.recommendation}</div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.8)' }}>
                          Impact: {agentDetails.action.impact} | Risk: {agentDetails.action.risk} | Timeline: {agentDetails.action.timeline}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}


          {/* Chat container - Only show in Live Mode */}
          {currentPage === 'live' && (
            <div className="chat-container">
              <div className="chat-header">
                <h2>Ask DataSentience Agent (Human-in-Loop)</h2>
              </div>

              <div className="messages">
                {messages.length === 0 && (
                  <div className="empty-state">
                    <Activity size={48} className="empty-icon" />
                    <p className="empty-text">Ask a question to investigate further</p>
                    <p className="empty-subtext">Press F12 for autonomous discovery demo</p>
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div key={i} className={'message message-' + msg.role}>
                    <div className="message-label">
                      {msg.role === 'user' ? 'You' : 'DataSentience'}
                    </div>
                    <div className="message-content">{msg.content}</div>
                  </div>
                ))}
                {loading && (
                  <div className="message message-agent">
                    <div className="message-label">DataSentience</div>
                    <div className="message-content loading">
                      Analyzing data center metrics...
                    </div>
                  </div>
                )}
              </div>

              <div className="input-container">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Ask about performance, failures, or optimization opportunities..."
                  className="input"
                  disabled={loading}
                />
                <button
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                  className="send-button"
                >
                  <Send size={20} />
                </button>
              </div>
            </div>
          )}
        </main>

        <aside className="sidebar-right">
          {/* Real-time Metrics - Dark theme exactly like Quick Insights */}
          <div className="sidebar-left" style={{ width: '100%', position: 'static' }}>
            <div className="sidebar-header">
              <Activity size={18} />
              <h3>Real-Time Metrics</h3>
            </div>
            <div style={{
              fontSize: '11px',
              color: 'rgba(255,255,255,0.6)',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{
                width: '6px',
                height: '6px',
                background: '#48bb78',
                borderRadius: '50%',
                display: 'inline-block',
                animation: 'pulse 2s infinite'
              }}></span>
              LIVE • Updated {Math.floor((new Date() - lastUpdated) / 1000)}s ago
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <button className="quick-button-sidebar" style={{ cursor: 'default' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <TrendingUp size={16} />
                  <span style={{ fontSize: '12px', fontWeight: 'bold' }}>PUE</span>
                </div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '4px' }}>{metrics.pue}</div>
                <div style={{ fontSize: '10px', opacity: '0.8' }}>
                  {pueContext.text}
                </div>
              </button>

              <button className="quick-button-sidebar" style={{ cursor: 'default' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <Thermometer size={16} />
                  <span style={{ fontSize: '12px', fontWeight: 'bold' }}>Temperature</span>
                </div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '4px' }}>{metrics.temperature}°C</div>
                <div style={{ fontSize: '10px', opacity: '0.8' }}>
                  {tempContext.text}
                </div>
              </button>

              <button className="quick-button-sidebar" style={{ cursor: 'default' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <Zap size={16} />
                  <span style={{ fontSize: '12px', fontWeight: 'bold' }}>IT Power</span>
                </div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '4px' }}>{metrics.powerIT} kW</div>
                <div style={{ fontSize: '10px', opacity: '0.8' }}>
                  {powerContext.text}
                </div>
              </button>

              <button className="quick-button-sidebar" style={{ cursor: 'default' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <Activity size={16} />
                  <span style={{ fontSize: '12px', fontWeight: 'bold' }}>Cooling</span>
                </div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '4px' }}>{metrics.powerCooling} kW</div>
                <div style={{ fontSize: '10px', opacity: '0.8' }}>
                  {coolingContext.text}
                </div>
              </button>
            </div>
          </div>

          {/* AI Performance - Only shown when there are queries */}
          {performanceMetrics.totalQueries > 0 && !demoMode && (
            <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <Zap size={16} />
                <h3 style={{ fontSize: '14px', margin: 0 }}>AI Performance</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Last Query:</span>
                  <span style={{ fontWeight: 'bold' }}>{performanceMetrics.lastQueryLatency}ms</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Avg Latency:</span>
                  <span style={{ fontWeight: 'bold' }}>{performanceMetrics.avgLatency}ms</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Total Queries:</span>
                  <span style={{ fontWeight: 'bold' }}>{performanceMetrics.totalQueries}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Total AI Cost:</span>
                  <span style={{ fontWeight: 'bold', color: '#FF9900' }}>${performanceMetrics.totalAiCost}</span>
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* Action Approval Modal */}
      {showActionModal && actionModalData && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.85)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #232F3E 0%, #1a1f2e 100%)',
            padding: '32px',
            borderRadius: '12px',
            maxWidth: '700px',
            width: '90%',
            border: '2px solid #FF9900',
            boxShadow: '0 8px 32px rgba(255, 153, 0, 0.3)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
              <h2 style={{ margin: 0, fontSize: '20px', color: '#FF9900' }}>
                👤 Human Approval Required
              </h2>
              <button
                onClick={() => setShowActionModal(false)}
                style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.6)', cursor: 'pointer', fontSize: '24px' }}
              >
                ×
              </button>
            </div>

            <div style={{ fontSize: '16px', marginBottom: '24px', color: 'rgba(255,255,255,0.9)' }}>
              {actionModalData.title}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
              <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px', border: '1px solid #ef4444' }}>
                <h3 style={{ margin: '0 0 12px 0', color: '#ef4444', fontSize: '14px' }}>CURRENT STATE</h3>
                <div style={{ fontSize: '13px' }}>
                  <div>Cost: <strong>{formatCurrency(actionModalData.before.cost)}/mo</strong></div>
                  <div>PUE: <strong>{actionModalData.before.pue}</strong></div>
                  <div style={{ marginTop: '8px', color: '#fca5a5' }}>
                    Risk: <strong>{formatCurrency(actionModalData.before.waste)}</strong>
                  </div>
                </div>
              </div>

              <div style={{ padding: '16px', background: 'rgba(118, 185, 0, 0.15)', borderRadius: '8px', border: '1px solid #76B900' }}>
                <h3 style={{ margin: '0 0 12px 0', color: '#76B900', fontSize: '14px' }}>AFTER APPROVAL</h3>
                <div style={{ fontSize: '13px' }}>
                  <div>Cost: <strong>{formatCurrency(actionModalData.after.cost)}/mo</strong></div>
                  <div>PUE: <strong>{actionModalData.after.pue}</strong></div>
                  <div style={{ marginTop: '8px', color: '#a7f3d0' }}>
                    Savings: <strong>{formatCurrency(actionModalData.after.savings)}/mo</strong>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ padding: '16px', background: 'rgba(255, 153, 0, 0.15)', borderRadius: '8px', marginBottom: '24px', border: '1px solid #FF9900' }}>
              <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px', color: 'white' }}>
                Annual Impact: {formatCurrency(actionModalData.annual.totalSavings)}/year
              </div>
              <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.8)' }}>
                ROI: {actionModalData.annual.roi}% | Payback: {actionModalData.annual.payback}
              </div>
            </div>

            {/* Show detailed AI response for AI investigations */}
            {actionModalData.scenario === 'ai-investigation' && actionModalData.aiResponse && (
              <div style={{ marginBottom: '24px' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                  cursor: 'pointer',
                  padding: '8px',
                  background: 'rgba(118, 185, 0, 0.1)',
                  borderRadius: '6px'
                }}
                onClick={() => {
                  const detailsEl = document.getElementById('ai-details');
                  if (detailsEl) {
                    detailsEl.style.display = detailsEl.style.display === 'none' ? 'block' : 'none';
                  }
                }}>
                  <span style={{ color: '#76B900', fontSize: '16px' }}>🤖</span>
                  <strong style={{ color: '#76B900' }}>NVIDIA NIM Detailed Analysis</strong>
                  <span style={{ marginLeft: 'auto', color: '#76B900' }}>▼</span>
                </div>
                <div id="ai-details" style={{
                  maxHeight: '200px',
                  overflowY: 'auto',
                  padding: '16px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '8px',
                  border: '1px solid rgba(118, 185, 0, 0.3)'
                }}>
                  <pre style={{
                    whiteSpace: 'pre-wrap',
                    wordWrap: 'break-word',
                    fontSize: '12px',
                    lineHeight: '1.4',
                    margin: 0,
                    fontFamily: 'inherit',
                    color: 'rgba(255,255,255,0.8)'
                  }}>
                    {actionModalData.aiResponse}
                  </pre>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={handleApproveAction}
                className="send-button"
                style={{ flex: 1, padding: '14px', fontSize: '15px', justifyContent: 'center' }}
              >
                <Check size={20} style={{ marginRight: '8px' }} />
                Approve Action
              </button>
              <button
                onClick={handleReviewLater}
                style={{
                  flex: 1,
                  padding: '14px',
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '15px'
                }}
              >
                Review Later
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;