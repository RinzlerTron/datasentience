"""True Multi-Agent System for DataSentience.

This module implements a genuine multi-agent architecture with:
- Agent 1: Data Retrieval Specialist (vector DB, telemetry analysis)
- Agent 2: Reasoning Specialist (cost engine, trend analysis)
- Agent 3: Action Planning Specialist (ROI calculator, implementation planner)
- Orchestrator: Coordinates agent workflows and handoffs
"""

import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.config import config
from src.vector_store import optimized_vector_store

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    agent_id: str
    findings: Dict[str, Any]
    execution_time: float
    confidence: float
    metadata: Dict[str, Any]


class BaseAgent:
    """Base agent with common functionality."""

    def __init__(self, agent_id: str, specialization: str):
        self.agent_id = agent_id
        self.specialization = specialization
        self.api_key = config.NVIDIA_API_KEY
        self.api_url = config.NVIDIA_API_URL
        self.model = config.MODEL_NAME

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=5))
    async def _call_nvidia_nim(self, prompt: str, max_tokens: int = 800) -> str:
        """Call NVIDIA NIM API with retry logic."""
        call_start = time.time()
        logger.info(f"NVIDIA_API[{self.agent_id}]: Starting API call, max_tokens={max_tokens}")

        if not self.api_key:
            logger.warning(f"NVIDIA_API[{self.agent_id}]: No API key - returning demo mode")
            return "Demo mode - NVIDIA API key not configured"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"You are {self.specialization}. Be concise and specific."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }

        try:
            # Extended timeout for SageMaker environment - 60 seconds
            logger.info(f"NVIDIA_API[{self.agent_id}]: Creating httpx client with 60s timeout")
            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.info(f"NVIDIA_API[{self.agent_id}]: Sending POST to {self.api_url}")
                response = await client.post(self.api_url, headers=headers, json=payload)

                response_time = time.time() - call_start
                logger.info(f"NVIDIA_API[{self.agent_id}]: Received response in {response_time:.2f}s, status={response.status_code}")

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                logger.info(f"NVIDIA_API[{self.agent_id}]: Success - received {len(content)} chars")
                return content

        except httpx.TimeoutException as e:
            elapsed = time.time() - call_start
            logger.error(f"NVIDIA_API[{self.agent_id}]: TIMEOUT after {elapsed:.2f}s - {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            elapsed = time.time() - call_start
            logger.error(f"NVIDIA_API[{self.agent_id}]: HTTP error after {elapsed:.2f}s - status={e.response.status_code}")
            logger.error(f"NVIDIA_API[{self.agent_id}]: Response body: {e.response.text[:500]}")
            raise
        except Exception as e:
            elapsed = time.time() - call_start
            logger.error(f"NVIDIA_API[{self.agent_id}]: Unexpected error after {elapsed:.2f}s - {type(e).__name__}: {str(e)}")
            raise


class DataRetrievalAgent(BaseAgent):
    """Agent 1: Specialized in data extraction, pattern recognition, and anomaly detection."""

    def __init__(self):
        super().__init__("agent_1", "Data Retrieval Specialist")
        # Initialize specialized tools
        self.vector_store = optimized_vector_store
        self.anomaly_threshold = 2.0  # Standard deviations

    def quick_query(self, question_type: str, current_metrics: Dict, user_question: str) -> AgentResponse:
        """Handle quick queries with specialized data payloads."""
        start_time = time.time()

        # Build specialized payload based on question type
        data_payload = self._build_specialized_payload(question_type, current_metrics)

        # Generate analysis using specialized data
        analysis = self._analyze_with_payload(question_type, data_payload, user_question)

        execution_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            findings=analysis,
            execution_time=execution_time,
            confidence=0.85,
            metadata={"question_type": question_type, "payload_size": len(str(data_payload))}
        )

    async def process_chat_query(self, user_question: str) -> Dict:
        """Process chat queries with concise, specific responses for quick insights."""
        start_time = time.time()

        # Get current metrics (mock data for chat context)
        current_metrics = {
            "pue": 1.41,
            "temperature": 24.5,
            "powerIT": 539,
            "powerCooling": 228,
            "occupied_racks": 102,
            "total_racks": 120
        }

        # Detect key metrics and anomalies for context
        anomalies = self._detect_anomalies(current_metrics)
        cooling_ratio = current_metrics["powerCooling"] / current_metrics["powerIT"]

        # Generate concise analysis using optimized prompt
        prompt = f"""Answer this data center question in 2 sentences: {user_question}

CURRENT STATE: PUE {current_metrics['pue']}, Cooling {current_metrics['powerCooling']}kW, IT {current_metrics['powerIT']}kW, Ratio {cooling_ratio:.2f}

REQUIREMENTS:
- Maximum 2 sentences
- Include specific savings calculation with numbers
- No explanations, just direct answer with $ amount or % savings"""

        analysis = await self._call_nvidia_nim(prompt, max_tokens=200)
        execution_time = time.time() - start_time

        return {
            "answer": analysis,
            "data_payload": {
                "metrics_used": current_metrics,
                "anomalies_detected": len(anomalies),
                "response_time": f"{execution_time:.2f}s"
            }
        }

    async def extract_patterns(self, context: str, current_metrics: Dict) -> AgentResponse:
        """Extract data patterns for full investigation."""
        start_time = time.time()

        # Perform vector search for similar patterns
        search_results = self.vector_store.search(context, top_k=5)

        # Analyze current metrics for anomalies
        anomalies = self._detect_anomalies(current_metrics)

        # Extract patterns using NVIDIA NIM
        prompt = f"""Analyze these data center metrics and identify key patterns:\n\nCURRENT METRICS: {json.dumps(current_metrics, indent=2)}\n\nHISTORICAL SIMILAR PATTERNS: {json.dumps([r.get('text', '')[:200] for r in search_results], indent=2)}\n\nDETECTED ANOMALIES: {json.dumps(anomalies, indent=2)}\n\nYour task as Data Retrieval Specialist:\n1. Identify key data patterns in current metrics\n2. Correlate with historical patterns\n3. Highlight significant anomalies\n4. Focus ONLY on data extraction and pattern identification\n\nFormat response as: "RETRIEVAL FINDINGS: [your analysis]" """

        analysis = await self._call_nvidia_nim(prompt, max_tokens=400)

        execution_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            findings={
                "patterns": analysis,
                "anomalies": anomalies,
                "historical_matches": len(search_results),
                "raw_search_results": search_results
            },
            execution_time=execution_time,
            confidence=0.9,
            metadata={"search_results": len(search_results)}
        )

    def _build_specialized_payload(self, question_type: str, current_metrics: Dict) -> Dict:
        """Build question-specific data payload."""
        base_metrics = current_metrics.copy()

        if question_type == "cost_spike":
            return {
                **base_metrics,
                "energy_cost_breakdown": {
                    "it_power_costs": self._calculate_hourly_costs(current_metrics.get("powerIT", 520)),
                    "cooling_power_costs": self._calculate_hourly_costs(current_metrics.get("powerCooling", 214)),
                    "facility_overhead": self._calculate_hourly_costs(50),
                    "total_hourly_cost": self._calculate_hourly_costs(784)
                },
                "efficiency_correlations": {
                    "pue_cost_impact": (float(current_metrics.get("pue", 1.41)) - 1.2) * 100,
                    "temperature_efficiency": self._calculate_temp_efficiency(current_metrics.get("temperature", 25.3)),
                    "cooling_ratio": current_metrics.get("powerCooling", 214) / current_metrics.get("powerIT", 520)
                },
                "cost_anomaly_score": self._calculate_cost_anomaly(current_metrics)
            }

        elif question_type == "failure_prediction":
            return {
                **base_metrics,
                "equipment_health": {
                    "thermal_stress": self._calculate_thermal_stress(current_metrics),
                    "power_stress": self._calculate_power_stress(current_metrics),
                    "utilization_stress": self._calculate_utilization_stress(current_metrics)
                },
                "failure_indicators": {
                    "temperature_deviation": abs(current_metrics.get("temperature", 25.3) - 22.5),
                    "pue_degradation": max(0, float(current_metrics.get("pue", 1.41)) - 1.2),
                    "cooling_inefficiency": max(0, current_metrics.get("powerCooling", 214) / current_metrics.get("powerIT", 520) - 0.3)
                },
                "mtbf_estimates": self._calculate_mtbf(current_metrics)
            }

        elif question_type == "cooling_savings":
            return {
                **base_metrics,
                "cooling_optimization": {
                    "current_efficiency": current_metrics.get("powerCooling", 214) / current_metrics.get("powerIT", 520),
                    "optimal_efficiency": 0.3,
                    "potential_savings": max(0, (current_metrics.get("powerCooling", 214) / current_metrics.get("powerIT", 520) - 0.3) * current_metrics.get("powerIT", 520)),
                    "night_cooling_potential": self._calculate_night_cooling_savings(current_metrics)
                },
                "environmental_factors": {
                    "current_temp": current_metrics.get("temperature", 25.3),
                    "optimal_temp": 22.5,
                    "cooling_load_factor": min(1.0, current_metrics.get("temperature", 25.3) / 22.5)
                }
            }

        elif question_type == "capacity_planning":
            return {
                **base_metrics,
                "capacity_analysis": {
                    "current_power_utilization": (current_metrics.get("powerIT", 520) / 1000) / 4.0,  # 4MW total
                    "current_rack_utilization": 102 / 120,  # Current/total racks
                    "growth_constraint": "power" if (current_metrics.get("powerIT", 520) / 1000) / 4.0 > 102/120 else "space",
                    "time_to_constraint": self._calculate_time_to_constraint(current_metrics)
                },
                "growth_projections": {
                    "monthly_power_growth": 2.5,  # Percentage
                    "monthly_space_growth": 1.8,  # Percentage
                    "seasonal_factors": {"summer": 1.2, "winter": 0.9}
                }
            }

        return base_metrics

    def _analyze_with_payload(self, question_type: str, payload: Dict, user_question: str) -> Dict:
        """Analyze using specialized payload."""
        prompt = f"""As a Data Retrieval Specialist, analyze this specialized data for: {user_question}

SPECIALIZED DATA PAYLOAD:
{json.dumps(payload, indent=2)}

Your analysis should focus on:
1. Key data patterns relevant to the question
2. Anomalies or concerning metrics
3. Data correlations and trends
4. Specific numerical insights

Be concise and focus only on data extraction and pattern identification.
Format: "RETRIEVAL FINDINGS: [analysis]" """

        analysis = self._call_nvidia_nim(prompt, max_tokens=500)

        return {
            "analysis": analysis,
            "payload_type": question_type,
            "key_metrics": self._extract_key_metrics(payload),
            "anomalies_detected": self._count_anomalies(payload)
        }

    def _detect_anomalies(self, metrics: Dict) -> List[Dict]:
        """Detect anomalies in current metrics."""
        anomalies = []

        # PUE anomaly
        pue = float(metrics.get("pue", 1.41))
        if pue > 1.6:
            anomalies.append({"type": "pue_high", "value": pue, "threshold": 1.6, "severity": "high"})
        elif pue > 1.4:
            anomalies.append({"type": "pue_elevated", "value": pue, "threshold": 1.4, "severity": "medium"})

        # Temperature anomaly
        temp = float(metrics.get("temperature", 25.3))
        if temp > 26:
            anomalies.append({"type": "temperature_high", "value": temp, "threshold": 26, "severity": "high"})
        elif temp < 20:
            anomalies.append({"type": "temperature_low", "value": temp, "threshold": 20, "severity": "medium"})

        # Cooling ratio anomaly
        cooling_ratio = metrics.get("powerCooling", 214) / metrics.get("powerIT", 520)
        if cooling_ratio > 0.5:
            anomalies.append({"type": "cooling_inefficient", "value": cooling_ratio, "threshold": 0.5, "severity": "high"})

        return anomalies

    def _calculate_hourly_costs(self, power_kw: float) -> float:
        """Calculate hourly costs for given power."""
        return power_kw * 0.12  # $0.12 per kWh

    def _calculate_temp_efficiency(self, temp: float) -> float:
        """Calculate temperature efficiency factor."""
        optimal_temp = 22.5
        return max(0.7, 1.0 - abs(temp - optimal_temp) / 10)

    def _calculate_cost_anomaly(self, metrics: Dict) -> float:
        """Calculate cost anomaly score."""
        pue_penalty = max(0, float(metrics.get("pue", 1.41)) - 1.2) * 100
        temp_penalty = max(0, abs(metrics.get("temperature", 25.3) - 22.5) - 2) * 10
        return pue_penalty + temp_penalty

    def _calculate_thermal_stress(self, metrics: Dict) -> float:
        """Calculate thermal stress indicator."""
        temp = metrics.get("temperature", 25.3)
        return max(0, (temp - 20) / 15)  # 0-1 scale

    def _calculate_power_stress(self, metrics: Dict) -> float:
        """Calculate power stress indicator."""
        utilization = (metrics.get("powerIT", 520) / 1000) / 4.0  # 4MW capacity
        return min(1.0, utilization / 0.8)  # Stress starts at 80% utilization

    def _calculate_utilization_stress(self, metrics: Dict) -> float:
        """Calculate utilization stress."""
        rack_util = 102 / 120  # Current racks / total racks
        return min(1.0, rack_util / 0.9)  # Stress starts at 90%

    def _calculate_mtbf(self, metrics: Dict) -> Dict:
        """Calculate Mean Time Between Failure estimates."""
        base_mtbf = 50000  # Hours

        # Reduce MTBF based on stress factors
        thermal_factor = 1.0 - self._calculate_thermal_stress(metrics) * 0.3
        power_factor = 1.0 - self._calculate_power_stress(metrics) * 0.2

        adjusted_mtbf = base_mtbf * thermal_factor * power_factor

        return {
            "estimated_mtbf_hours": int(adjusted_mtbf),
            "thermal_impact": 1.0 - thermal_factor,
            "power_impact": 1.0 - power_factor
        }

    def _calculate_night_cooling_savings(self, metrics: Dict) -> Dict:
        """Calculate potential night cooling savings."""
        current_cooling = metrics.get("powerCooling", 214)
        night_efficiency_gain = 0.2  # 20% more efficient at night

        return {
            "potential_savings_kw": current_cooling * night_efficiency_gain,
            "hourly_savings": current_cooling * night_efficiency_gain * 0.12,
            "annual_savings": current_cooling * night_efficiency_gain * 0.12 * 24 * 365 * 0.3  # 30% of time
        }

    def _calculate_time_to_constraint(self, metrics: Dict) -> Dict:
        """Calculate time until hitting capacity constraints."""
        power_util = (metrics.get("powerIT", 520) / 1000) / 4.0
        rack_util = 102 / 120

        # Assume 2.5% monthly growth in power, 1.8% in space
        months_to_power_limit = (0.9 - power_util) / 0.025 if power_util < 0.9 else 0
        months_to_space_limit = (0.95 - rack_util) / 0.018 if rack_util < 0.95 else 0

        return {
            "months_to_power_constraint": max(0, months_to_power_limit),
            "months_to_space_constraint": max(0, months_to_space_limit),
            "limiting_factor": "power" if months_to_power_limit < months_to_space_limit else "space"
        }

    def _extract_key_metrics(self, payload: Dict) -> Dict:
        """Extract key metrics from payload."""
        return {
            "pue": payload.get("pue", "N/A"),
            "temperature": payload.get("temperature", "N/A"),
            "power_efficiency": payload.get("powerIT", 0) / (payload.get("powerIT", 1) + payload.get("powerCooling", 1))
        }

    def _count_anomalies(self, payload: Dict) -> int:
        """Count anomalies in payload."""
        count = 0
        if payload.get("pue", 1.2) > 1.4:
            count += 1
        if payload.get("temperature", 22.5) > 26:
            count += 1
        return count


class ReasoningAgent(BaseAgent):
    """Agent 2: Specialized in causal analysis, correlation finding, and impact assessment."""

    def __init__(self):
        super().__init__("agent_2", "Reasoning Specialist")
        # Initialize reasoning tools
        self.cost_models = self._load_cost_models()
        self.correlation_engine = self._init_correlation_engine()

    async def analyze_patterns(self, retrieval_findings: Dict, context: str) -> AgentResponse:
        """Analyze patterns from Agent 1 to determine root causes."""
        start_time = time.time()

        # Extract Agent 1's findings
        agent1_data = retrieval_findings.get("findings", {})
        patterns = agent1_data.get("patterns", "")
        anomalies = agent1_data.get("anomalies", [])

        # Perform causal analysis
        root_causes = self._identify_root_causes(patterns, anomalies)
        correlations = self._find_correlations(agent1_data)
        impact_assessment = self._assess_impact(root_causes, correlations)

        # Generate reasoning using NVIDIA NIM
        prompt = f"""As a Reasoning Specialist, analyze these data patterns to determine root causes:\n\nRETRIEVAL AGENT FINDINGS:\n{patterns}\n\nDETECTED ANOMALIES:\n{json.dumps(anomalies, indent=2)}\n\nCORRELATION ANALYSIS:\n{json.dumps(correlations, indent=2)}\n\nIMPACT ASSESSMENT:\n{json.dumps(impact_assessment, indent=2)}\n\nYour task:\n1. Determine WHY these patterns exist\n2. Identify root causes of anomalies\n3. Assess correlations and dependencies\n4. Evaluate financial and operational impact\n\nFocus on causal reasoning and impact analysis.\nFormat response as: "REASONING ANALYSIS: [your analysis]" """

        analysis = await self._call_nvidia_nim(prompt, max_tokens=400)

        execution_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            findings={
                "reasoning": analysis,
                "root_causes": root_causes,
                "correlations": correlations,
                "impact_assessment": impact_assessment,
                "confidence_score": self._calculate_confidence(root_causes, correlations)
            },
            execution_time=execution_time,
            confidence=0.88,
            metadata={"root_causes_found": len(root_causes)}
        )

    def _load_cost_models(self) -> Dict:
        """Load cost calculation models."""
        return {
            "energy_cost_per_kwh": 0.12,
            "equipment_failure_cost": {"min": 75000, "max": 150000},
            "downtime_cost_per_hour": 10000,
            "maintenance_cost_factor": 0.15,
            "efficiency_savings_factor": 0.2
        }

    def _init_correlation_engine(self) -> Dict:
        """Initialize correlation analysis engine."""
        return {
            "pue_temperature_correlation": 0.85,
            "cooling_power_correlation": 0.92,
            "utilization_efficiency_correlation": 0.78,
            "temperature_failure_correlation": 0.73
        }

    def _identify_root_causes(self, patterns: str, anomalies: List[Dict]) -> List[Dict]:
        """Identify root causes from patterns and anomalies."""
        root_causes = []

        # Analyze each anomaly for root causes
        for anomaly in anomalies:
            cause = {
                "anomaly_type": anomaly["type"],
                "severity": anomaly["severity"],
                "root_cause": self._determine_root_cause(anomaly),
                "contributing_factors": self._find_contributing_factors(anomaly)
            }
            root_causes.append(cause)

        return root_causes

    def _determine_root_cause(self, anomaly: Dict) -> str:
        """Determine root cause for specific anomaly."""
        anomaly_type = anomaly["type"]

        if anomaly_type == "pue_high":
            return "Cooling system inefficiency or power distribution losses"
        elif anomaly_type == "temperature_high":
            return "Inadequate cooling capacity or airflow obstruction"
        elif anomaly_type == "cooling_inefficient":
            return "Oversized cooling system or poor load balancing"
        else:
            return "Multiple system interactions requiring analysis"

    def _find_contributing_factors(self, anomaly: Dict) -> List[str]:
        """Find contributing factors for anomaly."""
        factors = []

        if anomaly["type"] in ["pue_high", "pue_elevated"]:
            factors.extend([
                "Cooling system oversizing",
                "Power distribution inefficiencies",
                "Auxiliary system power consumption"
            ])

        if anomaly["type"] in ["temperature_high", "temperature_low"]:
            factors.extend([
                "HVAC system imbalance",
                "Blocked airflow paths",
                "Seasonal weather variations"
            ])

        return factors

    def _find_correlations(self, agent1_data: Dict) -> Dict:
        """Find correlations in the data."""
        correlations = {}

        # Extract metrics for correlation analysis
        raw_results = agent1_data.get("raw_search_results", [])
        anomalies = agent1_data.get("anomalies", [])

        if anomalies:
            correlations["anomaly_clustering"] = self._analyze_anomaly_clustering(anomalies)

        correlations["pattern_strength"] = len(raw_results) * 0.2  # Simple strength metric
        correlations["temporal_patterns"] = self._identify_temporal_patterns(raw_results)

        return correlations

    def _analyze_anomaly_clustering(self, anomalies: List[Dict]) -> Dict:
        """Analyze how anomalies cluster together."""
        if len(anomalies) < 2:
            return {"clustering": "none", "related_systems": []}

        # Group anomalies by system
        thermal_anomalies = [a for a in anomalies if "temperature" in a["type"]]
        power_anomalies = [a for a in anomalies if "pue" in a["type"] or "cooling" in a["type"]]

        return {
            "clustering": "high" if len(thermal_anomalies) > 0 and len(power_anomalies) > 0 else "low",
            "related_systems": ["thermal", "power"] if len(thermal_anomalies) > 0 and len(power_anomalies) > 0 else [],
            "anomaly_count": len(anomalies)
        }

    def _identify_temporal_patterns(self, search_results: List[Dict]) -> Dict:
        """Identify temporal patterns in search results."""
        return {
            "pattern_consistency": "high" if len(search_results) > 3 else "medium",
            "historical_precedence": len(search_results) > 0,
            "trend_direction": "deteriorating" if len(search_results) > 2 else "stable"
        }

    def _assess_impact(self, root_causes: List[Dict], correlations: Dict) -> Dict:
        """Assess the impact of identified issues."""
        total_severity = sum(1 if cause["severity"] == "high" else 0.5 for cause in root_causes)

        # Calculate financial impact
        base_monthly_cost = 50000  # Base monthly operating cost
        efficiency_loss = min(0.3, total_severity * 0.1)  # Max 30% efficiency loss
        additional_cost = base_monthly_cost * efficiency_loss

        return {
            "severity_score": total_severity,
            "financial_impact": {
                "monthly_additional_cost": additional_cost,
                "annual_impact": additional_cost * 12,
                "efficiency_loss_percent": efficiency_loss * 100
            },
            "operational_impact": {
                "reliability_risk": "high" if total_severity > 2 else "medium",
                "maintenance_frequency": "increased" if total_severity > 1.5 else "normal"
            },
            "risk_factors": self._identify_risk_factors(root_causes, correlations)
        }

    def _identify_risk_factors(self, root_causes: List[Dict], correlations: Dict) -> List[str]:
        """Identify risk factors from analysis."""
        risks = []

        high_severity_causes = [c for c in root_causes if c["severity"] == "high"]
        if high_severity_causes:
            risks.append("Equipment failure risk elevated")

        if correlations.get("anomaly_clustering", {}).get("clustering") == "high":
            risks.append("Cascading failure potential")

        if len(root_causes) > 2:
            risks.append("Multiple system degradation")

        return risks

    def _calculate_confidence(self, root_causes: List[Dict], correlations: Dict) -> float:
        """Calculate confidence score for reasoning."""
        base_confidence = 0.7

        # Increase confidence with more root causes identified
        cause_bonus = min(0.2, len(root_causes) * 0.05)

        # Increase confidence with strong correlations
        correlation_bonus = 0.1 if correlations.get("pattern_strength", 0) > 0.5 else 0

        return min(1.0, base_confidence + cause_bonus + correlation_bonus)


class ActionPlanningAgent(BaseAgent):
    """Agent 3: Specialized in generating actionable recommendations with ROI calculations."""

    def __init__(self):
        super().__init__("agent_3", "Action Planning Specialist")
        # Initialize planning tools
        self.roi_calculator = self._init_roi_calculator()
        self.implementation_planner = self._init_implementation_planner()
        self.resource_optimizer = self._init_resource_optimizer()

    async def generate_recommendations(self, reasoning_findings: Dict, context: str) -> AgentResponse:
        """Generate actionable recommendations based on reasoning analysis."""
        start_time = time.time()

        # Extract reasoning analysis
        reasoning_data = reasoning_findings.get("findings", {})
        root_causes = reasoning_data.get("root_causes", [])
        impact_assessment = reasoning_data.get("impact_assessment", {})

        # Generate recommendations
        recommendations = self._create_recommendations(root_causes, impact_assessment)
        roi_analysis = self._calculate_roi(recommendations, impact_assessment)
        implementation_plan = self._create_implementation_plan(recommendations)

        # Generate action plan using NVIDIA NIM
        prompt = f"""As an Action Planning Specialist, create specific actionable recommendations:

REASONING AGENT ANALYSIS:
{reasoning_data.get('reasoning', '')}

ROOT CAUSES IDENTIFIED:
{json.dumps(root_causes, indent=2)}

IMPACT ASSESSMENT:
{json.dumps(impact_assessment, indent=2)}

GENERATED RECOMMENDATIONS:
{json.dumps(recommendations, indent=2)}

ROI ANALYSIS:
{json.dumps(roi_analysis, indent=2)}

Your task:
1. Create specific actionable recommendations
2. Provide implementation timelines
3. Calculate costs and ROI
4. Prioritize actions by impact

Focus on actionable steps with clear business value.
Format response as: "ACTION RECOMMENDATIONS: [your recommendations]" """

        analysis = await self._call_nvidia_nim(prompt, max_tokens=600)

        execution_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            findings={
                "action_plan": analysis,
                "recommendations": recommendations,
                "roi_analysis": roi_analysis,
                "implementation_plan": implementation_plan,
                "priority_matrix": self._create_priority_matrix(recommendations, roi_analysis)
            },
            execution_time=execution_time,
            confidence=0.92,
            metadata={"recommendations_count": len(recommendations)}
        )

    def _init_roi_calculator(self) -> Dict:
        """Initialize ROI calculation engine."""
        return {
            "discount_rate": 0.1,  # 10% discount rate
            "implementation_cost_factor": 0.2,  # 20% of annual savings
            "risk_adjustment_factor": 0.9,  # 10% risk adjustment
            "payback_target_months": 24  # Target 2-year payback
        }

    def _init_implementation_planner(self) -> Dict:
        """Initialize implementation planning engine."""
        return {
            "standard_timelines": {
                "cooling_optimization": {"weeks": 2, "complexity": "medium"},
                "power_optimization": {"weeks": 3, "complexity": "high"},
                "monitoring_setup": {"weeks": 1, "complexity": "low"},
                "equipment_upgrade": {"weeks": 8, "complexity": "high"}
            },
            "resource_requirements": {
                "low": {"staff_hours": 40, "contractor_cost": 5000},
                "medium": {"staff_hours": 80, "contractor_cost": 15000},
                "high": {"staff_hours": 160, "contractor_cost": 35000}
            }
        }

    def _init_resource_optimizer(self) -> Dict:
        """Initialize resource optimization engine."""
        return {
            "staff_hourly_rate": 85,
            "contractor_hourly_rate": 125,
            "equipment_cost_multiplier": 1.15,  # 15% markup
            "downtime_tolerance": 0.02  # 2% downtime tolerance
        }

    def _create_recommendations(self, root_causes: List[Dict], impact_assessment: Dict) -> List[Dict]:
        """Create specific recommendations based on root causes."""
        recommendations = []

        for cause in root_causes:
            rec = self._generate_recommendation_for_cause(cause, impact_assessment)
            if rec:
                recommendations.append(rec)

        # Add general optimization recommendations
        if impact_assessment.get("financial_impact", {}).get("efficiency_loss_percent", 0) > 10:
            recommendations.append({
                "id": "general_optimization",
                "title": "Comprehensive System Optimization",
                "description": "Implement systematic optimization across all identified inefficiencies",
                "category": "optimization",
                "priority": "high",
                "estimated_savings": impact_assessment.get("financial_impact", {}).get("monthly_additional_cost", 5000),
                "implementation_complexity": "medium"
            })

        return recommendations

    def _generate_recommendation_for_cause(self, cause: Dict, impact_assessment: Dict) -> Optional[Dict]:
        """Generate specific recommendation for a root cause."""
        cause_type = cause["anomaly_type"]
        severity = cause["severity"]

        if cause_type in ["pue_high", "pue_elevated"]:
            return {
                "id": f"cooling_optimization_{cause_type}",
                "title": "Cooling System Optimization",
                "description": "Optimize cooling system efficiency and reduce PUE",
                "category": "cooling",
                "priority": "high" if severity == "high" else "medium",
                "estimated_savings": 8000 if severity == "high" else 4000,
                "implementation_complexity": "medium",
                "specific_actions": [
                    "Adjust cooling setpoints",
                    "Optimize airflow patterns",
                    "Calibrate CRAC units"
                ]
            }

        elif cause_type in ["temperature_high", "temperature_low"]:
            return {
                "id": f"thermal_management_{cause_type}",
                "title": "Thermal Management Improvement",
                "description": "Improve thermal management and temperature control",
                "category": "thermal",
                "priority": "high" if severity == "high" else "medium",
                "estimated_savings": 6000 if severity == "high" else 3000,
                "implementation_complexity": "low",
                "specific_actions": [
                    "Review HVAC settings",
                    "Check for airflow obstructions",
                    "Implement temperature monitoring"
                ]
            }

        elif cause_type == "cooling_inefficient":
            return {
                "id": "cooling_efficiency_improvement",
                "title": "Cooling Efficiency Enhancement",
                "description": "Reduce cooling power consumption and improve efficiency",
                "category": "efficiency",
                "priority": "medium",
                "estimated_savings": 5000,
                "implementation_complexity": "medium",
                "specific_actions": [
                    "Right-size cooling capacity",
                    "Implement variable speed drives",
                    "Optimize cooling distribution"
                ]
            }

        return None

    def _calculate_roi(self, recommendations: List[Dict], impact_assessment: Dict) -> Dict:
        """Calculate ROI for recommendations."""
        total_monthly_savings = sum(rec.get("estimated_savings", 0) for rec in recommendations)
        total_annual_savings = total_monthly_savings * 12

        # Estimate implementation costs
        total_implementation_cost = 0
        for rec in recommendations:
            complexity = rec.get("implementation_complexity", "medium")
            cost = self.implementation_planner["resource_requirements"][complexity]["contractor_cost"]
            total_implementation_cost += cost

        # Calculate ROI metrics
        payback_months = (total_implementation_cost / total_monthly_savings) if total_monthly_savings > 0 else 999
        roi_percentage = ((total_annual_savings - total_implementation_cost) / total_implementation_cost * 100) if total_implementation_cost > 0 else 0

        return {
            "total_monthly_savings": total_monthly_savings,
            "total_annual_savings": total_annual_savings,
            "total_implementation_cost": total_implementation_cost,
            "payback_months": round(payback_months, 1),
            "roi_percentage": round(roi_percentage, 1),
            "net_present_value": self._calculate_npv(total_annual_savings, total_implementation_cost),
            "risk_adjusted_roi": round(roi_percentage * self.roi_calculator["risk_adjustment_factor"], 1)
        }

    def _calculate_npv(self, annual_savings: float, initial_cost: float, years: int = 3) -> float:
        """Calculate Net Present Value."""
        discount_rate = self.roi_calculator["discount_rate"]
        npv = -initial_cost

        for year in range(1, years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)

        return round(npv, 2)

    def _create_implementation_plan(self, recommendations: List[Dict]) -> Dict:
        """Create detailed implementation plan."""
        # Sort recommendations by priority and complexity
        sorted_recs = sorted(recommendations, key=lambda x: (
            0 if x.get("priority") == "high" else 1,
            {"low": 1, "medium": 2, "high": 3}.get(x.get("implementation_complexity", "medium"), 2)
        ))

        timeline = []
        current_week = 0

        for rec in sorted_recs:
            complexity = rec.get("implementation_complexity", "medium")
            duration_weeks = self.implementation_planner["standard_timelines"].get(
                rec.get("category", "general"), {"weeks": 2}
            )["weeks"]

            timeline.append({
                "recommendation_id": rec["id"],
                "title": rec["title"],
                "start_week": current_week + 1,
                "duration_weeks": duration_weeks,
                "end_week": current_week + duration_weeks,
                "priority": rec.get("priority", "medium"),
                "dependencies": self._identify_dependencies(rec, sorted_recs)
            })

            current_week += duration_weeks

        return {
            "timeline": timeline,
            "total_duration_weeks": current_week,
            "critical_path": self._identify_critical_path(timeline),
            "resource_allocation": self._calculate_resource_allocation(timeline)
        }

    def _identify_dependencies(self, rec: Dict, all_recs: List[Dict]) -> List[str]:
        """Identify dependencies between recommendations."""
        dependencies = []

        # Cooling optimizations should come before efficiency improvements
        if rec.get("category") == "efficiency":
            cooling_recs = [r["id"] for r in all_recs if r.get("category") == "cooling"]
            dependencies.extend(cooling_recs)

        return dependencies

    def _identify_critical_path(self, timeline: List[Dict]) -> List[str]:
        """Identify critical path through implementation."""
        # Simple critical path - longest duration items first
        critical_items = sorted(timeline, key=lambda x: x["duration_weeks"], reverse=True)
        return [item["recommendation_id"] for item in critical_items[:3]]

    def _calculate_resource_allocation(self, timeline: List[Dict]) -> Dict:
        """Calculate resource allocation for implementation."""
        total_staff_hours = 0
        total_contractor_cost = 0

        for item in timeline:
            # Estimate resources based on duration
            complexity = "medium"  # Default
            resources = self.implementation_planner["resource_requirements"][complexity]

            total_staff_hours += resources["staff_hours"]
            total_contractor_cost += resources["contractor_cost"]

        return {
            "total_staff_hours": total_staff_hours,
            "total_contractor_cost": total_contractor_cost,
            "estimated_total_cost": total_staff_hours * self.resource_optimizer["staff_hourly_rate"] + total_contractor_cost,
            "peak_weekly_hours": total_staff_hours / max(1, len(timeline)) if timeline else 0
        }

    def _create_priority_matrix(self, recommendations: List[Dict], roi_analysis: Dict) -> Dict:
        """Create priority matrix for recommendations."""
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        low_priority = [r for r in recommendations if r.get("priority") == "low"]

        return {
            "high_impact_quick_wins": [r["id"] for r in high_priority if r.get("implementation_complexity") == "low"],
            "high_impact_major_projects": [r["id"] for r in high_priority if r.get("implementation_complexity") == "high"],
            "medium_impact_actions": [r["id"] for r in medium_priority],
            "low_priority_actions": [r["id"] for r in low_priority],
            "recommended_sequence": [r["id"] for r in sorted(recommendations, key=lambda x: (
                0 if x.get("priority") == "high" else 1,
                -x.get("estimated_savings", 0)
            ))]
        }


class MultiAgentOrchestrator:
    """Orchestrator that coordinates the three specialized agents."""

    def __init__(self):
        self.agent1 = DataRetrievalAgent()
        self.agent2 = ReasoningAgent()
        self.agent3 = ActionPlanningAgent()
        logger.info("Multi-agent orchestrator initialized")

    async def orchestrate_investigation(self, user_question: str, current_metrics: Dict) -> Dict:
        """Orchestrate a full investigation using all three agents."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR: Starting multi-agent investigation")
        logger.info(f"ORCHESTRATOR: Question={user_question[:100]}")
        logger.info(f"ORCHESTRATOR: Environment check - async context available")

        # Set default metrics if none provided
        if not current_metrics:
            current_metrics = {
                "pue": 1.41,
                "temperature": 24.5,
                "powerIT": 539,
                "powerCooling": 228,
                "occupied_racks": 102,
                "total_racks": 120
            }
            logger.info("ORCHESTRATOR: Using default metrics")

        results = {
            "orchestrator_id": "multi_agent_orchestrator",
            "investigation_type": "full_analysis",
            "user_question": user_question,
            "current_metrics": current_metrics,
            "agents": {}
        }

        try:
            # Stage 1: Data Retrieval Agent
            stage1_start = time.time()
            logger.info("ORCHESTRATOR: Stage 1/3 - Calling Data Retrieval Agent")
            agent1_response = await self.agent1.extract_patterns(user_question, current_metrics)
            stage1_duration = time.time() - stage1_start
            logger.info(f"ORCHESTRATOR: Stage 1/3 COMPLETED in {stage1_duration:.2f}s")
            logger.info(f"ORCHESTRATOR: Agent1 confidence={agent1_response.confidence:.2f}")
            results["agents"]["agent_1"] = {
                "status": "completed",
                "response": agent1_response,
                "stage": "data_retrieval",
                "duration": stage1_duration
            }

            # Stage 2: Reasoning Agent (processes Agent 1's output)
            stage2_start = time.time()
            logger.info("ORCHESTRATOR: Stage 2/3 - Calling Reasoning Agent")
            agent2_response = await self.agent2.analyze_patterns(
                {"findings": agent1_response.findings},
                user_question
            )
            stage2_duration = time.time() - stage2_start
            logger.info(f"ORCHESTRATOR: Stage 2/3 COMPLETED in {stage2_duration:.2f}s")
            logger.info(f"ORCHESTRATOR: Agent2 confidence={agent2_response.confidence:.2f}")
            results["agents"]["agent_2"] = {
                "status": "completed",
                "response": agent2_response,
                "stage": "reasoning_analysis",
                "duration": stage2_duration
            }

            # Stage 3: Action Planning Agent (processes Agent 2's output)
            stage3_start = time.time()
            logger.info("ORCHESTRATOR: Stage 3/3 - Calling Action Planning Agent")
            agent3_response = await self.agent3.generate_recommendations(
                {"findings": agent2_response.findings},
                user_question
            )
            stage3_duration = time.time() - stage3_start
            logger.info(f"ORCHESTRATOR: Stage 3/3 COMPLETED in {stage3_duration:.2f}s")
            logger.info(f"ORCHESTRATOR: Agent3 confidence={agent3_response.confidence:.2f}")
            results["agents"]["agent_3"] = {
                "status": "completed",
                "response": agent3_response,
                "stage": "action_planning",
                "duration": stage3_duration
            }

            # Combine all agent outputs
            logger.info("ORCHESTRATOR: Combining agent outputs")
            combined_response = self._combine_agent_outputs(
                agent1_response, agent2_response, agent3_response
            )

            results["combined_analysis"] = combined_response
            results["total_execution_time"] = time.time() - start_time
            results["status"] = "success"

            logger.info(f"ORCHESTRATOR: SUCCESS - Total time {results['total_execution_time']:.2f}s")
            logger.info(f"ORCHESTRATOR: Response preview: {combined_response[:150]}...")
            logger.info("=" * 80)
            return results

        except Exception as error:
            error_time = time.time() - start_time
            logger.error("=" * 80)
            logger.error(f"ORCHESTRATOR: FAILED after {error_time:.2f}s")
            logger.error(f"ORCHESTRATOR: Exception type: {type(error).__name__}")
            logger.error(f"ORCHESTRATOR: Exception message: {str(error)}")
            logger.error(f"ORCHESTRATOR: Exception details: {repr(error)}")

            # Log which stage failed
            completed_stages = len(results["agents"])
            logger.error(f"ORCHESTRATOR: Failed at stage {completed_stages + 1}/3")

            import traceback
            logger.error(f"ORCHESTRATOR: Traceback:\n{traceback.format_exc()}")
            logger.error("=" * 80)

            results["status"] = "error"
            results["error"] = str(error)
            results["error_type"] = type(error).__name__
            results["failed_stage"] = completed_stages + 1
            results["total_execution_time"] = error_time
            return results

    def _combine_agent_outputs(self, agent1: AgentResponse, agent2: AgentResponse, agent3: AgentResponse) -> str:
        """Combine outputs from all three agents into a coherent response."""
        return f"""MULTI-AGENT INVESTIGATION RESULTS:

{agent1.findings.get('patterns', 'No patterns analysis available')}

{agent2.findings.get('reasoning', 'No reasoning analysis available')}

{agent3.findings.get('action_plan', 'No action plan available')}

EXECUTION SUMMARY:
- Data Retrieval Agent: {agent1.execution_time:.2f}s (Confidence: {agent1.confidence:.0%})
- Reasoning Agent: {agent2.execution_time:.2f}s (Confidence: {agent2.confidence:.0%})
- Action Planning Agent: {agent3.execution_time:.2f}s (Confidence: {agent3.confidence:.0%})

RECOMMENDATIONS SUMMARY:
Total Recommendations: {len(agent3.findings.get('recommendations', []))}
Expected ROI: {agent3.findings.get('roi_analysis', {}).get('roi_percentage', 0):.1f}%
Payback Period: {agent3.findings.get('roi_analysis', {}).get('payback_months', 0):.1f} months"""


# Create module-level instances
multi_agent_orchestrator = MultiAgentOrchestrator()
data_retrieval_agent = DataRetrievalAgent()