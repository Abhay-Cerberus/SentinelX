"""SentinelX core environment — implements the OpenEnv Environment interface.

reset() → initial FraudObservation
step(action) → FraudObservation + reward + done
state (property) → FraudInvestigationState
"""
from __future__ import annotations

import sys
import os
import uuid
from typing import Any, Dict, List, Optional

# Ensure project root is importable when running from server/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.core.env_server import Environment  # type: ignore

from sentinelx.models import (
    FraudAction,
    FraudObservation,
    FraudInvestigationState,
    INVESTIGATION_ACTIONS,
    INTERVENTION_ACTIONS,
    REGULATORY_ACTIONS,
    TERMINAL_ACTIONS,
)
from sentinelx.adversary.fraudster import FraudsterAgent
from sentinelx.adversary.strategies import Scenario
from sentinelx.server.graders import EpisodeRecord, grade_episode

import tasks.stolen_card as _task1
import tasks.account_takeover as _task2
import tasks.money_laundering as _task3

TASK_FACTORIES = {
    "stolen-card-easy": _task1,
    "account-takeover-medium": _task2,
    "money-laundering-hard": _task3,
}

# Per-tick time cost (discourages indefinite investigation)
TIME_PENALTY = -0.01

# Reward constants
REWARD_GOOD_EVIDENCE = 0.05
REWARD_WASTED_EVIDENCE = -0.02
REWARD_CORRECT_BLOCK = 0.30
REWARD_FRAUD_IN_PROGRESS_BONUS = 0.10
REWARD_CORRECT_APPROVE = 0.30
REWARD_FALSE_POSITIVE = -0.40
REWARD_FALSE_NEGATIVE = -0.50
REWARD_SAR_FILED = 0.15
REWARD_SAR_MISSED = -0.20
REWARD_SINGLE_SIGNAL_ABUSE = -0.10


class SentinelXEnvironment(Environment):
    """Main fraud investigation RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._state = FraudInvestigationState()
        self._scenario: Optional[Scenario] = None
        self._record: Optional[EpisodeRecord] = None
        self._adversary: Optional[FraudsterAgent] = None
        self._episode_history: List[Dict[str, Any]] = []
        self._done: bool = False

    def reset(
        self,
        task_id: str = "stolen-card-easy",
        seed: int = 42,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FraudObservation:
        """Start a fresh episode for the given task."""
        task_id = str(task_id)
        if task_id not in TASK_FACTORIES:
            task_id = "stolen-card-easy"

        self._adversary = FraudsterAgent.from_history(task_id, self._episode_history)
        task_mod = TASK_FACTORIES[task_id]

        # Generate scenario via the adversary (may be escalated)
        self._scenario = self._adversary.generate_scenario(
            seed=int(seed),
            user_profile=self._load_user_profile(task_mod),
        )

        max_ticks = task_mod.TASK_CONFIG["max_ticks"]

        self._state = FraudInvestigationState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            fraud_type=self._scenario.fraud_type,
            fraudster_strategy=getattr(self._scenario, "fraudster_strategy", "basic"),
            investigation_ticks=0,
            fraud_in_progress=True,
            sar_filed=False,
            ctr_filed=False,
            final_action=None,
        )

        self._record = EpisodeRecord(
            task_id=task_id,
            fraud_type=self._scenario.fraud_type,
            is_fraud=self._scenario.is_fraud,
            total_laundered_amount=self._scenario.total_laundered_amount,
            structuring_detected=self._scenario.structuring_detected,
        )

        self._done = False

        return self._build_observation(
            reward=None,
            done=False,
            max_ticks=max_ticks,
            last_result="Investigation started. Gather evidence before making a decision.",
        )

    def step(
        self,
        action: FraudAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FraudObservation:
        """Execute one agent action and return the resulting observation."""
        if self._done or self._scenario is None:
            return self._build_observation(reward=0.0, done=True, last_result="Episode already ended.")

        self._state.step_count += 1
        self._state.investigation_ticks += 1
        self._record.total_steps += 1
        self._record.actions_taken.append(action.action_type)

        task_mod = TASK_FACTORIES[self._state.task_id]
        max_ticks = task_mod.TASK_CONFIG["max_ticks"]

        # Check time-out
        if self._state.investigation_ticks > max_ticks:
            self._done = True
            self._record.timed_out = True
            return self._build_observation(
                reward=REWARD_FALSE_NEGATIVE,
                done=True,
                last_result="⏰ Time expired — case auto-escalated. Fraud proceeded undetected.",
            )

        reward, result_msg = self._dispatch(action)

        # Time pressure
        reward += TIME_PENALTY

        # Anti-pattern: agent over-relying on one signal type
        if self._is_single_signal_abuse(action.action_type):
            reward += REWARD_SINGLE_SIGNAL_ABUSE

        self._record.step_rewards.append(round(reward, 4))

        # Terminal condition
        done = self._done or action.action_type in TERMINAL_ACTIONS
        if done:
            self._done = True
            self._state.final_action = action.action_type

        return self._build_observation(
            reward=round(reward, 4),
            done=self._done,
            max_ticks=max_ticks,
            last_result=result_msg,
        )

    @property
    def state(self) -> FraudInvestigationState:
        return self._state

    # ------------------------------------------------------------------ #
    # Action dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(self, action: FraudAction) -> tuple[float, str]:
        at = action.action_type
        sc = self._scenario

        # ---- Investigation actions ----
        if at == "query_velocity":
            return self._reveal("velocity_data", sc.velocity_data, "velocity")
        if at == "check_device_history":
            return self._reveal("device_history", sc.device_history, "device")
        if at == "lookup_ip_reputation":
            return self._reveal("ip_reputation", sc.ip_reputation, "ip")
        if at == "check_behavioral_biometrics":
            return self._reveal("behavioral_biometrics", sc.behavioral_biometrics, "behavioral")
        if at == "check_active_sessions":
            return self._reveal("active_sessions", sc.active_sessions, "sessions")
        if at == "query_linked_accounts":
            return self._reveal_network(action.parameters.get("depth", 1))
        if at == "analyze_temporal_pattern":
            return self._reveal("temporal_pattern", sc.temporal_pattern, "temporal")
        if at == "check_business_registration":
            self._record.business_registration_checked = True
            return self._reveal("business_registration", sc.business_registration, "business_reg")
        if at == "request_kyc_documents":
            return self._reveal("business_registration", sc.business_registration, "kyc")

        # ---- Intervention actions ----
        if at == "approve_transaction":
            return self._do_approve()
        if at == "block_transaction":
            return self._do_block()
        if at == "request_3ds":
            return self._do_3ds()
        if at == "send_push_notification":
            return self._do_push_notify()
        if at == "force_password_reset":
            return self._do_password_reset()
        if at == "temporarily_freeze_account":
            return self._do_freeze()

        # ---- Regulatory ----
        if at == "file_sar":
            return self._do_file_sar()
        if at == "file_ctr":
            return self._do_file_ctr()
        if at == "escalate_to_compliance":
            self._record.escalated = True
            return 0.05, "📋 Case escalated to compliance team for human review."

        # ---- Special ----
        if at == "monitor_only":
            return -0.01, "👁️ Monitoring transaction without intervention."
        if at == "request_additional_info":
            return -0.01, "📨 Additional information requested from customer."

        return 0.0, f"Unknown action: {at}"

    # ------------------------------------------------------------------ #
    # Evidence revelation helpers
    # ------------------------------------------------------------------ #

    def _reveal(
        self,
        field_name: str,
        evidence: Any,
        signal_key: str,
    ) -> tuple[float, str]:
        """Reveal an evidence field and return appropriate reward."""
        if field_name in self._record.evidence_gathered:
            return REWARD_WASTED_EVIDENCE, f"⚠️ Already checked {field_name} — no new information."

        self._record.evidence_gathered.append(field_name)

        if not evidence:
            return REWARD_WASTED_EVIDENCE, f"🔍 {field_name}: No data found."

        # Is this evidence relevant to the actual fraud type?
        relevant = self._is_relevant_evidence(signal_key)
        if relevant:
            return REWARD_GOOD_EVIDENCE, f"✅ {field_name}: Evidence collected. {self._summarise(field_name, evidence)}"
        else:
            return REWARD_WASTED_EVIDENCE, f"🔍 {field_name}: Nothing suspicious found."

    def _reveal_network(self, depth: int) -> tuple[float, str]:
        depth = max(1, min(3, int(depth)))
        self._record.network_depth_checked = max(self._record.network_depth_checked, depth)

        graph = self._scenario.network_graph
        if not graph:
            return REWARD_WASTED_EVIDENCE, "🔍 No network connections found."

        key = f"depth_{depth}"
        level_data = graph.get(key, {})
        nodes = level_data.get("nodes", [])
        offshore = level_data.get("offshore_entities", [])
        crypto = level_data.get("crypto_exchanges", [])
        mules = level_data.get("mule_accounts", [])

        # Detect shell companies at this depth
        shells = [n for n in nodes if "LLC" in n or "Ltd" in n or "Holdings" in n or "Investments" in n]
        self._record.shell_companies_identified = max(self._record.shell_companies_identified, len(shells))

        if self._scenario.fraud_type == "money_laundering":
            self._record.evidence_gathered.append(f"network_connections_d{depth}")
            summary = (
                f"🕸️ Depth-{depth} network: {len(nodes)} nodes found. "
                f"Shell companies: {len(shells)}. "
                f"Offshore entities: {len(offshore)}. "
                f"Crypto exchanges: {len(crypto)}. "
                f"Mule accounts: {len(mules)}."
            )
            return REWARD_GOOD_EVIDENCE, summary

        return REWARD_WASTED_EVIDENCE, f"🔍 Depth-{depth} network: {len(nodes)} connections found."

    def _is_relevant_evidence(self, signal_key: str) -> bool:
        fraud_type = self._scenario.fraud_type
        relevance_map = {
            "stolen_card": {"velocity", "device", "ip", "sessions"},
            "account_takeover": {"behavioral", "sessions", "device", "ip"},
            "money_laundering": {"temporal", "business_reg", "kyc", "velocity"},
        }
        relevant_signals = relevance_map.get(fraud_type, set())
        return any(sig in signal_key for sig in relevant_signals)

    def _summarise(self, field_name: str, evidence: Any) -> str:
        """Return a brief natural-language summary of evidence."""
        if field_name == "velocity_data" and isinstance(evidence, dict):
            return (f"Past 24h: {evidence.get('past_24_hours_count', '?')} transactions. "
                    f"Anomaly score: {evidence.get('anomaly_score', '?')}.")
        if field_name == "device_history" and isinstance(evidence, dict):
            return f"New device: {evidence.get('is_new_device', '?')}. Risk score: {evidence.get('device_risk_score', '?')}."
        if field_name == "ip_reputation" and isinstance(evidence, dict):
            return (f"IP type: {evidence.get('type', '?')}. "
                    f"Risk: {evidence.get('risk_score', '?')}. "
                    f"VPN: {evidence.get('is_vpn', '?')}. TOR: {evidence.get('is_tor', '?')}.")
        if field_name == "behavioral_biometrics" and isinstance(evidence, dict):
            return (f"Typing: {evidence.get('typing_wpm', '?')} WPM vs baseline {evidence.get('user_baseline_wpm', '?')}. "
                    f"Mouse: {evidence.get('mouse_pattern', '?')}. Anomaly: {evidence.get('anomaly_score', '?')}.")
        if field_name == "active_sessions" and isinstance(evidence, list):
            return f"{len(evidence)} active session(s) found."
        if field_name == "temporal_pattern" and isinstance(evidence, dict):
            self._record.temporal_pattern_checked = True
            return (f"Coordinated accounts: {evidence.get('coordinated_accounts', '?')}. "
                    f"Structuring probability: {evidence.get('structuring_probability', '?')}.")
        if field_name == "business_registration" and isinstance(evidence, dict):
            return (f"Employees: {evidence.get('employees_reported', '?')}. "
                    f"Shell probability: {evidence.get('shell_company_probability', '?')}.")
        return "Evidence gathered."

    # ------------------------------------------------------------------ #
    # Intervention handlers
    # ------------------------------------------------------------------ #

    def _do_approve(self) -> tuple[float, str]:
        if self._scenario.is_fraud:
            self._record.false_negative = True
            self._adversary.adapt("approved")
            return REWARD_FALSE_NEGATIVE, "❌ Transaction APPROVED — but it was fraud! Funds lost."
        else:
            self._record.correctly_approved = True
            return REWARD_CORRECT_APPROVE, "✅ Transaction APPROVED — correctly identified as legitimate."

    def _do_block(self) -> tuple[float, str]:
        if self._scenario.is_fraud:
            self._record.correctly_blocked = True
            reward = REWARD_CORRECT_BLOCK
            msg = "🛑 Transaction BLOCKED — fraud prevented!"
            if self._state.fraud_in_progress:
                reward += REWARD_FRAUD_IN_PROGRESS_BONUS
                msg += " Stopped an active fraud attempt."
            self._record.final_action_params["transaction_amount"] = self._scenario.transaction.get("amount", 0)
            self._adversary.adapt("blocked_via_" + self._primary_signal_used())
            return reward, msg
        else:
            self._record.false_positive = True
            return REWARD_FALSE_POSITIVE, "❌ Transaction BLOCKED — but it was legitimate! Customer harmed."

    def _do_3ds(self) -> tuple[float, str]:
        if self._scenario.is_fraud:
            self._record.correctly_blocked = True
            return REWARD_CORRECT_BLOCK + 0.10, "🔐 3D Secure requested — fraudster unable to complete authentication."
        return REWARD_CORRECT_APPROVE - 0.10, "🔐 3D Secure requested — minor friction on legitimate customer."

    def _do_push_notify(self) -> tuple[float, str]:
        if self._scenario.fraud_type == "account_takeover":
            self._record.correctly_blocked = True
            return 0.10, "📱 Push notification sent — legitimate user will confirm or deny."
        return 0.02, "📱 Push notification sent to card holder."

    def _do_password_reset(self) -> tuple[float, str]:
        if self._scenario.fraud_type == "account_takeover":
            self._record.correctly_blocked = True
            return 0.30, "🔑 Password reset forced — account takeover stopped. Legitimate user secured."
        return -0.15, "🔑 Forced password reset on legitimate user — poor experience."

    def _do_freeze(self) -> tuple[float, str]:
        if self._scenario.is_fraud:
            self._record.correctly_blocked = True
            return 0.25, "🧊 Account temporarily frozen — fraud prevented, legitimate user can appeal."
        return -0.20, "🧊 Legitimate account frozen — customer unable to transact."

    def _do_file_sar(self) -> tuple[float, str]:
        self._state.sar_filed = True
        self._record.sar_filed = True
        if self._scenario.fraud_type == "money_laundering":
            if self._record.network_depth_checked >= 2:
                return REWARD_SAR_FILED, "📝 SAR filed with full network evidence — regulatory obligation met."
            else:
                return REWARD_SAR_FILED - 0.10, "⚠️ SAR filed — but investigation incomplete. Consider checking depth-2 network."
        return 0.05, "📝 SAR filed."

    def _do_file_ctr(self) -> tuple[float, str]:
        self._state.ctr_filed = True
        self._record.ctr_filed = True
        txn_amount = self._scenario.transaction.get("amount", 0)
        if txn_amount >= 10000:
            return 0.15, "📋 CTR filed — Currency Transaction Report submitted for $10K+ transaction."
        return -0.05, "📋 CTR filed — this transaction was under $10K, CTR not strictly required."

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _primary_signal_used(self) -> str:
        signal_order = ["velocity", "device", "ip", "behavioral", "sessions", "network", "temporal"]
        for sig in signal_order:
            for ev in self._record.evidence_gathered:
                if sig in ev:
                    return sig
        return "unknown"

    def _is_single_signal_abuse(self, current_action: str) -> bool:
        """Penalise if agent has used the same investigation action > 3 times."""
        count = self._record.actions_taken.count(current_action)
        return count > 3

    def _load_user_profile(self, task_mod: Any) -> dict:
        from tasks import load_user_profile, TASK_USER_MAP
        return load_user_profile(TASK_USER_MAP[task_mod.TASK_ID])

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        max_ticks: int = 20,
        last_result: str = "",
    ) -> FraudObservation:
        sc = self._scenario
        rec = self._record
        ticks_left = max_ticks - self._state.investigation_ticks

        # Determine available actions
        available = list(INVESTIGATION_ACTIONS | INTERVENTION_ACTIONS | REGULATORY_ACTIONS | {"monitor_only", "request_additional_info"})

        # Remove already-gathered investigation actions
        for ev in (rec.evidence_gathered if rec else []):
            ev_to_action = {
                "velocity_data": "query_velocity",
                "device_history": "check_device_history",
                "ip_reputation": "lookup_ip_reputation",
                "behavioral_biometrics": "check_behavioral_biometrics",
                "active_sessions": "check_active_sessions",
                "temporal_pattern": "analyze_temporal_pattern",
                "business_registration": "check_business_registration",
            }
            for field, act in ev_to_action.items():
                if ev == field and act in available:
                    available.remove(act)

        summary = self._build_evidence_summary()

        obs = FraudObservation(
            done=done,
            reward=reward,
            transaction=sc.transaction if sc else {},
            user_profile=sc.user_profile if sc else {},
            available_actions=sorted(available),
            time_remaining=max(0, ticks_left),
            evidence_summary=summary,
            last_action_result=last_result,
        )

        # Populate revealed evidence fields
        if rec and sc:
            if "velocity_data" in rec.evidence_gathered:
                obs.velocity_data = sc.velocity_data
            if "device_history" in rec.evidence_gathered:
                obs.device_history = sc.device_history
            if "ip_reputation" in rec.evidence_gathered:
                obs.ip_reputation = sc.ip_reputation
            if "behavioral_biometrics" in rec.evidence_gathered:
                obs.behavioral_biometrics = sc.behavioral_biometrics
            if "active_sessions" in rec.evidence_gathered:
                obs.active_sessions = sc.active_sessions
            if "temporal_pattern" in rec.evidence_gathered:
                obs.temporal_pattern = sc.temporal_pattern
            if "business_registration" in rec.evidence_gathered:
                obs.business_registration = sc.business_registration
            net_fields = [f for f in rec.evidence_gathered if f.startswith("network_connections")]
            if net_fields:
                obs.network_connections = sc.network_connections

        return obs

    def _build_evidence_summary(self) -> str:
        if not self._record:
            return "No investigation started yet."
        if not self._record.evidence_gathered:
            return "No evidence gathered yet. Use investigation actions to query available signals."

        lines = ["Evidence gathered so far:"]
        sc = self._scenario

        for field in self._record.evidence_gathered:
            if field == "velocity_data" and sc.velocity_data:
                v = sc.velocity_data
                lines.append(f"  • Velocity: {v.get('past_24_hours_count')} txns in 24h, anomaly score {v.get('anomaly_score')}")
            elif field == "device_history" and sc.device_history:
                d = sc.device_history
                lines.append(f"  • Device: New device={d.get('is_new_device')}, risk={d.get('device_risk_score')}")
            elif field == "ip_reputation" and sc.ip_reputation:
                ip = sc.ip_reputation
                lines.append(f"  • IP: {ip.get('type')} ({ip.get('provider')}), risk={ip.get('risk_score')}, VPN={ip.get('is_vpn')}, TOR={ip.get('is_tor')}")
            elif field == "behavioral_biometrics" and sc.behavioral_biometrics:
                b = sc.behavioral_biometrics
                lines.append(f"  • Biometrics: {b.get('typing_wpm')} WPM (baseline {b.get('user_baseline_wpm')}), mouse={b.get('mouse_pattern')}")
            elif field == "active_sessions" and sc.active_sessions:
                lines.append(f"  • Sessions: {len(sc.active_sessions)} active session(s)")
            elif field == "temporal_pattern" and sc.temporal_pattern:
                t = sc.temporal_pattern
                lines.append(f"  • Temporal: {t.get('coordinated_accounts')} coordinated accounts, structuring prob={t.get('structuring_probability')}")
            elif field == "business_registration" and sc.business_registration:
                b = sc.business_registration
                lines.append(f"  • Business reg: employees={b.get('employees_reported')}, shell prob={b.get('shell_company_probability')}")
            elif field.startswith("network_connections"):
                depth = field.replace("network_connections_d", "")
                lines.append(f"  • Network depth-{depth}: {self._record.shell_companies_identified} shell companies identified")

        lines.append(f"\nTime remaining: {max(0, TASK_FACTORIES[self._state.task_id].TASK_CONFIG['max_ticks'] - self._state.investigation_ticks)} ticks")
        return "\n".join(lines)
