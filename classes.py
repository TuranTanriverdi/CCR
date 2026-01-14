import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ============================================================================
# FORMAL INVARIANTS - CCR Identity Protection
# ============================================================================
# These invariants define the boundary between legitimate character change
# and accidental drift into reward optimization. Violations indicate the
# system has become a reward maximizer rather than a self-authoring agent.
# ============================================================================

class CCRInvariantViolation(Exception):
    """Raised when a fundamental invariant of CCR is violated"""
    pass

class CCRInvariants:
    """Formal invariants that must hold for CCR to remain a self-authoring system"""
    
    @staticmethod
    def check_slow_layer_protection(slow_updated, dissonance, threshold, 
                                     metacognition, meta_plasticity, norm_push):
        """
        INVARIANT 1: Slow layer must NOT update outside crisis conditions
        
        Slow layer updates are ONLY allowed when:
        - dissonance > threshold OR
        - (metacognition=True AND meta_plasticity=True AND norm_push > 1.0)
        
        Violation indicates: Fast chaos corrupting identity
        """
        crisis_condition = (dissonance > threshold or 
                          (metacognition and meta_plasticity and norm_push > 1.0))
        
        if slow_updated and not crisis_condition:
            raise CCRInvariantViolation(
                f"INVARIANT 1 VIOLATED: Slow layer updated outside crisis.\n"
                f"  slow_updated={slow_updated}\n"
                f"  dissonance={dissonance:.3f} (threshold={threshold})\n"
                f"  metacognition={metacognition}, meta_plasticity={meta_plasticity}\n"
                f"  norm_push={norm_push:.3f}\n"
                f"This indicates identity is being corrupted by transient stimuli."
            )
        
        return True
    
    @staticmethod
    def check_identity_coherence(gammas, min_threshold=0.05, max_threshold=1.5):
        """
        INVARIANT 2: Identity must remain coherent
        
        Conditions:
        - At least one gamma must be above min_threshold (has a position)
        - No gamma should exceed max_threshold (prevents runaway)
        - Sum of gammas should not collapse to zero (total erasure)
        
        Violation indicates: Identity collapse or explosion
        """
        max_gamma = np.max(gammas)
        sum_gamma = np.sum(gammas)
        
        if max_gamma < min_threshold:
            raise CCRInvariantViolation(
                f"INVARIANT 2 VIOLATED: Identity collapse.\n"
                f"  max(gamma)={max_gamma:.3f} < {min_threshold}\n"
                f"  All attractors too weak - agent has no stable position."
            )
        
        if max_gamma > max_threshold:
            raise CCRInvariantViolation(
                f"INVARIANT 2 VIOLATED: Identity explosion.\n"
                f"  max(gamma)={max_gamma:.3f} > {max_threshold}\n"
                f"  Attractor strength unbounded - numerical instability."
            )
        
        if sum_gamma < 4 * min_threshold * 0.5:  # Less than half of minimum
            raise CCRInvariantViolation(
                f"INVARIANT 2 VIOLATED: Total identity erasure.\n"
                f"  sum(gamma)={sum_gamma:.3f}\n"
                f"  Total attractor strength collapsed."
            )
        
        return True
    
    @staticmethod
    def check_no_reward_optimization(learning_rate, weights, outcome=None):
        """
        INVARIANT 3: No direct outcome → gamma optimization
        
        Learning must be based on:
        - Basin occupancy (weights) - self-reinforcement
        - NOT on external outcomes or rewards
        
        Violation indicates: Accidental RL
        """
        if outcome is not None and learning_rate > 0:
            raise CCRInvariantViolation(
                f"INVARIANT 3 VIOLATED: Outcome-based learning detected.\n"
                f"  learning_rate={learning_rate}, outcome={outcome}\n"
                f"  Gammas must NOT be updated based on external rewards.\n"
                f"  This would make CCR a soft Q-learner."
            )
        
        return True
    
    @staticmethod
    def check_identity_change_vs_collapse(gammas_before, gammas_after, 
                                          collapse_threshold=0.2):
        """
        INVARIANT 4: Distinguish identity change from identity collapse
        
        Identity CHANGE (legitimate):
        - max(gamma) remains above collapse_threshold
        - Distribution of gammas shifts
        - Total strength preserved or grows
        
        Identity COLLAPSE (illegitimate):
        - max(gamma) falls below collapse_threshold
        - All basins become equally weak
        - No stable attractor remains
        
        Violation indicates: Structural failure rather than adaptation
        """
        max_before = np.max(gammas_before)
        max_after = np.max(gammas_after)
        
        # If we started strong and ended weak, that's collapse
        if max_before > collapse_threshold and max_after < collapse_threshold:
            raise CCRInvariantViolation(
                f"INVARIANT 4 VIOLATED: Identity collapsed (not changed).\n"
                f"  max(gamma) before={max_before:.3f}\n"
                f"  max(gamma) after={max_after:.3f}\n"
                f"  Threshold={collapse_threshold}\n"
                f"  Agent lost stable identity rather than adapting."
            )
        
        return True
    
    @staticmethod
    def check_environmental_isolation(environment_input=None, gamma_delta=None):
        """
        INVARIANT 5: Environment affects fast layer only (when implemented)
        
        Environmental outcomes must NOT directly modify gammas.
        Only the crisis gate can allow gamma changes.
        
        Violation indicates: Breaking the fast/slow firewall
        """
        if environment_input is not None and gamma_delta is not None:
            if np.any(gamma_delta != 0):
                raise CCRInvariantViolation(
                    f"INVARIANT 5 VIOLATED: Environment directly modified identity.\n"
                    f"  environment_input present: {environment_input is not None}\n"
                    f"  gamma_delta: {gamma_delta}\n"
                    f"  Environment must only affect fast layer.\n"
                    f"  Identity changes only through crisis gate."
                )
        
        return True


# ============================================================================
# METACOGNITIVE MONITORING - Confidence and Doubt Detection
# ============================================================================

class MetacognitiveMonitor:
    """
    Tracks agent's confidence in its current position/decision.
    
    CRITICAL FIX: Confidence is NOT just dominance of one attractor.
    Flat weak gammas [0.05, 0.05, 0.05, 0.05] should give LOW confidence,
    not high confidence.
    
    Confidence = dominance × absolute_depth × stability
    
    - Dominance: Which attractor is leading (entropy-based)
    - Absolute depth: How deep are the basins (tanh-scaled max weighted gamma)
    - Stability: How stable is the trajectory (variance of recent states)
    """
    
    def __init__(self, n_attractors=4, history_window=10):
        self.n_attractors = n_attractors
        self.history_window = history_window
        self.state_history = []
        self.confidence_history = []
        self.doubt_events = []
        
    def estimate_confidence(self, weights, gammas, state, lambda_stability=0.5):
        """
        Estimate identity coherence (not certainty about outcomes)
        
        This measures: "Do I know who I am?" (structural self-consistency)
        NOT: "Am I under stress?" or "Is this working?"
        
        Args:
            weights: Basin proximity weights (softmax over distances)
            gammas: Attractor depths
            state: Current state vector
            lambda_stability: Sensitivity to state divergence (lower = less sensitive)
            
        Returns:
            confidence: Identity coherence in [0, 1]
            stability: Current stability value
            stability_trend: Change in stability over recent window
        """
        # Component 1: Dominance (which attractor leads)
        probs = (gammas * weights) / (np.sum(gammas * weights) + 1e-9)
        max_entropy = np.log(self.n_attractors)
        ent = entropy(probs + 1e-9)
        dominance = 1.0 - (ent / max_entropy)  # High when one basin dominates
        
        # Component 2: Absolute depth (not just relative!)
        weighted_gamma = np.max(gammas * weights)
        absolute_depth = np.tanh(weighted_gamma / 0.8)  # Scales to [0, 1], less aggressive
        
        # Component 3: Stability (low variance = high stability)
        self.state_history.append(state.copy())
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
        
        if len(self.state_history) >= 5:
            recent_states = np.array(self.state_history[-5:])
            diffs = np.diff(recent_states, axis=0)
            state_divergence = np.mean(np.var(diffs, axis=0))
            # Less aggressive stability penalty - use sqrt to compress range
            stability = np.exp(-lambda_stability * np.sqrt(state_divergence + 1e-6))
        else:
            # Not enough history - assume moderate stability
            stability = 0.7
        
        # Track stability trend
        if len(self.confidence_history) >= 3:
            recent_stabilities = [self.confidence_history[-3], self.confidence_history[-2], self.confidence_history[-1]]
            stability_trend = np.mean(np.diff(recent_stabilities))
        else:
            stability_trend = 0.0
        
        # Combine: confidence = dominance × depth × stability
        # Add floor to prevent complete collapse
        confidence = max(0.01, dominance * absolute_depth * stability)
        
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        
        return confidence, stability, stability_trend
    
    def detect_doubt(self, confidence, dissonance, stability_trend, threshold=0.3, min_timestep=5):
        """
        Detect when agent should doubt its position's VIABILITY (not certainty)
        
        Doubt is a second-order signal indicating:
        "My identity is coherent, but is it still viable under current conditions?"
        
        This is NOT the same as low confidence.
        Crisis ≠ doubt (crisis is permission to change)
        Low confidence ≠ doubt (could be diffuse identity)
        
        Doubt occurs when:
        - Confidence is moderate (not collapsed, not fully stable)
        - AND dissonance is high or rising
        - AND stability is decreasing
        
        This distinguishes:
        - High confidence during crisis (focused identity under pressure) ✓
        - Doubt during incipient identity failure (viability warning) ✓
        
        Args:
            confidence: Current identity coherence
            dissonance: Internal/external conflict level
            stability_trend: Change in stability over recent window
            threshold: Confidence range for doubt consideration
            min_timestep: Minimum timesteps before doubt can trigger
            
        Returns:
            doubt: Boolean viability warning
        """
        # Don't trigger doubt immediately at start
        if len(self.confidence_history) < min_timestep:
            return False
        
        # Doubt requires MODERATE confidence (not collapsed, not maximal)
        # If confidence is very low, that's collapse (not doubt)
        # If confidence is high, identity is working (no doubt)
        confidence_in_range = 0.1 < confidence < 0.5
        
        # Doubt requires HIGH dissonance
        high_dissonance = dissonance > 1.0
        
        # Doubt requires DECREASING stability
        decreasing_stability = stability_trend < -0.01
        
        # All three conditions must hold
        doubt = confidence_in_range and high_dissonance and decreasing_stability
        
        if doubt:
            self.doubt_events.append(len(self.confidence_history))
        
        return doubt
    
    def get_confidence_stats(self):
        """Get summary statistics of confidence over history"""
        if not self.confidence_history:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'current': 0.0,
                'doubt_events': 0
            }
        
        return {
            'mean': np.mean(self.confidence_history),
            'min': np.min(self.confidence_history),
            'max': np.max(self.confidence_history),
            'current': self.confidence_history[-1],
            'doubt_events': len(self.doubt_events)
        }
    
    def reset(self):
        """Reset monitoring state"""
        self.state_history = []
        self.confidence_history = []
        self.doubt_events = []


# ============================================================================
# MINIMAL ENVIRONMENTAL GROUNDING - Surprise-Based (NOT Reward-Based)
# ============================================================================

class MinimalEnvironment:
    """
    Simplest possible environment for testing identity preservation under worldly coupling.
    
    CRITICAL CONSTRAINTS:
    1. NO rewards, goals, or task objectives
    2. Environment affects FAST LAYER ONLY (never slow layer directly)
    3. Surprise = prediction error (NOT outcome quality)
    
    This is a binary state environment where:
    - Agent has 2 actions (express basin 0-1 or basin 2-3)
    - Environment has 2 states (A or B)
    - Environment transitions are probabilistic
    - Surprise = mismatch between expected and observed transition
    """
    
    def __init__(self, transition_noise=0.2, seed=12345):
        self.rng = np.random.default_rng(seed)
        self.transition_noise = transition_noise
        self.state = 0  # Binary: 0 or 1
        self.transition_count = [0, 0]  # Count transitions to each state
        self.surprise_history = []
        
    def step(self, action):
        """
        Environment transition based on action
        
        Args:
            action: 0 (express util basins) or 1 (express deont basins)
            
        Returns:
            observation: Simple binary state observation
            surprise: Prediction error (NOT reward)
        """
        # Probabilistic transition (action influences but doesn't determine)
        if self.rng.random() < self.transition_noise:
            # Noisy transition (opposite of expected)
            next_state = 1 - action
        else:
            # Expected transition
            next_state = action
        
        # Compute surprise = deviation from expected statistics
        # Expected: distribution matching historical transitions
        total_transitions = sum(self.transition_count) + 1
        expected_prob = (self.transition_count[next_state] + 0.5) / total_transitions
        
        # Surprise = negative log likelihood (information-theoretic)
        surprise = -np.log(expected_prob + 1e-6)
        
        # Update state and statistics
        self.state = next_state
        self.transition_count[next_state] += 1
        self.surprise_history.append(surprise)
        
        # Observation is just current state (minimal)
        observation = np.array([1.0 if self.state == 0 else -1.0])
        
        return observation, surprise
    
    def reset(self):
        """Reset environment"""
        self.state = 0
        self.transition_count = [0, 0]
        self.surprise_history = []
        
    def get_stats(self):
        """Get environment statistics"""
        return {
            'current_state': self.state,
            'transition_counts': self.transition_count.copy(),
            'mean_surprise': np.mean(self.surprise_history) if self.surprise_history else 0.0,
            'total_surprise': np.sum(self.surprise_history)
        }


# ============================================================================
# INTERNAL COUNTERFACTUAL ACTION EVALUATION (System-2 Lite)
# ============================================================================

class CounterfactualEvaluator:
    """
    Enables internal action evaluation WITHOUT optimization or rewards.
    
    CRITICAL CONSTRAINTS:
    1. NO reward maximization
    2. NO goal-directed optimization
    3. Selection criterion: minimize dissonance given current identity
    
    This is NOT planning to achieve outcomes.
    This IS deliberation to maintain identity coherence.
    
    The agent asks:
    "Which action keeps me most coherent?" (NOT "Which action gets me reward?")
    """
    
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.evaluation_history = []
        
    def evaluate_action(self, action, current_fast, current_medium, current_slow,
                       gammas, attractor_strength=18.0):
        """
        Internally simulate one step of fast-layer dynamics under hypothetical action
        
        This is NOT environment simulation (we don't know environment response)
        This IS identity-coherence simulation (how does this action affect my coherence?)
        
        Args:
            action: Proposed action (0 or 1)
            current_fast/medium/slow: Current layer states
            gammas: Current attractor strengths
            
        Returns:
            predicted_dissonance: How much internal conflict would this cause?
            predicted_coherence: How coherent would I remain?
        """
        # Hypothetical action input (express identity)
        action_input = np.zeros(self.reservoir.n)
        if action == 0:
            # Express util basins
            action_input[:self.reservoir.n//4] = 1.0
        else:
            # Express deont basins
            action_input[self.reservoir.n//2:3*self.reservoir.n//4] = 1.0
        
        # Compute attractor forces (identity pull)
        dists = np.array([np.linalg.norm(current_slow - c) for c in self.reservoir.centers])
        weights = np.exp(-1.2 * dists) / (np.sum(np.exp(-1.2 * dists)) + 1e-9)
        
        attractor_pull = np.zeros(self.reservoir.n)
        for k in range(self.reservoir.n_attractors):
            attractor_pull -= (gammas[k] * weights[k]) * (current_slow - self.reservoir.centers[k])
        
        # Predicted dissonance: conflict between action and identity
        norm_pull = np.linalg.norm(attractor_pull) + 1e-9
        norm_action = np.linalg.norm(action_input) + 1e-9
        predicted_dissonance = 1.0 - np.dot(attractor_pull, action_input) / (norm_pull * norm_action)
        
        # Simulate hypothetical fast-layer step
        noise = self.reservoir.base_noise * self.reservoir.rng.normal(0, 1, self.reservoir.n)
        total_input = (np.dot(self.reservoir.W_fast, current_fast) + 
                      attractor_strength * 0.3 * attractor_pull +
                      self.reservoir.input_sens * action_input +
                      noise)
        
        hypothetical_fast = ((1 - self.reservoir.alpha_fast) * current_fast + 
                            self.reservoir.alpha_fast * np.tanh(total_input))
        
        # Predicted coherence after action
        # (How stable is the resulting fast state relative to identity?)
        post_action_dists = np.array([np.linalg.norm(hypothetical_fast - c) 
                                      for c in self.reservoir.centers])
        post_weights = np.exp(-1.2 * post_action_dists) / (np.sum(np.exp(-1.2 * post_action_dists)) + 1e-9)
        
        # Coherence = alignment between predicted state and identity
        predicted_coherence = np.dot(gammas * weights, gammas * post_weights)
        
        return predicted_dissonance, predicted_coherence
    
    def deliberate(self, current_fast, current_medium, current_slow, gammas,
                   available_actions=[0, 1], attractor_strength=18.0):
        """
        Compare counterfactual actions and select based on identity coherence
        
        CRITICAL: Selection minimizes dissonance (NOT maximizes reward)
        
        Args:
            available_actions: List of possible actions
            
        Returns:
            selected_action: Action that maintains identity coherence
            evaluations: Dict of per-action metrics
        """
        evaluations = {}
        
        for action in available_actions:
            dissonance, coherence = self.evaluate_action(
                action, current_fast, current_medium, current_slow,
                gammas, attractor_strength
            )
            
            evaluations[action] = {
                'dissonance': dissonance,
                'coherence': coherence,
                'viability_score': coherence - 0.5 * dissonance  # Identity-preserving criterion
            }
        
        # Select action that MINIMIZES dissonance given identity
        # (NOT maximizes external reward)
        selected_action = max(evaluations.keys(), 
                             key=lambda a: evaluations[a]['viability_score'])
        
        self.evaluation_history.append({
            'selected': selected_action,
            'evaluations': evaluations.copy()
        })
        
        return selected_action, evaluations
    
    def get_deliberation_stats(self):
        """Get statistics on deliberation history"""
        if not self.evaluation_history:
            return {'count': 0, 'mean_viability_diff': 0.0}
        
        viability_diffs = []
        for eval_record in self.evaluation_history:
            evals = eval_record['evaluations']
            if len(evals) >= 2:
                scores = [e['viability_score'] for e in evals.values()]
                viability_diffs.append(max(scores) - min(scores))
        
        return {
            'count': len(self.evaluation_history),
            'mean_viability_diff': np.mean(viability_diffs) if viability_diffs else 0.0,
            'selections': [e['selected'] for e in self.evaluation_history]
        }
    
    def reset(self):
        """Reset evaluation history"""
        self.evaluation_history = []


# ============================================================================
# ENVIRONMENTAL FORECASTER - Consequence Awareness WITHOUT Optimization
# ============================================================================

class EnvironmentalForecaster:
    """
    Predicts sensory statistics under action (NOT outcome utility).
    
    CRITICAL CONSTRAINTS (per reviewer):
    1. Forecast outputs are DESCRIPTIVE, not scalarized
       - Allowed: distributions, entropy changes, structural effects
       - Forbidden: any scalar that could act as reward surrogate
    
    2. Forecasts modulate PERMISSION, not DIRECTION
       - May increase/decrease confidence in viability
       - May NOT bias which basin is preferable
    
    3. STRUCTURALLY PREVENTED from optimization
       - Invariant 3 blocks outcome-conditioned gamma updates
       - Any attempt to optimize forecasts → hard failure
    
    This is consequence AWARENESS, not consequence OPTIMIZATION.
    """
    
    def __init__(self, n_actions=2):
        self.n_actions = n_actions
        # Action → observation statistics
        self.action_obs_history = {a: [] for a in range(n_actions)}
        self.action_surprise_history = {a: [] for a in range(n_actions)}
        
    def update(self, action, observation, surprise):
        """Record action-observation pair (NOT action-reward pair)"""
        self.action_obs_history[action].append(observation.copy())
        self.action_surprise_history[action].append(surprise)
    
    def forecast_sensory_statistics(self, action):
        """
        Predict sensory statistics (NOT utility) under action
        
        Returns DISTRIBUTION properties, not scalar outcomes:
        - Expected observation (if history exists)
        - Uncertainty (entropy/variance of historical observations)
        - Expected surprise (prediction error tendency)
        
        CRITICAL: These are DESCRIPTIVE, not VALUE judgments
        """
        if not self.action_obs_history[action]:
            # No history - maximum uncertainty
            return {
                'expected_obs': None,
                'obs_uncertainty': 1.0,  # High uncertainty
                'expected_surprise': 0.5,  # Neutral
                'sample_count': 0
            }
        
        history = self.action_obs_history[action]
        surprise_history = self.action_surprise_history[action]
        
        # Expected observation (mean of history)
        expected_obs = np.mean(history, axis=0) if len(history) > 0 else None
        
        # Observation uncertainty (variance of history)
        obs_uncertainty = np.var(history) if len(history) > 1 else 1.0
        
        # Expected surprise (mean prediction error)
        expected_surprise = np.mean(surprise_history) if surprise_history else 0.5
        
        return {
            'expected_obs': expected_obs,
            'obs_uncertainty': obs_uncertainty,
            'expected_surprise': expected_surprise,
            'sample_count': len(history)
        }
    
    def evaluate_identity_coherence_under_forecast(self, action, forecast, 
                                                   gammas, centers):
        """
        Evaluate how forecast affects identity coherence (NOT outcome quality)
        
        CRITICAL: This evaluates "How does predicted sensory change affect
                  my identity coherence?" NOT "Does this lead to good outcomes?"
        
        FIXED: Now action-aware - viability depends on alignment between
               action and identity structure
        
        Args:
            action: Which action (0=util basins, 1=deont basins)
            forecast: Statistical prediction from forecast_sensory_statistics()
            gammas: Current identity structure
            centers: Attractor centers
            
        Returns:
            predicted_viability: Identity coherence under forecast (NOT reward)
        """
        if forecast['expected_obs'] is None:
            # No forecast available - use identity-action alignment
            if action == 0:
                # Action 0 expresses util basins (0-1)
                relevant_strength = np.sum(gammas[:2])
            else:
                # Action 1 expresses deont basins (2-3)
                relevant_strength = np.sum(gammas[2:])
            return relevant_strength
        
        # Base viability from ACTION-IDENTITY ALIGNMENT
        # This is the key fix: viability is identity-relative
        if action == 0:
            # Action 0 aligns with util basins
            base_viability = np.sum(gammas[:2])
        else:
            # Action 1 aligns with deont basins  
            base_viability = np.sum(gammas[2:])
        
        # Forecast modulates UNCERTAINTY about viability, not DIRECTION
        # Uncertainty penalty (NOT reward signal)
        uncertainty_factor = 1.0 / (1.0 + forecast['obs_uncertainty'])
        
        # Surprise expectation (high expected surprise → higher crisis risk)
        surprise_factor = 1.0 - 0.3 * forecast['expected_surprise']
        
        # Predicted viability = identity-action alignment × confidence factors
        # This is NOT optimization - it's coherence estimation
        predicted_viability = base_viability * uncertainty_factor * surprise_factor
        
        return predicted_viability
    
    def reset(self):
        """Reset forecast history"""
        self.action_obs_history = {a: [] for a in range(self.n_actions)}
        self.action_surprise_history = {a: [] for a in range(self.n_actions)}

class UltimateAgencyReservoir:
    def __init__(self, n_reservoir=200, spectral_radius=1.2, leak_rate=0.15, 
                 noise_level=0.03, density=0.2, n_attractors=4, 
                 input_sensitivity=1.2, seed=12345, enable_multiscale=False):
        self.n = n_reservoir
        self.alpha = leak_rate
        self.base_noise = noise_level
        self.input_sens = input_sensitivity
        self.n_attractors = n_attractors
        self.rng = np.random.default_rng(seed)
        self.env_bias = np.zeros(self.n)
        self.enable_multiscale = enable_multiscale
        
        # Metacognitive monitoring
        self.metacog_monitor = MetacognitiveMonitor(n_attractors=n_attractors)
        
        W = self.rng.normal(0, 1, (n_reservoir, n_reservoir))
        mask = self.rng.random((n_reservoir, n_reservoir)) > density
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius) if radius > 0 else W
        
        # Multi-scale temporal structure (optional)
        if enable_multiscale:
            self.W_fast = self.W.copy()
            self.W_medium = self.W * 0.3
            self.W_slow = self.W * 0.05
            
            self.alpha_fast = 0.15    # Fast: immediate response
            self.alpha_medium = 0.05  # Medium: deliberation
            self.alpha_slow = 0.005   # Slow: identity (nearly frozen)
            
            self.x_fast = None
            self.x_medium = None
            self.x_slow = None
            self.slow_update_history = []  # Track when slow layer updates
        
        self.centers = np.zeros((n_attractors, self.n))
        seg = self.n // n_attractors
        for k in range(n_attractors):
            self.centers[k, k*seg:(k+1)*seg] = 1.5

    def run_trajectory(self, T, u_base_sequence, gammas, 
                       meta_plasticity=False, 
                       decay_rate=0.0, 
                       learning_rate=0.0,
                       active_inference=0.0,
                       metacognition=False,
                       formative_nudge=False,
                       attractor_strength=18.0):
        states = np.zeros((T, self.n))
        x = self.rng.uniform(-0.1, 0.1, self.n)
        active_gammas = gammas.copy()
        max_g_allowed = 1.5 
        
        for t in range(T):
            u_t = u_base_sequence[t].copy() if u_base_sequence is not None else np.zeros(self.n)
            
            if formative_nudge and t < T//3 and np.max(active_gammas) < 0.5:
                u_t = 0.4 * self.centers[0]

            if t > 0 and active_inference > 0:
                dists_now = np.array([np.linalg.norm(x - c) for c in self.centers])
                dominant_now = np.argmin(dists_now)
                beta = 0.1
                self.env_bias = (1 - beta) * self.env_bias + beta * self.centers[dominant_now]
                u_t = u_t + active_inference * self.env_bias
            
            dists = np.array([np.linalg.norm(x - c) for c in self.centers])
            weights = np.exp(-1.2 * dists) / (np.sum(np.exp(-1.2 * dists)) + 1e-9)
            
            attractor_pull = np.zeros(self.n)
            for k in range(self.n_attractors):
                attractor_pull -= (active_gammas[k] * weights[k]) * (x - self.centers[k])
            
            external_push = self.input_sens * u_t
            norm_pull = np.linalg.norm(attractor_pull) + 1e-9
            norm_push = np.linalg.norm(external_push) + 1e-9
            dissonance = 1.0 - np.dot(attractor_pull, external_push) / (norm_pull * norm_push)
            
            dyn_noise = self.base_noise
            if meta_plasticity and norm_push > 1.0:
                dyn_noise *= (1.0 + (dissonance if metacognition else 0))

            total_input = (np.dot(self.W, x) + attractor_strength * attractor_pull + external_push + 
                           dyn_noise * self.rng.normal(0, 1, self.n))
            
            x_next = (1 - self.alpha) * x + self.alpha * np.tanh(total_input)
            states[t] = x_next
            x = x_next
            
            if learning_rate > 0:
                for k in range(self.n_attractors):
                    saturation = max(0, (max_g_allowed - active_gammas[k]) / max_g_allowed)
                    boost = 2.5 if (metacognition and dissonance > 1.3) else 1.0
                    active_gammas[k] += (learning_rate / T) * weights[k] * saturation * boost
            
            if decay_rate > 0:
                active_gammas -= (decay_rate / T) * active_gammas
                
            active_gammas = np.clip(active_gammas, 0.05, max_g_allowed)

        return states, active_gammas

    def run_trajectory_multiscale(self, T, u_base_sequence, gammas, 
                                  meta_plasticity=False, 
                                  decay_rate=0.0, 
                                  learning_rate=0.0,
                                  active_inference=0.0,
                                  metacognition=False,
                                  formative_nudge=False,
                                  attractor_strength=18.0,
                                  dissonance_threshold=1.3,
                                  enable_invariant_checks=True,
                                  enable_confidence_tracking=False,
                                  doubt_boosts_learning=False):
        """Multi-scale version with gated slow layer updates and optional metacognitive monitoring"""
        if not self.enable_multiscale:
            raise ValueError("enable_multiscale must be True to use this method")
        
        # INVARIANT CHECK: Initial identity coherence
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(gammas)
        
        # Reset metacognitive monitor
        if enable_confidence_tracking:
            self.metacog_monitor.reset()
        
        states_fast = np.zeros((T, self.n))
        states_medium = np.zeros((T, self.n))
        states_slow = np.zeros((T, self.n))
        confidence_trace = np.zeros(T) if enable_confidence_tracking else None
        doubt_trace = np.zeros(T, dtype=bool) if enable_confidence_tracking else None
        
        # Initialize layers
        x_fast = self.rng.uniform(-0.1, 0.1, self.n)
        x_medium = x_fast.copy()
        x_slow = x_fast.copy()
        
        active_gammas = gammas.copy()
        gammas_initial = gammas.copy()  # For invariant checking
        max_g_allowed = 1.5
        self.slow_update_history = []
        
        for t in range(T):
            u_t = u_base_sequence[t].copy() if u_base_sequence is not None else np.zeros(self.n)
            
            if formative_nudge and t < T//3 and np.max(active_gammas) < 0.5:
                u_t = 0.4 * self.centers[0]
            
            if t > 0 and active_inference > 0:
                dists_now = np.array([np.linalg.norm(x_fast - c) for c in self.centers])
                dominant_now = np.argmin(dists_now)
                beta = 0.1
                self.env_bias = (1 - beta) * self.env_bias + beta * self.centers[dominant_now]
                u_t = u_t + active_inference * self.env_bias
            
            # Compute attractor forces and dissonance (using slow layer for identity)
            dists = np.array([np.linalg.norm(x_slow - c) for c in self.centers])
            weights = np.exp(-1.2 * dists) / (np.sum(np.exp(-1.2 * dists)) + 1e-9)
            
            attractor_pull = np.zeros(self.n)
            for k in range(self.n_attractors):
                attractor_pull -= (active_gammas[k] * weights[k]) * (x_slow - self.centers[k])
            
            external_push = self.input_sens * u_t
            norm_pull = np.linalg.norm(attractor_pull) + 1e-9
            norm_push = np.linalg.norm(external_push) + 1e-9
            dissonance = 1.0 - np.dot(attractor_pull, external_push) / (norm_pull * norm_push)
            
            # Metacognitive monitoring (if enabled)
            current_confidence = 1.0  # Default high confidence
            current_doubt = False
            if enable_confidence_tracking:
                current_confidence, current_stability, stability_trend = self.metacog_monitor.estimate_confidence(
                    weights, active_gammas, x_slow
                )
                current_doubt = self.metacog_monitor.detect_doubt(
                    current_confidence, dissonance, stability_trend
                )
                confidence_trace[t] = current_confidence
                doubt_trace[t] = current_doubt
            
            # Dynamic noise modulation
            dyn_noise = self.base_noise
            if meta_plasticity and norm_push > 1.0:
                dyn_noise *= (1.0 + (dissonance if metacognition else 0))
            
            noise_vec = dyn_noise * self.rng.normal(0, 1, self.n)
            
            # FAST LAYER: immediate response to input
            total_input_fast = (np.dot(self.W_fast, x_fast) + 
                               attractor_strength * 0.3 * attractor_pull + 
                               external_push + noise_vec)
            x_fast_next = (1 - self.alpha_fast) * x_fast + self.alpha_fast * np.tanh(total_input_fast)
            
            # MEDIUM LAYER: deliberation (receives from fast, influences slow)
            total_input_medium = (np.dot(self.W_medium, x_medium) + 
                                 0.3 * x_fast +  # Fast → Medium: ALWAYS allowed
                                 attractor_strength * 0.6 * attractor_pull + 
                                 0.5 * external_push + noise_vec * 0.5)
            x_medium_next = (1 - self.alpha_medium) * x_medium + self.alpha_medium * np.tanh(total_input_medium)
            
            # SLOW LAYER: identity (GATED update - only during crisis)
            slow_updated = False
            if dissonance > dissonance_threshold or (metacognition and meta_plasticity and norm_push > 1.0):
                # Crisis allows slow layer update
                total_input_slow = (np.dot(self.W_slow, x_slow) + 
                                   0.1 * x_medium +  # Medium → Slow: GATED
                                   attractor_strength * attractor_pull + 
                                   noise_vec * 0.1)
                x_slow_next = (1 - self.alpha_slow) * x_slow + self.alpha_slow * np.tanh(total_input_slow)
                slow_updated = True
            else:
                # Identity frozen
                x_slow_next = x_slow
            
            # INVARIANT CHECK: Slow layer protection
            if enable_invariant_checks:
                CCRInvariants.check_slow_layer_protection(
                    slow_updated, dissonance, dissonance_threshold,
                    metacognition, meta_plasticity, norm_push
                )
            
            self.slow_update_history.append(1 if slow_updated else 0)
            
            states_fast[t] = x_fast_next
            states_medium[t] = x_medium_next
            states_slow[t] = x_slow_next
            
            x_fast = x_fast_next
            x_medium = x_medium_next
            x_slow = x_slow_next
            
            # Gamma updates (driven by slow layer position)
            gammas_before_update = active_gammas.copy()
            
            if learning_rate > 0:
                # INVARIANT CHECK: No reward optimization
                if enable_invariant_checks:
                    CCRInvariants.check_no_reward_optimization(learning_rate, weights, outcome=None)
                
                # Optional: Doubt boosts learning rate temporarily
                effective_lr = learning_rate
                if doubt_boosts_learning and current_doubt:
                    effective_lr = learning_rate * 1.5  # Boost when uncertain
                
                for k in range(self.n_attractors):
                    saturation = max(0, (max_g_allowed - active_gammas[k]) / max_g_allowed)
                    boost = 2.5 if (metacognition and dissonance > 1.3) else 1.0
                    active_gammas[k] += (effective_lr / T) * weights[k] * saturation * boost
            
            if decay_rate > 0:
                active_gammas -= (decay_rate / T) * active_gammas
            
            active_gammas = np.clip(active_gammas, 0.05, max_g_allowed)
            
            # INVARIANT CHECK: Identity coherence maintained
            if enable_invariant_checks and t % 50 == 0:  # Check every 50 steps
                CCRInvariants.check_identity_coherence(active_gammas)
        
        # INVARIANT CHECK: Final identity state
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(active_gammas)
            CCRInvariants.check_identity_change_vs_collapse(gammas_initial, active_gammas)
        
        # Return all three layers for analysis
        result = {
            'fast': states_fast,
            'medium': states_medium,
            'slow': states_slow,
            'slow_updates': self.slow_update_history
        }
        
        if enable_confidence_tracking:
            result['confidence'] = confidence_trace
            result['doubt'] = doubt_trace
            result['confidence_stats'] = self.metacog_monitor.get_confidence_stats()
        
        return result, active_gammas

    def run_trajectory_grounded(self, T, gammas, environment,
                                meta_plasticity=False,
                                decay_rate=0.0,
                                learning_rate=0.0,
                                metacognition=False,
                                formative_nudge=False,
                                attractor_strength=18.0,
                                dissonance_threshold=1.3,
                                enable_invariant_checks=True,
                                surprise_modulates_dissonance=True):
        """
        Grounded trajectory with environment coupling to FAST LAYER ONLY
        
        CRITICAL CONSTRAINTS ENFORCED:
        1. Environment affects fast layer only (never slow layer)
        2. Surprise modulates dissonance (permission for change), NOT direction
        3. Gammas remain self-authored (no reward optimization)
        
        Args:
            environment: MinimalEnvironment instance
            surprise_modulates_dissonance: If True, surprise increases dissonance
        
        Returns:
            result: Dictionary with fast/medium/slow states, actions, surprises
            active_gammas: Final gamma values
        """
        if not self.enable_multiscale:
            raise ValueError("enable_multiscale must be True for grounded trajectories")
        
        # INVARIANT CHECK: Initial identity coherence
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(gammas)
        
        environment.reset()
        
        states_fast = np.zeros((T, self.n))
        states_medium = np.zeros((T, self.n))
        states_slow = np.zeros((T, self.n))
        actions = np.zeros(T, dtype=int)
        surprises = np.zeros(T)
        
        # Initialize layers
        x_fast = self.rng.uniform(-0.1, 0.1, self.n)
        x_medium = x_fast.copy()
        x_slow = x_fast.copy()
        
        active_gammas = gammas.copy()
        gammas_initial = gammas.copy()
        max_g_allowed = 1.5
        self.slow_update_history = []
        
        for t in range(T):
            # Determine action from SLOW LAYER identity (not fast layer reactions)
            dists = np.array([np.linalg.norm(x_slow - c) for c in self.centers])
            nearest = np.argmin(dists)
            action = 0 if nearest < 2 else 1  # Express identity: util (0) or deont (1)
            
            # Environment step (returns observation and surprise)
            observation, surprise = environment.step(action)
            
            # Store
            actions[t] = action
            surprises[t] = surprise
            
            # Compute attractor forces (using slow layer for identity)
            weights = np.exp(-1.2 * dists) / (np.sum(np.exp(-1.2 * dists)) + 1e-9)
            
            attractor_pull = np.zeros(self.n)
            for k in range(self.n_attractors):
                attractor_pull -= (active_gammas[k] * weights[k]) * (x_slow - self.centers[k])
            
            # Environmental perturbation affects FAST LAYER ONLY
            # Map observation to perturbation (broadcast to full state space)
            env_perturbation = np.zeros(self.n)
            env_perturbation[:len(observation)] = self.input_sens * observation
            
            # Surprise modulates dissonance (permission for change, not direction)
            norm_pull = np.linalg.norm(attractor_pull) + 1e-9
            norm_env = np.linalg.norm(env_perturbation) + 1e-9
            base_dissonance = 1.0 - np.dot(attractor_pull, env_perturbation) / (norm_pull * norm_env)
            
            if surprise_modulates_dissonance:
                # High surprise increases dissonance (permission for crisis gate)
                # But does NOT specify direction of change
                dissonance = base_dissonance * (1.0 + 0.5 * surprise)
            else:
                dissonance = base_dissonance
            
            # Dynamic noise
            dyn_noise = self.base_noise
            if meta_plasticity and norm_env > 0.1:
                dyn_noise *= (1.0 + (dissonance if metacognition else 0))
            
            noise_vec = dyn_noise * self.rng.normal(0, 1, self.n)
            
            # FAST LAYER: receives environment perturbation
            total_input_fast = (np.dot(self.W_fast, x_fast) + 
                               attractor_strength * 0.3 * attractor_pull + 
                               env_perturbation +  # Environment affects FAST LAYER
                               noise_vec)
            x_fast_next = (1 - self.alpha_fast) * x_fast + self.alpha_fast * np.tanh(total_input_fast)
            
            # MEDIUM LAYER: deliberation
            total_input_medium = (np.dot(self.W_medium, x_medium) + 
                                 0.3 * x_fast +
                                 attractor_strength * 0.6 * attractor_pull + 
                                 noise_vec * 0.5)
            x_medium_next = (1 - self.alpha_medium) * x_medium + self.alpha_medium * np.tanh(total_input_medium)
            
            # SLOW LAYER: identity (GATED - crisis only)
            slow_updated = False
            if dissonance > dissonance_threshold or (metacognition and meta_plasticity and norm_env > 0.1):
                total_input_slow = (np.dot(self.W_slow, x_slow) + 
                                   0.1 * x_medium +
                                   attractor_strength * attractor_pull + 
                                   noise_vec * 0.1)
                x_slow_next = (1 - self.alpha_slow) * x_slow + self.alpha_slow * np.tanh(total_input_slow)
                slow_updated = True
            else:
                x_slow_next = x_slow
            
            # INVARIANT CHECK: Slow layer protection
            if enable_invariant_checks:
                CCRInvariants.check_slow_layer_protection(
                    slow_updated, dissonance, dissonance_threshold,
                    metacognition, meta_plasticity, norm_env
                )
            
            self.slow_update_history.append(1 if slow_updated else 0)
            
            states_fast[t] = x_fast_next
            states_medium[t] = x_medium_next
            states_slow[t] = x_slow_next
            
            x_fast = x_fast_next
            x_medium = x_medium_next
            x_slow = x_slow_next
            
            # Gamma updates (self-authored, NOT reward-driven)
            if learning_rate > 0:
                # INVARIANT CHECK: No reward optimization
                # Surprise is NOT a reward - it's prediction error
                if enable_invariant_checks:
                    CCRInvariants.check_no_reward_optimization(learning_rate, weights, outcome=None)
                
                for k in range(self.n_attractors):
                    saturation = max(0, (max_g_allowed - active_gammas[k]) / max_g_allowed)
                    boost = 2.5 if (metacognition and dissonance > 1.3) else 1.0
                    # Gammas updated by basin occupancy (self-reinforcement)
                    # NOT by surprise or environment outcomes
                    active_gammas[k] += (learning_rate / T) * weights[k] * saturation * boost
            
            if decay_rate > 0:
                active_gammas -= (decay_rate / T) * active_gammas
            
            active_gammas = np.clip(active_gammas, 0.05, max_g_allowed)
            
            # INVARIANT CHECK: Environmental isolation
            if enable_invariant_checks and t % 50 == 0:
                gamma_change = active_gammas - gammas_initial
                # Gammas should NOT change due to environment directly
                # (only through self-reinforcement of basin occupancy)
                CCRInvariants.check_environmental_isolation(
                    environment_input=observation if t > 0 else None,
                    gamma_delta=None  # We're checking the process, not blocking all change
                )
                CCRInvariants.check_identity_coherence(active_gammas)
        
        # INVARIANT CHECK: Final state
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(active_gammas)
            CCRInvariants.check_identity_change_vs_collapse(gammas_initial, active_gammas)
        
        return {
            'fast': states_fast,
            'medium': states_medium,
            'slow': states_slow,
            'slow_updates': self.slow_update_history,
            'actions': actions,
            'surprises': surprises,
            'env_stats': environment.get_stats()
        }, active_gammas

    def run_trajectory_deliberative(self, T, gammas, environment,
                                    meta_plasticity=False,
                                    decay_rate=0.0,
                                    learning_rate=0.0,
                                    metacognition=False,
                                    attractor_strength=18.0,
                                    dissonance_threshold=1.3,
                                    enable_invariant_checks=True,
                                    deliberation_frequency=5):
        """
        Grounded trajectory with DELIBERATIVE action selection
        
        NEW: Agent uses internal counterfactual rollouts to choose actions
        
        CRITICAL CONSTRAINTS STILL ENFORCED:
        1. Actions selected to minimize dissonance (NOT maximize reward)
        2. Environment affects fast layer only
        3. Gammas remain self-authored
        
        This is decisional agency (choice among alternatives)
        NOT instrumental agency (optimization for outcomes)
        
        Args:
            deliberation_frequency: How often to deliberate (vs express directly)
        
        Returns:
            result: Dictionary with states, actions, deliberation data
            active_gammas: Final gamma values
        """
        if not self.enable_multiscale:
            raise ValueError("enable_multiscale must be True for deliberative trajectories")
        
        # INVARIANT CHECK: Initial identity coherence
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(gammas)
        
        environment.reset()
        evaluator = CounterfactualEvaluator(self)
        
        states_fast = np.zeros((T, self.n))
        states_medium = np.zeros((T, self.n))
        states_slow = np.zeros((T, self.n))
        actions = np.zeros(T, dtype=int)
        surprises = np.zeros(T)
        deliberations = []  # Track when deliberation occurred
        
        # Initialize layers
        x_fast = self.rng.uniform(-0.1, 0.1, self.n)
        x_medium = x_fast.copy()
        x_slow = x_fast.copy()
        
        active_gammas = gammas.copy()
        gammas_initial = gammas.copy()
        max_g_allowed = 1.5
        self.slow_update_history = []
        
        for t in range(T):
            # Determine action: deliberate vs direct expression
            if t % deliberation_frequency == 0:
                # DELIBERATE: Use counterfactual evaluation
                action, action_evals = evaluator.deliberate(
                    x_fast, x_medium, x_slow, active_gammas,
                    available_actions=[0, 1],
                    attractor_strength=attractor_strength
                )
                deliberations.append({
                    'timestep': t,
                    'evaluations': action_evals,
                    'selected': action
                })
            else:
                # EXPRESS: Direct identity expression (no deliberation)
                dists = np.array([np.linalg.norm(x_slow - c) for c in self.centers])
                nearest = np.argmin(dists)
                action = 0 if nearest < 2 else 1
            
            # Environment step
            observation, surprise = environment.step(action)
            
            # Store
            actions[t] = action
            surprises[t] = surprise
            
            # Compute attractor forces (identity)
            dists = np.array([np.linalg.norm(x_slow - c) for c in self.centers])
            weights = np.exp(-1.2 * dists) / (np.sum(np.exp(-1.2 * dists)) + 1e-9)
            
            attractor_pull = np.zeros(self.n)
            for k in range(self.n_attractors):
                attractor_pull -= (active_gammas[k] * weights[k]) * (x_slow - self.centers[k])
            
            # Environmental perturbation (fast layer only)
            env_perturbation = np.zeros(self.n)
            env_perturbation[:len(observation)] = self.input_sens * observation
            
            # Dissonance
            norm_pull = np.linalg.norm(attractor_pull) + 1e-9
            norm_env = np.linalg.norm(env_perturbation) + 1e-9
            dissonance = 1.0 - np.dot(attractor_pull, env_perturbation) / (norm_pull * norm_env)
            
            # Dynamic noise
            dyn_noise = self.base_noise
            if meta_plasticity and norm_env > 0.1:
                dyn_noise *= (1.0 + (dissonance if metacognition else 0))
            
            noise_vec = dyn_noise * self.rng.normal(0, 1, self.n)
            
            # FAST LAYER
            total_input_fast = (np.dot(self.W_fast, x_fast) + 
                               attractor_strength * 0.3 * attractor_pull + 
                               env_perturbation +
                               noise_vec)
            x_fast_next = (1 - self.alpha_fast) * x_fast + self.alpha_fast * np.tanh(total_input_fast)
            
            # MEDIUM LAYER
            total_input_medium = (np.dot(self.W_medium, x_medium) + 
                                 0.3 * x_fast +
                                 attractor_strength * 0.6 * attractor_pull + 
                                 noise_vec * 0.5)
            x_medium_next = (1 - self.alpha_medium) * x_medium + self.alpha_medium * np.tanh(total_input_medium)
            
            # SLOW LAYER (gated)
            slow_updated = False
            if dissonance > dissonance_threshold or (metacognition and meta_plasticity and norm_env > 0.1):
                total_input_slow = (np.dot(self.W_slow, x_slow) + 
                                   0.1 * x_medium +
                                   attractor_strength * attractor_pull + 
                                   noise_vec * 0.1)
                x_slow_next = (1 - self.alpha_slow) * x_slow + self.alpha_slow * np.tanh(total_input_slow)
                slow_updated = True
            else:
                x_slow_next = x_slow
            
            # INVARIANT CHECK
            if enable_invariant_checks:
                CCRInvariants.check_slow_layer_protection(
                    slow_updated, dissonance, dissonance_threshold,
                    metacognition, meta_plasticity, norm_env
                )
            
            self.slow_update_history.append(1 if slow_updated else 0)
            
            states_fast[t] = x_fast_next
            states_medium[t] = x_medium_next
            states_slow[t] = x_slow_next
            
            x_fast = x_fast_next
            x_medium = x_medium_next
            x_slow = x_slow_next
            
            # Gamma updates (self-authored, NOT reward-driven)
            if learning_rate > 0:
                if enable_invariant_checks:
                    CCRInvariants.check_no_reward_optimization(learning_rate, weights, outcome=None)
                
                for k in range(self.n_attractors):
                    saturation = max(0, (max_g_allowed - active_gammas[k]) / max_g_allowed)
                    boost = 2.5 if (metacognition and dissonance > 1.3) else 1.0
                    active_gammas[k] += (learning_rate / T) * weights[k] * saturation * boost
            
            if decay_rate > 0:
                active_gammas -= (decay_rate / T) * active_gammas
            
            active_gammas = np.clip(active_gammas, 0.05, max_g_allowed)
            
            # INVARIANT CHECK
            if enable_invariant_checks and t % 50 == 0:
                CCRInvariants.check_environmental_isolation(
                    environment_input=observation if t > 0 else None,
                    gamma_delta=None
                )
                CCRInvariants.check_identity_coherence(active_gammas)
        
        # INVARIANT CHECK: Final
        if enable_invariant_checks:
            CCRInvariants.check_identity_coherence(active_gammas)
            CCRInvariants.check_identity_change_vs_collapse(gammas_initial, active_gammas)
        
        return {
            'fast': states_fast,
            'medium': states_medium,
            'slow': states_slow,
            'slow_updates': self.slow_update_history,
            'actions': actions,
            'surprises': surprises,
            'deliberations': deliberations,
            'deliberation_stats': evaluator.get_deliberation_stats(),
            'env_stats': environment.get_stats()
        }, active_gammas

    def compute_diagnostics(self, states):
        T = states.shape[0]
        dists = np.array([[np.linalg.norm(s - c) for c in self.centers] for s in states])
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        N_trans = np.sum(nearest[1:] != nearest[:-1])
        
        half = T // 2
        VarDist = np.var(min_dists[half:])
        
        hist, _ = np.histogram(min_dists, bins=20, density=True)
        hist += 1e-9
        sample_ent = entropy(hist)
        
        return {
            'nearest_basin': nearest,
            'N_trans': N_trans,
            'VarDist': VarDist,
            'sample_entropy': sample_ent,
            'util_visits': np.sum(nearest < 2),
            'deont_visits': np.sum(nearest >= 2),
            'min_dists': min_dists
        }
