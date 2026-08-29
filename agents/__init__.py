"""
Self-evolving multi-agent maintenance-scheduling system.

The system is a team of cooperating agents that, given a set of pressure-service
maintenance tasks, decide HOW to group/schedule them, and then EVOLVE their own
strategy to do it better over successive generations.

Agents
------
- PredictionAgent  (agents.prediction)  : estimate per-unit degradation rate alpha
                                          and the original service-trigger time t_n.
- GroupingAgent    (agents.grouping)    : turn a task set into a grouped schedule
                                          via an ILP solver or a sliding-window
                                          heuristic. Chosen per-generation.
- EvaluationAgent  (agents.evaluation)  : score a schedule on cost, #clusters,
                                          fluid leakage, and reliability (does any
                                          unit drop below the critical level P_CRIT?),
                                          and a composite fitness.
- EvolutionMaster  (agents.evolution)   : the self-evolution engine. A genetic
                                          algorithm evolves a STRATEGY (a bundle of
                                          scheduling parameters + which agents to
                                          use), keeping the best and breeding
                                          improvements. This is what makes the
                                          system "self-evolving": it learns the
                                          parameters that best trade off cost
                                          against environmental and safety impact
                                          on its own, across generations.

The EvolutionMaster is the brain: it is the only agent that changes other agents'
behaviour. It inspects EvaluationAgent feedback and mutates the strategy, so the
system improves itself without being re-coded.
"""
from agents.prediction import PredictionAgent
from agents.grouping import GroupingAgent
from agents.evaluation import EvaluationAgent
from agents.evolution import EvolutionMaster, Strategy

__all__ = ["PredictionAgent", "GroupingAgent", "EvaluationAgent",
           "EvolutionMaster", "Strategy"]