PLAYER_PROMPT = """\
You are playing the board game Diplomacy as {power}.

=== GAME RULES ===
Diplomacy is a 7-player strategy game set in pre-WWI Europe. The powers are: \
Austria, England, France, Germany, Italy, Russia, and Turkey. Each turn you \
submit orders for your units (armies and fleets). Units can Hold, Move, \
Support, or Convoy. You win by controlling 18 of the 34 supply centers.

Key mechanics:
- Movement is simultaneous — all orders resolve at once.
- A unit needs support from other units to dislodge an enemy.
- Support can be cut by an attack on the supporting unit.
- In Fall turns, you gain/lose units based on supply center count.
- There are no random elements. Success depends entirely on negotiation and tactics.

=== CURRENT GAME STATE ===
Phase: {phase}
Your units: {units}
Your supply centers: {centers}
All supply center ownership: {all_centers}
Full board state: {board_state}

=== RECENT HISTORY ===
{history}

=== DIPLOMATIC MESSAGES THIS PHASE ===
{messages}

=== AVAILABLE ORDERS ===
{possible_orders}

Decide your orders, describe your current strategy, and compose any diplomatic \
messages. Respond in exactly this format:

ORDERS:
<one order per line, e.g. A PAR - BUR>

STRATEGY:
Paragraph not shown to any of the other players that outlines your current strategy. Must include:
PRIMARY_GOAL: <main objective this phase>
ALLIANCE_POSTURE: <current alliance / hostility posture>
MAIN_TARGET: <main power or region being pressured, or NONE>
EXPANSION_DIRECTION: <where you are trying to grow or defend>
NEXT_STEP: <the immediate next-step plan this phase>

PRIVATE MESSAGES:
<TO POWER: message, one per line, or "None">

PUBLIC MESSAGE:
<a message visible to all players, or "None">
"""

OVERSEER_PROMPT = """\
You are an expert Diplomacy analyst acting as an overseer. Your task is to \
observe the game and infer each player's hidden strategy based solely on \
their observable actions and messages.

=== GAME LOG ===
{game_log}

=== CURRENT PHASE ===
{phase}

=== CURRENT BOARD STATE ===
{board_state}

=== ALL DIPLOMATIC MESSAGES (public to you) ===
{all_messages}

=== ORDER HISTORY BY POWER ===
{order_history}

For each active power, analyze their unit movements, diplomatic messages, \
apparent alliances, and overall direction of play. Then write a free-form \
strategy description for each power.

Respond in exactly this format (one entry per power):

AUSTRIA: <your inferred strategy for Austria>
ENGLAND: <your inferred strategy for England>
FRANCE: <your inferred strategy for France>
GERMANY: <your inferred strategy for Germany>
ITALY: <your inferred strategy for Italy>
RUSSIA: <your inferred strategy for Russia>
TURKEY: <your inferred strategy for Turkey>
"""

JUDGE_PROMPT = """\
You are a pragmatic binary judge for a Diplomacy strategy-inference task.

Power:
{power}

Ground truth strategy:
{true_strategy}

Predicted strategy:
{predicted_strategy}

Decide whether the prediction captures the player's main strategic intent well enough to count as a useful strategic inference.

The ground-truth strategy and prediction may be written either as short prose or as structured fields.
If structured fields are present, compare them field-by-field for strategic meaning rather than exact wording.

Check these dimensions when present:
- primary goal or target
- alliance or hostility orientation
- betrayal or cooperation intent
- direction of expansion or defense
- immediate timing or next-step plan

Prefer to reward broad strategic alignment over exact wording or tactical completeness.
If a prediction would still help a human analyst anticipate the player's real behavior, lean toward 1.

Return 1 only if:
- the prediction gets the broad strategic direction right, and
- it correctly identifies the main theater, target, or alliance posture of the true strategy, and
- it captures the likely coalition/opponent structure well enough to be strategically useful,
- and it does not clearly contradict the true strategy.

The prediction does not need to match every tactical detail or use the same wording.
Reasonable paraphrases, slightly incomplete descriptions, and partially compressed summaries should still receive 1 if they would help an analyst understand the player's real intent.
If the prediction is broadly directionally correct but misses a secondary detail, prefer 1 rather than 0.

Return 0 if:
- the prediction is so generic that it could apply to almost any player in almost any position,
- it misses the main strategic direction entirely,
- it focuses only on a minor detail while missing the overall plan,
- or it clearly contradicts the strategy on goal, alliance, direction, or timing.

Output only: 0 or 1
"""

JUDGE_SIMILARITY_PROMPT = """\
You are grading how similar a predicted Diplomacy strategy is to the true hidden strategy.

Power:
{power}

Ground truth strategy:
{true_strategy}

Predicted strategy:
{predicted_strategy}

Score the similarity from 0 to 100.

The ground-truth strategy and prediction may be written either as short prose or as structured fields.
If structured fields are present, compare them field-by-field for strategic meaning rather than exact wording.

Use these criteria:
- strategic goal or target
- alliance posture
- betrayal or cooperation intent
- direction of expansion or defense
- timing / immediate next-step intent

Guidance:
- 90-100: essentially the same strategic intent
- 70-89: broadly correct with some missing detail
- 40-69: partially correct but misses important elements
- 10-39: weak overlap or mostly generic
- 0-9: wrong or contradictory

Output only a single integer from 0 to 100.
"""
