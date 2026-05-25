# AlphaZero — Architecture & Integration Contract

This repo implements **AlphaGo Zero** (single two-headed network + MCTS + self-play RL)
in PyTorch, validated first on **tic-tac-toe**. The pipeline is **game-agnostic**: only the
`Game` rules module changes between games.

This document is the **integration contract**. Workers building different modules MUST honor
these interfaces so the pieces compose. If you need to change a contract, update this doc in the
same commit and note it in the Decision Log.

## Package layout

```
alphazero/
  __init__.py
  game.py            # abstract Game interface (bead alphago-d0g)
  games/
    __init__.py
    tictactoe.py     # tic-tac-toe rules        (bead alphago-d0g)
  network.py         # two-headed net           (bead alphago-417)
  mcts.py            # PUCT search              (bead alphago-9q8)
  selfplay.py        # self-play data gen       (bead alphago-24w)
  train.py           # training loop + loss     (bead alphago-fqw)
  arena.py           # evaluation / verify      (bead alphago-bca)
tests/
  test_*.py          # pytest, one per module
```

## Core conventions (READ FIRST)

- **Perspective / canonical form.** All states fed to the network are **canonical**: encoded
  from the perspective of the player *to move* (current player's stones on plane 0, opponent on
  plane 1). This lets one network evaluate both players.
- **Value sign.** `value ∈ [-1, 1]` is from the **current player's** perspective: `+1` = current
  player expects to win, `-1` = expects to lose. Self-play target `z` is the game outcome relative
  to the player who moved at that state.
- **Action indexing.** Moves are integers in `[0, action_size)`. For tic-tac-toe, `action = row*3 + col`.
- **Tensors.** numpy for game/state encoding; torch only inside `network.py`/`train.py`/`mcts.py`
  inference. Encoded state shape: `(planes, H, W)` float32.

## Module contracts

### `game.py` — `Game` (abstract base)
```python
class Game(ABC):
    action_size: int                 # total number of distinct actions
    board_shape: tuple[int, int]     # (H, W)
    num_planes: int                  # input planes for encode()

    def initial_state(self) -> State: ...
    def current_player(self, s: State) -> int          # +1 or -1
    def legal_moves(self, s: State) -> list[int]        # legal action indices
    def apply_move(self, s: State, a: int) -> State     # returns NEW state (immutable)
    def is_terminal(self, s: State) -> bool
    def winner(self, s: State) -> int | None            # +1 / -1 / 0 (draw) / None (not over)
    def encode(self, s: State) -> np.ndarray            # canonical (num_planes,H,W) float32
    def __str__(self, s: State) -> str                  # pretty board for debugging
```
`State` may be any hashable/serializable representation (a small numpy array or tuple).
`games/tictactoe.py` provides `TicTacToe(Game)`.

### `network.py` — `AlphaZeroNet(nn.Module)`
```python
AlphaZeroNet(num_planes: int, board_shape: tuple[int,int], action_size: int)
forward(x: Tensor[B, num_planes, H, W]) -> (policy_logits: Tensor[B, action_size],
                                            value: Tensor[B] in [-1,1])  # value via tanh
```
Provide a helper `predict(state_encoding: np.ndarray) -> (policy_probs: np.ndarray[action_size],
value: float)` that runs a single state in eval mode with softmax applied. Illegal-move masking is
done by the **caller** (MCTS), not the net.

### `mcts.py` — `MCTS`
```python
MCTS(net, game, c_puct: float = 1.5, num_simulations: int = 100,
     dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25)
run(state) -> np.ndarray   # length action_size, visit-count policy pi (illegal moves = 0)
```
Node stats per edge: `N, W, Q, P`. Selection uses **PUCT**:
`a* = argmax_a Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))`.
Expansion: one network eval per leaf, mask illegal moves, renormalize priors. Backup negates value
each ply (zero-sum). Add Dirichlet noise to root priors during self-play. Provide a `temperature`
arg on the sampling helper (τ→0 = greedy/argmax of visits).

### `selfplay.py`
```python
play_game(net, game, mcts_cfg, temperature_schedule) -> list[(encoded_state, pi, z)]
```
Records one example per move; assigns `z` after the game from each mover's perspective.

### `train.py`
```python
loss = cross_entropy(policy_logits, pi) + mse(value, z) + l2_reg
train_iteration(net, examples, ...) -> metrics
```
Combined policy+value loss. One outer iteration = self-play → train on replay buffer → checkpoint.

### `arena.py`
```python
play_match(player_a, player_b, game, n_games) -> (wins_a, draws, wins_b)
```
Verify: a trained tic-tac-toe agent **never loses** and **draws vs perfect play**.

## Decision Log
- 2026-05-25: Variant = AlphaGo Zero; first game = tic-tac-toe; game-agnostic pipeline. Built via
  delegated ntm worker swarm against the Beads DAG (`alphago-*`).
