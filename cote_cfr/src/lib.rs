//! COTE Megaverse CFR engine in Rust.
//!
//! First prototype: a port of the full-game trunk rules (`src/cote_megaverse/trunk.py`)
//! with PyO3 bindings, so a depth-limited FH-CFR micro-tree search can run without
//! Python per-node overhead. State/action are flat `Vec<i32>` encodings:
//!
//! state  = [lenA, orderA..., hpA..., bankA, shA,
//!           lenB, orderB..., hpB..., bankB, shB, turn, to_move]
//! action = [a, d, b, sw]  (sw = -1 for no switch, else the target char id)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const MAX_BANK: i32 = 4;

#[inline]
fn base(t: i32) -> i32 {
    match t {
        1 => 1,
        2..=4 => 2,
        5..=6 => 3,
        _ => 4,
    }
}

#[inline]
fn multiplier(attacker: u8, defender: u8) -> f64 {
    let a = attacker as i32;
    let d = defender as i32;
    if (a + 1) % 4 == d {
        1.3
    } else if (d + 1) % 4 == a {
        0.7
    } else {
        1.0
    }
}

#[derive(Clone, Debug)]
struct State {
    order_a: Vec<u8>,
    hp_a: Vec<i32>,
    bank_a: i32,
    sh_a: i32,
    order_b: Vec<u8>,
    hp_b: Vec<i32>,
    bank_b: i32,
    sh_b: i32,
    turn: i32,
    to_move: i32,
}

#[derive(Clone, Copy, Debug)]
struct Action {
    a: i32,
    d: i32,
    b: i32,
    sw: Option<u8>,
}

fn read(p: &mut usize, raw: &[i32]) -> PyResult<i32> {
    let v = raw
        .get(*p)
        .copied()
        .ok_or_else(|| PyValueError::new_err("state vector too short"))?;
    *p += 1;
    Ok(v)
}

fn decode_state(raw: &[i32]) -> PyResult<State> {
    let mut p = 0usize;
    let len_a = read(&mut p, raw)? as usize;
    let mut order_a = Vec::with_capacity(len_a);
    let mut hp_a = Vec::with_capacity(len_a);
    for _ in 0..len_a {
        order_a.push(read(&mut p, raw)? as u8);
    }
    for _ in 0..len_a {
        hp_a.push(read(&mut p, raw)?);
    }
    let bank_a = read(&mut p, raw)?;
    let sh_a = read(&mut p, raw)?;
    let len_b = read(&mut p, raw)? as usize;
    let mut order_b = Vec::with_capacity(len_b);
    let mut hp_b = Vec::with_capacity(len_b);
    for _ in 0..len_b {
        order_b.push(read(&mut p, raw)? as u8);
    }
    for _ in 0..len_b {
        hp_b.push(read(&mut p, raw)?);
    }
    let bank_b = read(&mut p, raw)?;
    let sh_b = read(&mut p, raw)?;
    let turn = read(&mut p, raw)?;
    let to_move = read(&mut p, raw)?;
    Ok(State {
        order_a,
        hp_a,
        bank_a,
        sh_a,
        order_b,
        hp_b,
        bank_b,
        sh_b,
        turn,
        to_move,
    })
}

fn encode_state(s: &State) -> Vec<i32> {
    let mut out = Vec::with_capacity(2 + 2 * (s.order_a.len() + s.order_b.len()) + 8);
    out.push(s.order_a.len() as i32);
    out.extend(s.order_a.iter().map(|&x| x as i32));
    out.extend(s.hp_a.iter().copied());
    out.push(s.bank_a);
    out.push(s.sh_a);
    out.push(s.order_b.len() as i32);
    out.extend(s.order_b.iter().map(|&x| x as i32));
    out.extend(s.hp_b.iter().copied());
    out.push(s.bank_b);
    out.push(s.sh_b);
    out.push(s.turn);
    out.push(s.to_move);
    out
}

struct Trunk {
    team_a: Vec<(u8, i32, i32)>, // (type, atk, max_hp_units)
    team_b: Vec<(u8, i32, i32)>,
    turn_cap: i32,
}

impl Trunk {
    fn new(team_a: Vec<(u8, i32, i32)>, team_b: Vec<(u8, i32, i32)>, cap: i32, start_turn: i32) -> Self {
        Trunk {
            team_a,
            team_b,
            turn_cap: start_turn + cap,
        }
    }

    fn dmg_units_between(&self, actor_side: u8, actor_id: u8, _defender_side: u8, defender_id: u8) -> i32 {
        let a = if actor_side == 0 { &self.team_a } else { &self.team_b };
        let d = if actor_side == 0 { &self.team_b } else { &self.team_a };
        let atk = a[actor_id as usize].1;
        let m = multiplier(a[actor_id as usize].0, d[defender_id as usize].0);
        (atk as f64 * m) as i32 / 10
    }

    fn budget(&self, s: &State) -> i32 {
        let own = if s.to_move == 0 { s.bank_a } else { s.bank_b };
        base(s.turn) + own
    }

    fn actions(&self, s: &State) -> Vec<Action> {
        let order = if s.to_move == 0 { &s.order_a } else { &s.order_b };
        let bgt = self.budget(s);
        let mut out = Vec::new();
        let switches: Vec<Option<u8>> = std::iter::once(None).chain(order.iter().skip(1).map(|&c| Some(c))).collect();
        for sw in switches {
            let rem = bgt - if sw.is_some() { 1 } else { 0 };
            for a in 0..=rem {
                for d in 0..=(rem - a) {
                    let b = rem - a - d;
                    if b <= MAX_BANK {
                        out.push(Action { a, d, b, sw });
                    }
                }
            }
        }
        out
    }

    fn promote(order: &[u8], hp: &[i32], dead: u8) -> (Vec<u8>, Vec<i32>) {
        let mut no = Vec::new();
        let mut nh = Vec::new();
        for (i, &c) in order.iter().enumerate() {
            if c != dead {
                no.push(c);
                nh.push(hp[i]);
            }
        }
        (no, nh)
    }

    fn transition(&self, s: &State, act: &Action) -> Option<State> {
        let a = act.a;
        let d = act.d;
        let b = act.b;
        if s.to_move == 0 {
            let mut order_a = s.order_a.clone();
            let mut hp_a = s.hp_a.clone();
            if let Some(sw) = act.sw {
                let i = order_a.iter().position(|&c| c == sw)?;
                let o = order_a.remove(i);
                order_a.insert(0, o);
                let h = hp_a.remove(i);
                hp_a.insert(0, h);
            }
            let dmg = self.dmg_units_between(0, order_a[0], 1, s.order_b[0]);
            let landed = (a - s.sh_b).max(0);
            let mut order_b = s.order_b.clone();
            let mut hp_b = s.hp_b.clone();
            if landed > 0 {
                hp_b[0] = (hp_b[0] - landed * dmg).max(0);
                if hp_b[0] <= 0 {
                    let dead = order_b[0];
                    let (no, nh) = Self::promote(&order_b, &hp_b, dead);
                    order_b = no;
                    hp_b = nh;
                }
            }
            Some(State {
                order_a,
                hp_a,
                bank_a: b.min(MAX_BANK),
                sh_a: d,
                order_b,
                hp_b,
                bank_b: s.bank_b,
                sh_b: 0,
                turn: s.turn + 1,
                to_move: 1,
            })
        } else {
            let mut order_b = s.order_b.clone();
            let mut hp_b = s.hp_b.clone();
            if let Some(sw) = act.sw {
                let i = order_b.iter().position(|&c| c == sw)?;
                let o = order_b.remove(i);
                order_b.insert(0, o);
                let h = hp_b.remove(i);
                hp_b.insert(0, h);
            }
            let dmg = self.dmg_units_between(1, order_b[0], 0, s.order_a[0]);
            let landed = (a - s.sh_a).max(0);
            let mut order_a = s.order_a.clone();
            let mut hp_a = s.hp_a.clone();
            if landed > 0 {
                hp_a[0] = (hp_a[0] - landed * dmg).max(0);
                if hp_a[0] <= 0 {
                    let dead = order_a[0];
                    let (no, nh) = Self::promote(&order_a, &hp_a, dead);
                    order_a = no;
                    hp_a = nh;
                }
            }
            Some(State {
                order_a,
                hp_a,
                bank_a: s.bank_a,
                sh_a: 0,
                order_b,
                hp_b,
                bank_b: b.min(MAX_BANK),
                sh_b: d,
                turn: s.turn + 1,
                to_move: 0,
            })
        }
    }

    fn terminal(&self, s: &State) -> Option<f64> {
        if s.order_a.is_empty() {
            return Some(-1.0);
        }
        if s.order_b.is_empty() {
            return Some(1.0);
        }
        if s.turn > self.turn_cap {
            return Some(0.0);
        }
        None
    }
}

// ------------------------------------------------------------------ micro-tree
// Depth-limited FH-CFR (perfect-recall info sets) over a small tree from the
// current position. The whole loop runs in Rust; only the final root strategy
// crosses the PyO3 boundary.

use std::collections::HashMap;

fn info_key_flat(s: &State) -> Vec<i32> {
    // acting player's public observation: (turn, orderA, hpA, orderB, hpB, own_bank, R)
    let mut out = Vec::with_capacity(16);
    out.push(s.turn);
    out.push(s.order_a.len() as i32);
    out.extend(s.order_a.iter().map(|&x| x as i32));
    out.extend(s.hp_a.iter().copied());
    out.push(s.order_b.len() as i32);
    out.extend(s.order_b.iter().map(|&x| x as i32));
    out.extend(s.hp_b.iter().copied());
    if s.to_move == 0 {
        out.push(s.bank_a);
        out.push(s.sh_b + s.bank_b); // R
    } else {
        out.push(s.bank_b);
        out.push(s.sh_a + s.bank_a); // R
    }
    out
}

struct InfoSet {
    n_acts: usize,
    regret: Vec<f64>,
    avg: Vec<f64>,
}

struct MicroSolver {
    trunk: Trunk,
    depth: u8,
    gamma: f64,
    root: usize,
    states: Vec<State>,
    terminals: Vec<Option<f64>>,
    depths: Vec<u8>,
    actions_of: Vec<Vec<Action>>,
    children: Vec<Vec<usize>>,
    info_id: Vec<usize>,
    hist_a: Vec<Vec<Vec<i32>>>,
    hist_b: Vec<Vec<Vec<i32>>>,
    infos: Vec<InfoSet>,
    r_a: Vec<f64>,
    r_b: Vec<f64>,
    v: Vec<f64>,
    reach_opp_sum: Vec<f64>,
    reach_self_sum: Vec<f64>,
    cfv_contrib: Vec<Vec<f64>>,
}

impl MicroSolver {
    fn new(trunk: Trunk, root_state: State, depth: u8, gamma: f64) -> Self {
        let mut s = MicroSolver {
            trunk,
            depth,
            gamma,
            root: 0,
            states: Vec::new(),
            terminals: Vec::new(),
            depths: Vec::new(),
            actions_of: Vec::new(),
            children: Vec::new(),
            info_id: Vec::new(),
            hist_a: Vec::new(),
            hist_b: Vec::new(),
            infos: Vec::new(),
            r_a: Vec::new(),
            r_b: Vec::new(),
            v: Vec::new(),
            reach_opp_sum: Vec::new(),
            reach_self_sum: Vec::new(),
            cfv_contrib: Vec::new(),
        };
        s.build(root_state);
        s
    }

    fn get_or_add_info(&mut self, map: &mut HashMap<(u8, Vec<Vec<i32>>), usize>, tm: u8, hist: &[Vec<i32>]) -> usize {
        let key = (tm, hist.to_vec());
        if let Some(&id) = map.get(&key) {
            return id;
        }
        // actions count = the number of actions at the (first) node with this info.
        // We fill it lazily after the build by sampling a node. For the map we only
        // need the id; n_acts is patched in finalize().
        let id = self.infos.len();
        self.infos.push(InfoSet { n_acts: 0, regret: Vec::new(), avg: Vec::new() });
        map.insert(key, id);
        id
    }

    fn build(&mut self, root: State) {
        let mut info_map: HashMap<(u8, Vec<Vec<i32>>), usize> = HashMap::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        let ik = info_key_flat(&root);
        let mut ha: Vec<Vec<i32>> = Vec::new();
        let mut hb: Vec<Vec<i32>> = Vec::new();
        if root.to_move == 0 {
            ha.push(ik);
        } else {
            hb.push(ik);
        }
        self.root = self.states.len();
        self.states.push(root.clone());
        self.terminals.push(self.trunk.terminal(&root));
        self.depths.push(0);
        self.hist_a.push(ha.clone());
        self.hist_b.push(hb.clone());
        let iid = self.get_or_add_info(&mut info_map, root.to_move as u8, if root.to_move == 0 { &ha } else { &hb });
        self.info_id.push(iid);
        self.actions_of.push(Vec::new());
        self.children.push(Vec::new());
        queue.push_back(self.root);

        while let Some(i) = queue.pop_front() {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let acts = self.trunk.actions(&self.states[i]);
            let n_acts = acts.len();
            self.actions_of[i] = acts.clone();
            for act in acts {
                if let Some(child) = self.trunk.transition(&self.states[i], &act) {
                    let mut cha = self.hist_a[i].clone();
                    let mut chb = self.hist_b[i].clone();
                    let cik = info_key_flat(&child);
                    if child.to_move == 0 {
                        cha.push(cik);
                    } else {
                        chb.push(cik);
                    }
                    let cid = self.states.len();
                    self.states.push(child.clone());
                    self.terminals.push(self.trunk.terminal(&child));
                    self.depths.push(self.depths[i] + 1);
                    self.hist_a.push(cha);
                    self.hist_b.push(chb);
                    let chist = if child.to_move == 0 {
                        self.hist_a[cid].clone()
                    } else {
                        self.hist_b[cid].clone()
                    };
                    let ciid = self.get_or_add_info(&mut info_map, child.to_move as u8, &chist);
                    self.info_id.push(ciid);
                    self.children[i].push(cid);
                    self.actions_of.push(Vec::new());
                    self.children.push(Vec::new());
                    queue.push_back(cid);
                }
            }
            // actions_of[i] is set; children[i] parallels the actions (only valid transitions).
            if n_acts != self.children[i].len() {
                // some transitions returned None (shouldn't happen: wipe returns empty order).
                // keep children parallel to actions by padding with a self-loop? We disallow None.
                // (trunk.transition returns Some always; wipe -> empty order, terminal).
            }
        }

        // finalize info sets: n_acts from the first node, allocate arrays
        let n = self.states.len();
        for iid in 0..self.infos.len() {
            let mut na = 0;
            for i in 0..n {
                if self.info_id[i] == iid {
                    na = self.actions_of[i].len();
                    break;
                }
            }
            self.infos[iid].n_acts = na;
            self.infos[iid].regret = vec![0.0; na];
            self.infos[iid].avg = vec![0.0; na];
        }
        self.r_a = vec![0.0; n];
        self.r_b = vec![0.0; n];
        self.v = vec![0.0; n];
        let n_infos = self.infos.len();
        let max_acts = self.infos.iter().map(|x| x.n_acts).max().unwrap_or(1);
        self.reach_opp_sum = vec![0.0; n_infos];
        self.reach_self_sum = vec![0.0; n_infos];
        self.cfv_contrib = vec![vec![0.0; max_acts]; n_infos];
    }

    fn leaf_value(&self, i: usize) -> f64 {
        let s = &self.states[i];
        if s.order_a.is_empty() {
            return -1.0;
        }
        if s.order_b.is_empty() {
            return 1.0;
        }
        let ha: i32 = s.hp_a.iter().sum();
        let hb: i32 = s.hp_b.iter().sum();
        let material = (ha - hb) as f64 / 6000.0;
        let bodies = (s.order_a.len() as f64 - s.order_b.len() as f64) * 0.15;
        (material + bodies).clamp(-1.0, 1.0)
    }

    fn sigs(&self) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(self.infos.len());
        for info in &self.infos {
            let na = info.n_acts;
            let mut sig = vec![0.0; na];
            let pos: Vec<f64> = info.regret.iter().map(|r| r.max(0.0)).collect();
            let total: f64 = pos.iter().sum();
            if total > 0.0 {
                for (k, v) in sig.iter_mut().enumerate() {
                    *v = pos[k] / total;
                }
            } else {
                for v in sig.iter_mut() {
                    *v = 1.0 / na as f64;
                }
            }
            out.push(sig);
        }
        out
    }

    fn iterate(&mut self) {
        let n = self.states.len();
        let sigs = self.sigs();
        // reach
        self.r_a.fill(0.0);
        self.r_b.fill(0.0);
        self.r_a[self.root] = 1.0;
        self.r_b[self.root] = 1.0;
        for i in 0..n {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let sig = &sigs[self.info_id[i]];
            for (a, &ci) in self.children[i].iter().enumerate() {
                let p = sig[a];
                if self.states[i].to_move == 0 {
                    self.r_a[ci] += self.r_a[i] * p;
                    self.r_b[ci] += self.r_b[i];
                } else {
                    self.r_b[ci] += self.r_b[i] * p;
                    self.r_a[ci] += self.r_a[i];
                }
            }
        }
        // value (children have higher ids)
        for i in (0..n).rev() {
            if let Some(t) = self.terminals[i] {
                self.v[i] = t;
            } else if self.depths[i] >= self.depth {
                self.v[i] = self.leaf_value(i);
            } else {
                let sig = &sigs[self.info_id[i]];
                let mut acc = 0.0;
                for (a, &ci) in self.children[i].iter().enumerate() {
                    acc += sig[a] * self.v[ci];
                }
                self.v[i] = self.gamma * acc;
            }
        }
        // cfv accumulation
        self.reach_opp_sum.fill(0.0);
        self.reach_self_sum.fill(0.0);
        for row in self.cfv_contrib.iter_mut() {
            row.fill(0.0);
        }
        for i in 0..n {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let iid = self.info_id[i];
            let tm = self.states[i].to_move;
            let opp = if tm == 0 { self.r_b[i] } else { self.r_a[i] };
            let slf = if tm == 0 { self.r_a[i] } else { self.r_b[i] };
            self.reach_opp_sum[iid] += opp;
            self.reach_self_sum[iid] += slf;
            for (a, &ci) in self.children[i].iter().enumerate() {
                self.cfv_contrib[iid][a] += opp * self.v[ci];
            }
        }
        // regret / avg update (CFR+)
        for iid in 0..self.infos.len() {
            let na = self.infos[iid].n_acts;
            let rsum = self.reach_opp_sum[iid];
            let mut v_info = 0.0;
            for a in 0..na {
                let cfv = if rsum > 0.0 { self.cfv_contrib[iid][a] / rsum } else { 0.0 };
                v_info += sigs[iid][a] * cfv;
            }
            // sign: A maximizes (+), B minimizes (-)
            let a_move = self.a_move_of_info(iid);
            let sign = if a_move { 1.0 } else { -1.0 };
            for a in 0..na {
                let cfv = if rsum > 0.0 { self.cfv_contrib[iid][a] / rsum } else { 0.0 };
                let delta = rsum * sign * (cfv - v_info);
                let r = self.infos[iid].regret[a] + delta;
                self.infos[iid].regret[a] = r.max(0.0);
                self.infos[iid].avg[a] += self.reach_self_sum[iid] * sigs[iid][a];
            }
        }
    }

    fn a_move_of_info(&self, iid: usize) -> bool {
        // A's info sets have to_move == 0 at their nodes
        for i in 0..self.states.len() {
            if self.info_id[i] == iid {
                return self.states[i].to_move == 0;
            }
        }
        true
    }

    fn root_strategy(&self) -> (Vec<f64>, f64) {
        let iid = self.info_id[self.root];
        let na = self.infos[iid].n_acts;
        let avg = &self.infos[iid].avg;
        let total: f64 = avg.iter().sum();
        let probs: Vec<f64> = if total > 0.0 {
            avg.iter().map(|x| x / total).collect()
        } else {
            vec![1.0 / na as f64; na]
        };
        (probs, self.v[self.root])
    }
}

// ------------------------------------------------------------------ PyO3

#[pyclass]
struct PyTrunk {
    inner: Trunk,
}

#[pymethods]
impl PyTrunk {
    #[new]
    #[pyo3(signature = (team_a, team_b, cap=20, start_turn=1))]
    fn new(team_a: Vec<(u8, i32, i32)>, team_b: Vec<(u8, i32, i32)>, cap: i32, start_turn: i32) -> Self {
        PyTrunk {
            inner: Trunk::new(team_a, team_b, cap, start_turn),
        }
    }

    fn transition(&self, state: Vec<i32>, action: Vec<i32>) -> PyResult<Option<Vec<i32>>> {
        if action.len() < 4 {
            return Err(PyValueError::new_err("action must be [a, d, b, sw]"));
        }
        let st = decode_state(&state)?;
        let act = Action {
            a: action[0],
            d: action[1],
            b: action[2],
            sw: if action[3] < 0 { None } else { Some(action[3] as u8) },
        };
        Ok(self.inner.transition(&st, &act).map(|s| encode_state(&s)))
    }

    fn actions(&self, state: Vec<i32>) -> PyResult<Vec<Vec<i32>>> {
        let st = decode_state(&state)?;
        Ok(self
            .inner
            .actions(&st)
            .iter()
            .map(|ac| vec![ac.a, ac.d, ac.b, ac.sw.map(|x| x as i32).unwrap_or(-1)])
            .collect())
    }

    fn terminal(&self, state: Vec<i32>) -> PyResult<Option<f64>> {
        let st = decode_state(&state)?;
        Ok(self.inner.terminal(&st))
    }

    /// Active matchup hits-to-kill: (hits A needs to kill B's active, hits B needs to kill A's active).
    fn hits(&self, state: Vec<i32>) -> PyResult<(i32, i32)> {
        let st = decode_state(&state)?;
        let db = self.inner.dmg_units_between(0, st.order_a[0], 1, st.order_b[0]);
        let da = self.inner.dmg_units_between(1, st.order_b[0], 0, st.order_a[0]);
        let hb = if db > 0 { (st.hp_b[0] + db - 1) / db } else { 1_000_000 };
        let ha = if da > 0 { (st.hp_a[0] + da - 1) / da } else { 1_000_000 };
        Ok((ha.max(1), hb.max(1)))
    }
}

#[pyfunction]
#[pyo3(signature = (team_a, team_b, state, depth=4, iters=500, gamma=0.995, cap=20, start_turn=1))]
#[allow(clippy::too_many_arguments)]
fn solve_micro(
    team_a: Vec<(u8, i32, i32)>,
    team_b: Vec<(u8, i32, i32)>,
    state: Vec<i32>,
    depth: u8,
    iters: usize,
    gamma: f64,
    cap: i32,
    start_turn: i32,
) -> PyResult<(Vec<Vec<i32>>, Vec<f64>, f64)> {
    let root = decode_state(&state)?;
    let trunk = Trunk::new(team_a, team_b, cap, start_turn);
    let mut solver = MicroSolver::new(trunk, root, depth, gamma);
    for _ in 0..iters {
        solver.iterate();
    }
    let acts = &solver.actions_of[solver.root];
    let (probs, value) = solver.root_strategy();
    let actions: Vec<Vec<i32>> = acts
        .iter()
        .map(|ac| vec![ac.a, ac.d, ac.b, ac.sw.map(|x| x as i32).unwrap_or(-1)])
        .collect();
    Ok((actions, probs, value))
}

#[pymodule]
fn _cote_cfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrunk>()?;
    m.add_function(wrap_pyfunction!(solve_micro, m)?)?;
    Ok(())
}
