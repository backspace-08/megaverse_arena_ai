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

    fn units(&self, side: u8, id: u8) -> i32 {
        let team = if side == 0 { &self.team_a } else { &self.team_b };
        team[id as usize].2
    }

    fn dmg_units_between(&self, actor_side: u8, actor_id: u8, defender_side: u8, defender_id: u8) -> i32 {
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

#[pymodule]
fn _cote_cfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrunk>()?;
    Ok(())
}
