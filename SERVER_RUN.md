# Запуск сборки 1v1-таблицы на арендованном сервере

Цель прогона — собрать монолитную 1v1-таблицу (Option A): значения равновесия
всех 1v1-дуэлей в belief-ключе. Результат — CSV
`cote_cfr/1v1_table.csv` (~1.45M строк, ~60МБ), который затем загружает
`cfr_bot.py` / `gen_value_data.py` вместо старой кап-6 таблицы (в ней были
ложные ничьи и провалы в material-fallback на turn>=8).

Ожидаемое время: **~30-60 минут** (фаза 4 ≈ 20-30 мин, рампа ≈ 5 мин, CSV ≈ 1 мин),
с запасом влезает в один арендованный час.

## Что именно делает прогон

1. **Фаза 4 (turn >= 7, стационарный бюджет 4-8)** — backward induction с
   `gamma=1.0` и усечённым горизонтом: итерация `T_2` (depth-2 CFR с листьями =
   текущая таблица) = значения (2k)-плёсной игры; горизонт растёт на 2 хода за
   итерацию, итерации до `max-abs-delta < tol` (по умолчанию 0.02) или
   `--max-iters`. Сетка: hits 1..16 × mover × own_bank 0..4 × own_sh 0..8 × R 0..8
   = 207 360 belief-состояний.
2. **Рампа (turn 6..1)** — по одному точному backward-induction шагу на ход
   (листья = уже готовые значения turn+2).
3. **CSV** — belief-ключ `hA,hB,to_move,own_bank,own_sh,R,turn,value`;
   turn>=7 отображается на 7 (значения фазы 4 инвариантны к ходу).

Параллелизм — **multiprocessing** (не потоки!): потоки упираются в общий
аллокатор, процессы масштабируются линейно (проверено). Rust-функция
`solve_1v1_step` решает слайс сетки за вызов.

Чекпоинты в `server/ckpt_1v1/` после каждой итерации фазы 4 и каждого хода
рампы; при обрыве тот же запуск продолжает с последнего чекпоинта.

## Пошагово (Ubuntu 22.04/24.04, всё внутри tmux)

```bash
# 1. База
sudo apt update && sudo apt install -y git curl build-essential python3 python3-pip python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
python3 -m pip install --user maturin

# 2. Код
git clone <ваш репозиторий> megaverse
cd megaverse
git checkout <ветка/коммит с этим изменением>

# 3. Сборка движка под сервер (AVX-512 / Ice Lake)
cd cote_cfr
RUSTFLAGS="-C target-cpu=native" maturin build --release
python3 -m pip install --user target/wheels/*.whl
cd ..

# 4. Запуск (в tmux, чтобы пережил обрыв SSH)
tmux new -s solver
python3 server/build_1v1_table.py --workers 56 --out cote_cfr/1v1_table.csv
```

Прогресс: каждая итерация печатает `delta`, горизонт, репрезентативные значения
(свежий (5,5), (7,7), burst-ready) и ETA. Калибровка: первые 2-3 итерации —
убедиться, что итерация ~< 2-3 мин; если нет — уменьшить `--workers`.

При обрыве: перезайти, `tmux attach`, запустить ту же команду (подхватит чекпоинты).
Форс-перезапуск: `--force`.

## Выгрузка результата

```bash
# в tmux после [done]:
# проверить
head -3 cote_cfr/1v1_table.csv
du -h cote_cfr/1v1_table.csv

# на ноуте:
scp <user>@<host>:megaverse/cote_cfr/1v1_table.csv cote_cfr/1v1_table.csv
```

Затем удалить сервер в панели управления. Таблица кладётся в
`cote_cfr/1v1_table.csv` (gitignored — это артефакт сборки) и подхватывается
ботом автоматически.

## Параметры драйвера

`--workers` (0 = все CPU), `--max-iters` (35), `--tol` (0.02),
`--solve-iters` (CFR-итераций на дерево, 150), `--hits-max` (16, для калибровки
можно меньше), `--no-ramp`, `--no-dump`, `--force`, `--ckpt-dir`, `--out`.

## Чек-лист «не навернуть»

- Сборка строго `RUSTFLAGS="-C target-cpu=native" maturin build --release` — иначе
  без AVX-512 и медленнее.
- Запуск строго в `tmux`/`screen` — иначе обрыв SSH = потерянный час.
- Первые 2 итерации смотреть на delta/ETA: если итерация дольше ~3 мин —
  остановить, снизить `--workers`, продолжить с чекпоинта (не теряется прогресс).
- На 64GB RAM 56 воркеров безопасно (каждый ~50-200МБ).
