#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocolo HFT WIN — versão 2 (revisada)
=======================================

Reescrita do protocolo de fluxos informacionais no WIN (B3) incorporando as
correções da revisão crítica:

  1. Grade temporal COMPLETA de dt_ms por SEGMENTO de pregão contínuo (sem
     cruzar dias nem leilões). Mid preenchido para frente, fluxos zerados em
     janelas vazias. Horizontes passam a ser de relógio de verdade.
  2. Margens de exclusão no início e no fim de cada segmento.
  3. OFI separado em componente de cotações (adições - cancelamentos) e
     componente de execução (≈ TI), eliminando a colinearidade mecânica.
  4. Atribuição Lit / RL / L RL exata quando a linha tem uma só condição e
     proporcional apenas nas linhas mistas (fração reportada).
  5. H1 na forma contemporânea de Cont, Kukanov & Stoikov (por blocos).
  6. Testes preditivos e de impacto de preço por tipo de negócio com
     inferência Fama-MacBeth entre dias.
  7. VAR de Hasbrouck com fluxos antes do retorno, impacto acumulado
     (componente de passeio aleatório) e limites sobre todas as ordenações.
  8. QID em frequência mais baixa, com mínimo de eventos; H3 contra
     baseline HAR, com avaliação fora da amostra.
  9. QIDres e teste de reversão (Eq. 7) — CONFERIR a especificação com o
     artigo de Barardehi, Dixon & Liu antes de interpretar.

Uso:
  python protocolo_hft_win_v2.py --parquet melhores_ofertas_win.parquet
  python protocolo_hft_win_v2.py --usar-cache --etapas var,qid
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR
import warnings

# avisos conhecidos e inofensivos: os dados já estão ordenados por timestamp antes de cada join_asof
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")


# =============================================================================
# Configuração
# =============================================================================
@dataclass
class Config:
    parquet_path: str = "melhores_ofertas_win.parquet"
    saida_dir: str = "resultados/protocolo_hft"
    dt_ms: int = 100                 # janela base da grade
    tick: float = 5.0                # tick do WIN em pontos
    margem_inicio_s: int = 300       # exclui início de cada segmento contínuo
    margem_fim_s: int = 300          # exclui fim de cada segmento contínuo

    # Mapeamento das condições de negócio (CONFIRMAR semântica no UMDF da B3)
    cols_lit: tuple = ("negocio_qtd_l", "negocio_qtd_sem_condicao")
    cols_rl: tuple = ("negocio_qtd_rl",)
    cols_lrl: tuple = ("negocio_qtd_l_rl",)

    # H1 contemporâneo (CKS)
    h1_intervalo_s: int = 10
    h1_bloco_min: int = 30
    h1_min_obs: int = 30

    # Horizontes (segundos de relógio) para testes preditivos e de impacto
    horizontes_s: tuple = (1, 5, 30, 60, 300)

    # VAR de Hasbrouck
    var_intervalo_ms: int = 1000
    var_lags: int = 10
    var_horizonte_ir: int = 120

    # QID / H3 / Eq. 7
    qid_intervalo: str = "1m"
    qid_min_eventos: int = 5
    qid_definicao: str = "diferenca"   # "diferenca" = (I-D)/(I+D) ; "razao" = I/D
    reversao_horizonte: int = 5        # em intervalos de qid_intervalo
    dias_treino: int = 8               # dias para estimação na avaliação fora da amostra
    hac_lags_qid: int = 5

    # Etapa LINHA
    atrasos_s: tuple = (0.01, 0.1)     # linhas de base atrasadas e perfil fino do impacto
    esgotamento_s: float = 0.1         # janela para decidir se o negócio esgotou a fila do topo

    # Etapa FILAS
    filas_lim_prox_s: int = 60         # só usa janelas em que o preço muda de novo em até N segundos

    # Etapa RLP_FILAS
    rlp_intervalo: str = "5m"          # intervalo para medir a participação do RLP

    # Etapa ROMPIMENTO
    rompimento_janela_s: float = 1.0   # olhar no máximo N s antes do rompimento para achar a causa
    rompimento_limiar: float = 0.8     # fração mínima para classificar como agressão ou cancelamento

    # Etapa QID_DR
    dr_min_saltos: int = 20            # mínimo de pares de saltos de 1 tick para estimar eta no intervalo

    # Etapa QID_CURTO
    curto_janela_saltos: int = 500     # janela móvel (pares de saltos) do spread implícito, como no Gemini


PROTOCOLO_VERSAO = "1.0"


def _n_treino(cfg, n: int) -> int:
    """Dias de treino: valor fixo (exploratório) ou 2/3 dos dias, arredondado para cima (confirmatório, dias_treino <= 0)."""
    if cfg.dias_treino <= 0:
        return min(max(1, math.ceil(2 * n / 3)), max(1, n - 1))
    return cfg.dias_treino if n > cfg.dias_treino + 1 else max(1, int(round(n * 0.67)))


SOMAS = [
    "ofi_cot", "exec_liq", "ofi_total", "ti", "ti_lit", "ti_rl", "ti_lrl",
    "v_lit", "v_rl", "v_lrl", "v_cls", "v_total", "v_misto", "impr", "deter",
    "adic", "melh", "canc_tot", "canc_parc",
]
FLUXOS_VAR = ["ofi_cot", "ti_lit", "ti_rl", "ti_lrl"]


# =============================================================================
# Utilitários
# =============================================================================
class Relatorio:
    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.linhas: list[str] = []
        self.tabelas: dict[str, pd.DataFrame] = {}

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.linhas.append(msg)

    def tabela(self, nome: str, df: pd.DataFrame, mostrar: bool = True, fmt: str = "{:.5g}") -> None:
        self.tabelas[nome] = df
        if mostrar:
            with pd.option_context("display.max_columns", 50, "display.width", 200):
                self.log(df.to_string(float_format=lambda x: fmt.format(x)))

    def salvar(self, cfg: Config, nome_relatorio: str = "relatorio.txt") -> None:
        self.pasta.mkdir(parents=True, exist_ok=True)
        for nome, df in self.tabelas.items():
            df.to_csv(self.pasta / f"{nome}.csv")
        (self.pasta / nome_relatorio).write_text("\n".join(self.linhas), encoding="utf-8")
        (self.pasta / nome_relatorio.replace("relatorio", "config").replace(".txt", ".json")).write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")


def ols_np(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS com constante via lstsq. Retorna (beta com constante na posição 0, R²)."""
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    vy = y.var()
    r2 = 1.0 - resid.var() / vy if vy > 0 else np.nan
    return beta, r2


def resumo_fm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Inferência Fama-MacBeth: média das estimativas por unidade (dia) e t com n-1 g.l."""
    linhas = []
    for c in cols:
        x = df[c].dropna().to_numpy(dtype=float)
        n = len(x)
        if n < 2:
            linhas.append(dict(variavel=c, media=np.nan, ep=np.nan, t=np.nan, p=np.nan,
                               ic95_inf=np.nan, ic95_sup=np.nan, n=n, frac_pos=np.nan))
            continue
        m = x.mean()
        ep = x.std(ddof=1) / math.sqrt(n)
        t = m / ep if ep > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
        tc = stats.t.ppf(0.975, n - 1)
        linhas.append(dict(variavel=c, media=m, ep=ep, t=t, p=p, ic95_inf=m - tc * ep, ic95_sup=m + tc * ep,
                           n=n, frac_pos=(x > 0).mean()))
    return pd.DataFrame(linhas).set_index("variavel")


def teste_permutacao(valores: np.ndarray, grupo: np.ndarray, n_max: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Diferença de médias (grupo True - False) com p-valor por permutação exata (ou Monte Carlo)."""
    ok = np.isfinite(valores)
    v, g = valores[ok], grupo[ok].astype(bool)
    n, n1 = len(v), int(g.sum())
    if n1 == 0 or n1 == n:
        return np.nan, np.nan
    obs = v[g].mean() - v[~g].mean()
    total = v.sum()

    def dif(idx):
        s1 = v[list(idx)].sum()
        return s1 / n1 - (total - s1) / (n - n1)

    if math.comb(n, n1) <= n_max:
        difs = np.array([dif(c) for c in itertools.combinations(range(n), n1)])
    else:
        rng = np.random.default_rng(seed)
        difs = np.array([dif(rng.choice(n, n1, replace=False)) for _ in range(n_max)])
    p = float(np.mean(np.abs(difs) >= abs(obs) - 1e-12))
    return obs, p


def diebold_mariano(e_base: np.ndarray, e_novo: np.ndarray, lags: int) -> tuple[float, float]:
    """H0: perdas quadráticas iguais. H1: modelo novo melhor. Retorna (t, p unilateral)."""
    d = e_base ** 2 - e_novo ** 2
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(m.tvalues[0])
    return t, float(stats.norm.sf(t))


def _f0(c: str) -> pl.Expr:
    return pl.col(c).fill_null(0).cast(pl.Float64)


def _soma(cols) -> pl.Expr:
    return pl.sum_horizontal([_f0(c) for c in cols])


def _expr_timestamp(schema) -> pl.Expr:
    td, th = schema["data"], schema["hora"]
    if th == pl.Datetime:
        return pl.col("hora")
    if td == pl.Date and th == pl.Time:
        return pl.col("data").dt.combine(pl.col("hora"))
    return (pl.col("data").cast(pl.Utf8) + " " + pl.col("hora").cast(pl.Utf8)).str.to_datetime("%Y-%m-%d %H:%M:%S%.f")


# =============================================================================
# Etapa 1: grade temporal completa
# =============================================================================
def _lazy_continuo(cfg: Config, colunas: list[str]) -> pl.LazyFrame:
    """Leitura comum: timestamp, ordenação, segmentos contínuos (dia + seg_n) e filtro de lado vazio."""
    lf = pl.scan_parquet(cfg.parquet_path)
    try:
        schema = lf.collect_schema()
    except AttributeError:
        schema = lf.schema
    necessarias = list(dict.fromkeys(["data", "hora", "fase", "bid_preco", "ask_preco"] + colunas))
    faltando = [c for c in necessarias if c not in schema]
    if faltando:
        raise SystemExit(f"Colunas ausentes na base: {faltando}")
    return (
        lf.select(necessarias).with_row_index("ordem")
        .with_columns(_expr_timestamp(schema).alias("timestamp"))
        .with_columns(pl.col("timestamp").dt.date().alias("dia"))
        .sort(["timestamp", "ordem"])
        # identifica segmentos contíguos de mesma fase dentro do dia
        .with_columns(
            (pl.col("fase") != pl.col("fase").shift(1)).fill_null(True).cast(pl.UInt32).cum_sum().over("dia").alias("seg_n")
        )
        .filter(pl.col("fase") == "pregao_continuo")
        .filter(pl.col("bid_preco").is_not_null() & pl.col("ask_preco").is_not_null())
        .with_columns((pl.col("dia").cast(pl.Utf8) + "_" + pl.col("seg_n").cast(pl.Utf8)).alias("segmento"))
    )


def construir_grade(cfg: Config, rel: Relatorio) -> pl.DataFrame:
    rel.log(f"[grade] Lendo {cfg.parquet_path} ...")
    t0 = time.time()

    eventos = [f"{lado}_{c}" for lado in ("bid", "ask") for c in (
        "nova_melhora_qtd", "nova_topo_qtd", "cancel_total_qtd", "cancel_parcial_qtd", "execucao_qtd",
        "nova_melhora_n", "deter_trade_n")]
    necessarias = (["data", "hora", "fase", "bid_preco", "ask_preco", "bid_qtd", "ask_qtd", "mid_simples",
                    "negocio_qtd", "negocio_qtd_agr_compra", "negocio_qtd_agr_venda"]
                   + eventos + list(cfg.cols_lit) + list(cfg.cols_rl) + list(cfg.cols_lrl))
    lf = _lazy_continuo(cfg, necessarias)

    # ----- variáveis por linha -----
    add_b = _f0("bid_nova_melhora_qtd") + _f0("bid_nova_topo_qtd")
    can_b = _f0("bid_cancel_total_qtd") + _f0("bid_cancel_parcial_qtd")
    exe_b = _f0("bid_execucao_qtd")
    add_a = _f0("ask_nova_melhora_qtd") + _f0("ask_nova_topo_qtd")
    can_a = _f0("ask_cancel_total_qtd") + _f0("ask_cancel_parcial_qtd")
    exe_a = _f0("ask_execucao_qtd")

    ti = _f0("negocio_qtd_agr_compra") - _f0("negocio_qtd_agr_venda")
    v_lit, v_rl, v_lrl = _soma(cfg.cols_lit), _soma(cfg.cols_rl), _soma(cfg.cols_lrl)
    v_cls = v_lit + v_rl + v_lrl
    n_cls = (v_lit > 0).cast(pl.Int8) + (v_rl > 0).cast(pl.Int8) + (v_lrl > 0).cast(pl.Int8)

    def parte(v: pl.Expr) -> pl.Expr:
        # exata se só há uma condição na linha; proporcional nas linhas mistas
        return pl.when(v_cls > 0).then(ti * v / v_cls).otherwise(0.0)

    lf = lf.with_columns([
        ((add_b - can_b) - (add_a - can_a)).alias("ofi_cot"),
        (exe_a - exe_b).alias("exec_liq"),
        (_f0("bid_nova_topo_qtd") - _f0("ask_nova_topo_qtd")).alias("adic"),
        (_f0("bid_nova_melhora_qtd") - _f0("ask_nova_melhora_qtd")).alias("melh"),
        (_f0("ask_cancel_total_qtd") - _f0("bid_cancel_total_qtd")).alias("canc_tot"),
        (_f0("ask_cancel_parcial_qtd") - _f0("bid_cancel_parcial_qtd")).alias("canc_parc"),
        ((add_b - can_b - exe_b) - (add_a - can_a - exe_a)).alias("ofi_total"),
        ti.alias("ti"),
        parte(v_lit).alias("ti_lit"),
        parte(v_rl).alias("ti_rl"),
        parte(v_lrl).alias("ti_lrl"),
        v_lit.alias("v_lit"), v_rl.alias("v_rl"), v_lrl.alias("v_lrl"), v_cls.alias("v_cls"),
        _f0("negocio_qtd").alias("v_total"),
        pl.when(n_cls > 1).then(v_cls).otherwise(0.0).alias("v_misto"),
        (_f0("bid_nova_melhora_n") + _f0("ask_nova_melhora_n")).alias("impr"),
        (_f0("bid_deter_trade_n") + _f0("ask_deter_trade_n")).alias("deter"),
        ((_f0("bid_qtd") + _f0("ask_qtd")) / 2.0).alias("depth"),
        ((pl.col("ask_preco") - pl.col("bid_preco")) / cfg.tick).alias("spread_ticks_l"),
        pl.when((_f0("bid_qtd") + _f0("ask_qtd")) > 0)
          .then((_f0("bid_qtd") - _f0("ask_qtd")) / (_f0("bid_qtd") + _f0("ask_qtd"))).alias("qi_l"),
        pl.col("timestamp").dt.truncate(f"{cfg.dt_ms}ms").alias("janela"),
    ])

    agg = (
        lf.group_by(["segmento", "dia", "janela"])
        .agg([
            pl.col("mid_simples").last().alias("mid"),
            pl.col("spread_ticks_l").last().alias("spread_ticks"),
            pl.col("qi_l").last().alias("qi"),
            pl.col("depth").mean().alias("depth"),
            pl.len().alias("n_linhas"),
        ] + [pl.col(c).sum() for c in SOMAS])
        .collect()
    )
    rel.log(f"[grade] {agg.height:,} janelas não vazias agregadas em {time.time() - t0:.1f}s")

    # ----- grade completa por segmento -----
    dtype = agg.schema["janela"]
    tu, tz = getattr(dtype, "time_unit", "us"), getattr(dtype, "time_zone", None)
    lim = agg.group_by(["segmento", "dia"]).agg(pl.col("janela").min().alias("ini"), pl.col("janela").max().alias("fim"))
    partes = []
    for r in lim.sort("segmento").iter_rows(named=True):
        rng = pl.datetime_range(r["ini"], r["fim"], interval=f"{cfg.dt_ms}ms", eager=True, time_unit=tu, time_zone=tz)
        partes.append(pl.DataFrame({"janela": rng}).with_columns(
            pl.lit(r["segmento"]).alias("segmento"), pl.lit(r["dia"]).alias("dia")))
    grade = pl.concat(partes).join(agg.drop("dia"), on=["segmento", "janela"], how="left")

    grade = (
        grade.sort(["segmento", "janela"])
        .with_columns(
            [pl.col(c).forward_fill().over("segmento") for c in ("mid", "spread_ticks", "depth", "qi")]
            + [pl.col(c).fill_null(0) for c in SOMAS + ["n_linhas"]]
        )
    )

    def k(h: float) -> int:
        return int(round(h * 1000 / cfg.dt_ms))

    leads = []
    for h in cfg.horizontes_s:
        leads.append(((pl.col("mid").shift(-k(h)) - pl.col("mid")).over("segmento") / cfg.tick).alias(f"fwd_{h}s"))
        leads.append(((pl.col("mid").shift(-k(h)) - pl.col("mid").shift(1)).over("segmento") / cfg.tick).alias(f"pi_{h}s"))

    grade = grade.with_columns([
        ((pl.col("mid") - pl.col("mid").shift(1)).over("segmento") / cfg.tick).alias("ret_ticks"),
        ((pl.col("janela") - pl.col("janela").min().over("segmento")).dt.total_milliseconds() / 1000).alias("t_ini_s"),
        ((pl.col("janela").max().over("segmento") - pl.col("janela")).dt.total_milliseconds() / 1000).alias("t_fim_s"),
    ] + leads)
    grade = grade.with_columns(
        ((pl.col("t_ini_s") >= cfg.margem_inicio_s) & (pl.col("t_fim_s") >= cfg.margem_fim_s)
         & pl.col("ret_ticks").is_not_null() & pl.col("mid").is_not_null()).alias("valido")
    )
    rel.log(f"[grade] Grade completa: {grade.height:,} janelas de {cfg.dt_ms} ms "
            f"({grade['valido'].sum():,} válidas) em {time.time() - t0:.1f}s")
    return grade


def agregar_intervalo(grade: pl.DataFrame, every: str) -> pl.DataFrame:
    return (
        grade.with_columns(pl.col("janela").dt.truncate(every).alias("intervalo"))
        .group_by(["segmento", "dia", "intervalo"])
        .agg([
            pl.col("mid").last(),
            pl.col("ret_ticks").fill_null(0).sum().alias("dmid"),
            pl.col("ret_ticks").fill_null(0).pow(2).sum().alias("rv2"),
            pl.col("depth").mean(),
            pl.col("spread_ticks").mean(),
            pl.col("valido").all().alias("valido"),
            pl.len().alias("n_janelas"),
        ] + [pl.col(c).sum() for c in SOMAS + ["n_linhas"]])
        .sort(["segmento", "intervalo"])
    )


# =============================================================================
# Etapa 0: diagnósticos
# =============================================================================
def diagnosticos(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[0] DIAGNÓSTICOS DA BASE\n" + "=" * 78)
    seg = (grade.group_by(["dia", "segmento"])
           .agg(pl.col("janela").min().alias("inicio"), pl.col("janela").max().alias("fim"),
                pl.len().alias("janelas"), pl.col("valido").sum().alias("validas"),
                (pl.col("n_linhas") > 0).mean().alias("ocupacao"))
           .sort("segmento").to_pandas().set_index("segmento"))
    rel.log(f"Dias: {grade['dia'].n_unique()} | Segmentos contínuos: {len(seg)}")
    rel.tabela("diag_segmentos", seg)

    ocup = (grade["n_linhas"] > 0).mean()
    rel.log(f"\nOcupação das janelas de {cfg.dt_ms} ms (com ao menos 1 registro): {ocup:.1%}")
    rel.log("  -> na versão anterior, só as janelas ocupadas existiam; os horizontes eram em tempo de evento.")

    s = grade.select([pl.col(c).sum() for c in ("v_lit", "v_rl", "v_lrl", "v_cls", "v_total", "v_misto")]).row(0, named=True)
    rel.log("\nComposição do volume:")
    for c in ("v_lit", "v_rl", "v_lrl"):
        rel.log(f"  {c:6s}: {s[c] / max(s['v_cls'], 1):.2%}")
    rel.log(f"  Soma das condições / negocio_qtd: {s['v_cls'] / max(s['v_total'], 1):.4f}  (deve ser ~1)")
    rel.log(f"  Volume em linhas com condições mistas (alocado proporcionalmente): {s['v_misto'] / max(s['v_cls'], 1):.2%}")

    v = grade.filter(pl.col("valido"))
    corr = np.corrcoef(v["exec_liq"].to_numpy(), v["ti"].to_numpy())[0, 1]
    rel.log(f"\nCorrelação (janela) entre execuções no topo (ask-bid) e TI: {corr:.3f}")
    rel.log("  -> mede a sobreposição mecânica entre OFI_total e TI; usar ofi_cot quando os dois entram juntos.")

    sp = v["spread_ticks"].to_numpy()
    rel.log("\nSpread em ticks (ponderado por tempo):")
    for lab, m in (("1 tick", sp <= 1.0001), ("2 ticks", (sp > 1.0001) & (sp <= 2.0001)), (">= 3 ticks", sp > 2.0001)):
        rel.log(f"  {lab:10s}: {m.mean():.2%}")
    rel.log(f"Janelas de {cfg.dt_ms} ms sem nenhum evento de QID (impr+deter=0): "
            f"{((v['impr'] + v['deter']) == 0).mean():.2%}")


# =============================================================================
# Etapa H1: OFI contemporâneo (Cont, Kukanov & Stoikov)
# =============================================================================
def h1_contemporaneo(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + f"\n[H1] OFI CONTEMPORÂNEO — intervalos de {cfg.h1_intervalo_s}s, "
            f"regressões por bloco de {cfg.h1_bloco_min} min\n" + "=" * 78)
    iv = (agregar_intervalo(grade, f"{cfg.h1_intervalo_s}s").filter(pl.col("valido"))
          .with_columns(pl.col("intervalo").dt.truncate(f"{cfg.h1_bloco_min}m").alias("bloco")))
    pdf = iv.select(["segmento", "dia", "bloco", "dmid", "ofi_total", "ofi_cot", "ti", "depth"]).to_pandas()
    modelos = {"OFI_total": ["ofi_total"], "OFI_cot": ["ofi_cot"], "TI": ["ti"], "OFI_cot+TI": ["ofi_cot", "ti"]}
    linhas = []
    for (sg, bl), g in pdf.groupby(["segmento", "bloco"]):
        y = g["dmid"].to_numpy(float)
        if len(g) < cfg.h1_min_obs or y.var() == 0:
            continue
        rec = dict(segmento=sg, dia=g["dia"].iloc[0], bloco=bl, n=len(g), depth=g["depth"].mean())
        for nome, xs in modelos.items():
            b, r2 = ols_np(y, g[xs].to_numpy(float))
            rec[f"r2_{nome}"] = r2
            for i, x in enumerate(xs):
                rec[f"b_{nome}_{x}"] = b[i + 1]
        linhas.append(rec)
    res = pd.DataFrame(linhas)
    if res.empty:
        rel.log("Sem blocos suficientes.")
        return
    rel.tabelas["h1_blocos"] = res
    r2c = [c for c in res.columns if c.startswith("r2_")]
    resumo = res[r2c].describe(percentiles=[0.25, 0.5, 0.75]).T[["count", "25%", "50%", "75%"]]
    rel.log("R² contemporâneo por bloco (quartis):")
    rel.tabela("h1_resumo_r2", resumo)
    rel.log("  Interpretação: H1 (forma canônica) prevê R²(OFI) substancialmente > R²(TI).")

    b = res["b_OFI_total_ofi_total"]
    ok = (b > 0) & (res["depth"] > 0)
    if ok.sum() >= 5:
        x = np.log(res.loc[ok, "depth"].to_numpy())
        y = np.log(b[ok].to_numpy())
        m = sm.OLS(y, sm.add_constant(x)).fit()
        rel.log(f"\nCKS: log(beta_OFI) = a + c*log(profundidade): c = {m.params[1]:.3f} "
                f"(ep {m.bse[1]:.3f}, n = {ok.sum()} blocos). Teoria: c ≈ -1.")


# =============================================================================
# Etapa preditiva (H1/H2) com Fama-MacBeth entre dias
# =============================================================================
def preditivo_fm(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[PRED] REGRESSÕES PREDITIVAS POR DIA — inferência Fama-MacBeth\n" + "=" * 78)
    xcols = ["ofi_cot", "ti_lit", "ti_rl", "ti_lrl", "ret_ticks"]
    v = grade.filter(pl.col("valido"))
    dp = v.select([pl.col(c).std().alias(c) for c in xcols]).row(0, named=True)
    rel.log("Regressores padronizados (coef. = ticks por 1 desvio-padrão). ret_ticks = retorno da própria janela.")
    for h in cfg.horizontes_s:
        y = f"fwd_{h}s"
        linhas = []
        for sub in v.select(["dia", y] + xcols).drop_nulls().partition_by("dia"):
            Y = sub[y].to_numpy()
            X = np.column_stack([sub[c].to_numpy() / dp[c] if dp[c] else sub[c].to_numpy() for c in xcols])
            b, r2 = ols_np(Y, X)
            rec = dict(dia=sub["dia"][0], r2=r2, **{c: b[i + 1] for i, c in enumerate(xcols)})
            rec["lit_menos_rl"] = rec["ti_lit"] - rec["ti_rl"]
            rec["lit_menos_lrl"] = rec["ti_lit"] - rec["ti_lrl"]
            linhas.append(rec)
        porDia = pd.DataFrame(linhas).set_index("dia")
        rel.tabelas[f"pred_por_dia_{h}s"] = porDia
        rel.log(f"\nHorizonte {h}s — R² mediano entre dias: {porDia['r2'].median():.5f}")
        rel.tabela(f"pred_fm_{h}s", resumo_fm(porDia, xcols + ["lit_menos_rl", "lit_menos_lrl"]))


# =============================================================================
# Etapa impacto de preço por tipo de negócio (H2 na forma de seleção adversa)
# =============================================================================
def impacto_por_tipo(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> pd.DataFrame | None:
    rel.log("\n" + "=" * 78 + "\n[IMP] IMPACTO DE PREÇO POR TIPO DE NEGÓCIO (janelas 'puras')\n" + "=" * 78)
    rel.log("PI_h = sinal(TI) x (mid_{t+h} - mid_{t-1}) em ticks, ponderado por volume. h=0 é o impacto imediato.")
    g = grade.filter(pl.col("valido") & (pl.col("ti") != 0)).with_columns(
        pl.when((pl.col("v_lit") > 0) & (pl.col("v_rl") == 0) & (pl.col("v_lrl") == 0)).then(pl.lit("Lit"))
        .when((pl.col("v_rl") > 0) & (pl.col("v_lit") == 0) & (pl.col("v_lrl") == 0)).then(pl.lit("RL"))
        .when((pl.col("v_lrl") > 0) & (pl.col("v_lit") == 0) & (pl.col("v_rl") == 0)).then(pl.lit("L_RL"))
        .otherwise(pl.lit("misto")).alias("classe"),
        pl.col("ti").sign().alias("D"),
    )
    cob = g.group_by("classe").agg(pl.col("v_cls").sum()).to_pandas().set_index("classe")["v_cls"]
    rel.log("Fração do volume (em janelas com TI≠0) coberta por janelas puras: "
            + ", ".join(f"{k}={cob.get(k, 0) / cob.sum():.1%}" for k in ("Lit", "RL", "L_RL", "misto")))

    g = g.filter(pl.col("classe") != "misto")
    hs = [("0s", "ret_ticks")] + [(f"{h}s", f"pi_{h}s") for h in cfg.horizontes_s]
    tabs = []
    for lab, col in hs:
        t = (g.filter(pl.col(col).is_not_null())
             .group_by(["dia", "classe"])
             .agg(((pl.col("D") * pl.col(col) * pl.col("v_cls")).sum() / pl.col("v_cls").sum()).alias("pi"))
             .with_columns(pl.lit(lab).alias("h")))
        tabs.append(t)
    dia = pl.concat(tabs).to_pandas()
    largo = dia.pivot_table(index=["h", "dia"], columns="classe", values="pi")
    rel.tabelas["impacto_por_dia"] = largo
    saida = []
    for lab, _ in hs:
        if lab not in largo.index.get_level_values(0):
            continue
        d = largo.loc[lab].copy()
        for c in ("Lit", "RL", "L_RL"):
            if c not in d:
                d[c] = np.nan
        d["Lit-RL"] = d["Lit"] - d["RL"]
        d["Lit-L_RL"] = d["Lit"] - d["L_RL"]
        r = resumo_fm(d, ["Lit", "RL", "L_RL", "Lit-RL", "Lit-L_RL"])[["media", "t", "p", "n"]]
        r.insert(0, "h", lab)
        saida.append(r)
    res = pd.concat(saida)
    rel.tabela("impacto_fm", res)
    rel.log("  Leitura: se RL ≈ 0 já em h=0 e permanece ≈ 0, o 'baixo impacto' pode ser mecânico (não toca o livro).")
    rel.log("  A evidência de seleção adversa está no PI em horizontes longos (60s, 300s), não em h=0.")
    return largo


# =============================================================================
# Etapa VAR de Hasbrouck
# =============================================================================
def hasbrouck_var(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> pd.DataFrame | None:
    rel.log("\n" + "=" * 78 + f"\n[VAR] HASBROUCK — intervalos de {cfg.var_intervalo_ms} ms, {cfg.var_lags} lags, "
            f"impacto acumulado em {cfg.var_horizonte_ir} passos\n" + "=" * 78)
    iv = agregar_intervalo(grade, f"{cfg.var_intervalo_ms}ms").filter(pl.col("valido"))
    ordem = FLUXOS_VAR + ["dmid"]
    linhas = []
    for sub in iv.partition_by("segmento"):
        sub = sub.sort("intervalo")
        if sub.height < 30 * cfg.var_lags:
            continue
        Y = sub.select(ordem).to_numpy().astype(float)
        sd = Y.std(0)
        usar = [i for i in range(len(ordem)) if sd[i] > 0]
        if len(ordem) - 1 not in usar:
            continue
        nomes = [ordem[i] for i in usar]
        Z = (Y[:, usar] - Y[:, usar].mean(0)) / sd[usar]
        res = VAR(Z).fit(maxlags=cfg.var_lags, ic=None)
        psi = res.ma_rep(cfg.var_horizonte_ir)
        C = psi.sum(0)
        C_meio = psi[: cfg.var_horizonte_ir // 2 + 1].sum(0)
        S = np.asarray(res.sigma_u)
        k = len(nomes)
        ir = nomes.index("dmid")

        def partes(perm):
            perm = list(perm)
            P = np.linalg.cholesky(S[np.ix_(perm, perm)])
            c = C[ir, perm] @ P
            s = c ** 2 / (c ** 2).sum()
            out = np.empty(k)
            out[perm] = s
            return out

        hasb = partes(range(k))                                  # fluxos primeiro, retorno por último
        retp = partes([ir] + [i for i in range(k) if i != ir])   # retorno primeiro (como na versão anterior)
        todas = np.array([partes(p) for p in itertools.permutations(range(k))])
        conv = np.max(np.abs(C[ir] - C_meio[ir])) / max(np.max(np.abs(C[ir])), 1e-12)
        rec = dict(segmento=sub["segmento"][0], dia=sub["dia"][0], n=sub.height, convergencia_ir=conv)
        for j, nm in enumerate(nomes):
            rec[f"{nm}_hasb"] = hasb[j]
            rec[f"{nm}_retprimeiro"] = retp[j]
            rec[f"{nm}_min"] = todas[:, j].min()
            rec[f"{nm}_max"] = todas[:, j].max()
        rec["fluxos_total_hasb"] = sum(hasb[j] for j in range(k) if j != ir)
        linhas.append(rec)
    if not linhas:
        rel.log("Nenhum segmento com observações suficientes.")
        return None
    seg = pd.DataFrame(linhas)
    rel.tabelas["var_segmentos"] = seg
    if seg["convergencia_ir"].max() > 0.01:
        rel.log(f"AVISO: impacto acumulado não convergiu (máx. variação {seg['convergencia_ir'].max():.3f}); "
                "aumente var_horizonte_ir.")

    cols = [c for c in seg.columns if c.endswith(("_hasb", "_retprimeiro", "_min", "_max"))]
    dia = seg.groupby("dia").apply(lambda d: pd.Series(np.average(d[cols], axis=0, weights=d["n"]), index=cols),
                                   include_groups=False)
    rel.tabelas["var_por_dia"] = dia
    rel.log("Participação na variância do componente de passeio aleatório (information share), média entre dias:")
    tab = pd.DataFrame({
        "hasbrouck(fluxos 1º)": [dia.get(f"{v}_hasb", pd.Series(dtype=float)).mean() for v in ordem],
        "retorno 1º": [dia.get(f"{v}_retprimeiro", pd.Series(dtype=float)).mean() for v in ordem],
        "mín. ordenações": [dia.get(f"{v}_min", pd.Series(dtype=float)).mean() for v in ordem],
        "máx. ordenações": [dia.get(f"{v}_max", pd.Series(dtype=float)).mean() for v in ordem],
    }, index=ordem)
    rel.tabela("var_resumo", tab, fmt="{:.4f}")
    rel.log("\nInferência Fama-MacBeth (ordenação de Hasbrouck):")
    rel.tabela("var_fm", resumo_fm(dia, [f"{v}_hasb" for v in ordem if f"{v}_hasb" in dia] + ["fluxos_total_hasb"])
               if "fluxos_total_hasb" in dia else resumo_fm(dia, [f"{v}_hasb" for v in ordem if f"{v}_hasb" in dia]))
    rel.log("  Leitura: a distância entre 'retorno 1º' e 'hasbrouck' mostra quanto o resultado anterior era artefato de ordenação.")
    return dia


# =============================================================================
# Regimes de volatilidade (teste de permutação exato entre dias)
# =============================================================================
def regimes(grade: pl.DataFrame, var_dia: pd.DataFrame | None, imp: pd.DataFrame | None,
            cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[REG] REGIMES DE VOLATILIDADE — permutação exata entre dias\n" + "=" * 78)
    rv = (grade.filter(pl.col("valido")).group_by("dia")
          .agg((pl.col("ret_ticks").pow(2).sum().sqrt()).alias("rv")).sort("dia").to_pandas().set_index("dia"))
    rv["alta"] = rv["rv"] > rv["rv"].median()
    rel.log("Dias de alta volatilidade: " + ", ".join(str(d)[:10] for d in rv.index[rv["alta"]]))
    rel.log("AVISO: a classificação usa a própria volatilidade do dia (endógena); trate como descritivo.")
    linhas = []
    if var_dia is not None:
        for c in [c for c in var_dia.columns if c.endswith("_hasb")]:
            d = var_dia[c].reindex(rv.index)
            obs, p = teste_permutacao(d.to_numpy(float), rv["alta"].to_numpy())
            linhas.append(dict(medida=f"VAR {c}", media_alta=d[rv["alta"]].mean(), media_baixa=d[~rv["alta"]].mean(),
                               diferenca=obs, p_perm=p))
    if imp is not None:
        hmax = f"{max(cfg.horizontes_s)}s"
        if hmax in imp.index.get_level_values(0):
            d0 = imp.loc[hmax].reindex(rv.index)
            for c in [c for c in ("Lit", "RL", "L_RL") if c in d0]:
                obs, p = teste_permutacao(d0[c].to_numpy(float), rv["alta"].to_numpy())
                linhas.append(dict(medida=f"PI {hmax} {c}", media_alta=d0[c][rv["alta"]].mean(),
                                   media_baixa=d0[c][~rv["alta"]].mean(), diferenca=obs, p_perm=p))
    if linhas:
        rel.tabela("regimes", pd.DataFrame(linhas).set_index("medida"))


# =============================================================================
# QID, QIDres, H3 e Eq. 7
# =============================================================================
def qid_h3_eq7(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + f"\n[QID] QID em intervalos de {cfg.qid_intervalo} (definição: {cfg.qid_definicao}, "
            f"mín. {cfg.qid_min_eventos} eventos)\n" + "=" * 78)
    n = cfg.reversao_horizonte
    iv = agregar_intervalo(grade, cfg.qid_intervalo)
    tot = pl.col("impr") + pl.col("deter")
    iv = iv.with_columns([
        pl.when(tot >= cfg.qid_min_eventos).then((pl.col("impr") - pl.col("deter")) / tot).alias("qid_dif"),
        pl.when((tot >= cfg.qid_min_eventos) & (pl.col("deter") > 0)).then(pl.col("impr") / pl.col("deter")).alias("qid_razao"),
        pl.when(pl.col("rv2") > 0).then(pl.col("rv2").sqrt().log()).alias("log_rv"),
        (pl.col("n_linhas") + 1).log().alias("log_eventos"),
        (pl.col("v_total") + 1).log().alias("log_volume"),
        (pl.col("spread_ticks") * cfg.tick / pl.col("mid")).log().alias("log_spread_pct"),
        pl.col("intervalo").dt.hour().alias("hora"),
    ]).with_columns(pl.col("qid_dif" if cfg.qid_definicao == "diferenca" else "qid_razao").alias("QID"))
    iv = iv.with_columns([
        pl.col("log_rv").shift(-1).over("segmento").alias("y_log_rv"),
        pl.col("valido").shift(-1).over("segmento").alias("valido_prox"),
        pl.col("log_rv").rolling_mean(5).over("segmento").alias("har5"),
        pl.col("log_rv").rolling_mean(30).over("segmento").alias("har30"),
        ((pl.col("mid").shift(-n) - pl.col("mid")).over("segmento") / cfg.tick).alias("CAR"),
        pl.col("valido").shift(-n).over("segmento").alias("valido_car"),
    ])
    pdf = iv.to_pandas()
    pdf = pdf[pdf["valido"]].copy()
    rel.log(f"Intervalos válidos: {len(pdf):,} | com QID definido: {pdf['QID'].notna().mean():.1%}")

    # ----- QIDres (Eq. 2-3 adaptadas) -----
    q = pdf.dropna(subset=["QID", "log_spread_pct"])
    m = sm.OLS(q["QID"], sm.add_constant(q["log_spread_pct"])).fit()
    pdf.loc[q.index, "QIDres"] = -(q["QID"] - m.fittedvalues) / q["QID"].std()
    rel.log(f"QIDres: R² de QID ~ ln(%spread) = {m.rsquared:.4f} | desvio-padrão de ln(%spread) = "
            f"{q['log_spread_pct'].std():.4f} | corr(QIDres, QID) = {pdf['QIDres'].corr(pdf['QID']):.4f}")
    rel.log("  Se corr ≈ -1, a 'ortogonalização' é vazia: QIDres é só -QID padronizado (spread quase constante).")

    # ----- H3 -----
    rel.log("\n--- H3: volatilidade realizada do próximo intervalo (log) ---")
    d = pdf[pdf["valido_prox"].fillna(False).astype(bool)].dropna(
        subset=["y_log_rv", "log_rv", "har5", "har30", "QID", "QIDres", "log_eventos", "log_volume", "spread_ticks"]).copy()
    if len(d) < 100:
        rel.log("Observações insuficientes para H3.")
        return
    dum = pd.get_dummies(d["hora"], prefix="h", drop_first=True, dtype=float)
    d = pd.concat([d, dum], axis=1)
    base = ["log_rv", "har5", "har30", "log_eventos", "log_volume", "spread_ticks"] + list(dum.columns)
    modelos = {
        "M0 QID+spread (como antes)": ["QID", "spread_ticks"],
        "M1 HAR+controles": base,
        "M2 HAR+controles+QID": base + ["QID"],
        "M3 HAR+controles+QIDres": base + ["QIDres"],
    }
    linhas = []
    for nome, xs in modelos.items():
        r = sm.OLS(d["y_log_rv"], sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": cfg.hac_lags_qid})
        var = "QIDres" if "QIDres" in xs else ("QID" if "QID" in xs else None)
        linhas.append(dict(modelo=nome, r2_aj=r.rsquared_adj,
                           coef=r.params.get(var, np.nan) if var else np.nan,
                           t_hac=r.tvalues.get(var, np.nan) if var else np.nan,
                           p=r.pvalues.get(var, np.nan) if var else np.nan, n=int(r.nobs)))
    rel.tabela("h3_insample", pd.DataFrame(linhas).set_index("modelo"))
    rel.log("  Hipótese original (H3): QUEDA do QID antecede volatilidade -> coeficiente de QID NEGATIVO em M2.")

    # Fama-MacBeth para o coeficiente de QID em M2
    fm = []
    for dia, g in d.groupby("dia"):
        if len(g) < 30:
            continue
        b, _ = ols_np(g["y_log_rv"].to_numpy(float), g[modelos["M2 HAR+controles+QID"]].to_numpy(float))
        fm.append(dict(dia=dia, QID=b[-1]))
    if fm:
        rel.log("Fama-MacBeth (coef. de QID em M2, por dia):")
        rel.tabela("h3_fm", resumo_fm(pd.DataFrame(fm).set_index("dia"), ["QID"]))

    # Fora da amostra
    dias = sorted(d["dia"].unique())
    n_tr = _n_treino(cfg, len(dias))
    tr, te = d[d["dia"].isin(dias[:n_tr])], d[d["dia"].isin(dias[n_tr:])]
    if len(te) > 50:
        rel.log(f"\nFora da amostra: treino = {n_tr} dias, teste = {len(dias) - n_tr} dias")
        sst = ((te["y_log_rv"] - tr["y_log_rv"].mean()) ** 2).sum()
        erros, oos = {}, []
        for nome, xs in modelos.items():
            r = sm.OLS(tr["y_log_rv"], sm.add_constant(tr[xs])).fit()
            pred = r.predict(sm.add_constant(te[xs], has_constant="add"))
            e = (te["y_log_rv"] - pred).to_numpy()
            erros[nome] = e
            oos.append(dict(modelo=nome, r2_oos=1 - (e ** 2).sum() / sst))
        rel.tabela("h3_oos", pd.DataFrame(oos).set_index("modelo"))
        dm_linhas = []
        for novo in ("M2 HAR+controles+QID", "M3 HAR+controles+QIDres"):
            t, p = diebold_mariano(erros["M1 HAR+controles"], erros[novo], cfg.hac_lags_qid)
            rel.log(f"Diebold-Mariano {novo} vs M1: t = {t:.3f}, p (unilateral, novo melhor) = {p:.4f}")
            dm_linhas.append(dict(modelo=novo, dm_t=t, p_unilateral_novo_melhor=p))
        rel.tabelas["h3_dm"] = pd.DataFrame(dm_linhas).set_index("modelo")

    # ----- Eq. 7 (reversão) -----
    rel.log(f"\n--- Eq. 7 adaptada: CAR(t, t+{n}) = a + b·AR_t + c·QIDres_t + d·QIDres_t·AR_t ---")
    rel.log("  CONFERIR com o artigo: definição de AR, janela do CAR e frequência de QIDres.")
    e7 = pdf[pdf["valido_car"].fillna(False).astype(bool)].dropna(subset=["CAR", "dmid", "QIDres"]).copy()
    e7["QIDres_x_AR"] = e7["QIDres"] * e7["dmid"]
    xs = ["dmid", "QIDres", "QIDres_x_AR"]
    if len(e7) > 100:
        r = sm.OLS(e7["CAR"], sm.add_constant(e7[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": n})
        rel.tabela("eq7_pooled", pd.DataFrame({"coef": r.params, "t_hac": r.tvalues, "p": r.pvalues}))
        fm = []
        for dia, g in e7.groupby("dia"):
            if len(g) >= 30:
                b, _ = ols_np(g["CAR"].to_numpy(float), g[xs].to_numpy(float))
                fm.append(dict(dia=dia, **{x: b[i + 1] for i, x in enumerate(xs)}))
        if fm:
            rel.log("Fama-MacBeth entre dias:")
            rel.tabela("eq7_fm", resumo_fm(pd.DataFrame(fm).set_index("dia"), xs))


# =============================================================================
# Etapa PRED_QI: previsão com controle de desequilíbrio de filas
# =============================================================================
def preditivo_qi(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[PRED_QI] PREVISÃO COM CONTROLE DE DESEQUILÍBRIO DE FILAS — Fama-MacBeth\n" + "=" * 78)
    rel.log("qi = (bid_qtd - ask_qtd)/(bid_qtd + ask_qtd) no fim da janela; "
            "ofi_hist = soma de ofi_cot nos 10 s anteriores (persistência do fluxo).")
    n10 = int(round(10_000 / cfg.dt_ms))
    v = (grade.with_columns(pl.col("ofi_cot").shift(1).rolling_sum(n10).over("segmento").alias("ofi_hist"))
         .filter(pl.col("valido")))
    xb = ["ofi_cot", "ti_lit", "ti_rl", "ti_lrl", "ret_ticks"]
    modelos = {
        "base": xb,
        "so_qi": ["qi", "ret_ticks"],
        "+qi": xb + ["qi"],
        "+qi+hist": xb + ["qi", "ofi_hist"],
    }
    todas = sorted({c for xs in modelos.values() for c in xs})
    dp = v.select([pl.col(c).std().alias(c) for c in todas]).row(0, named=True)
    rel.log("Desvios-padrão (unidades originais): "
            + ", ".join(f"{c}={dp[c]:.4g}" for c in ("ofi_cot", "ti_lit", "qi", "ofi_hist")))
    rel.log("Coeficientes em ticks por 1 desvio-padrão do regressor.\n")

    resumo = []
    for h in cfg.horizontes_s:
        y = f"fwd_{h}s"
        dias = v.select(["dia", y] + todas).drop_nulls().partition_by("dia")
        for nome, xs in modelos.items():
            linhas = []
            for sub in dias:
                X = np.column_stack([sub[c].to_numpy() / (dp[c] or 1.0) for c in xs])
                b, r2 = ols_np(sub[y].to_numpy(), X)
                linhas.append(dict(dia=sub["dia"][0], r2=r2, **{c: b[i + 1] for i, c in enumerate(xs)}))
            pdia = pd.DataFrame(linhas).set_index("dia")
            rel.tabelas[f"predqi_por_dia_{h}s_{nome}"] = pdia
            fm = resumo_fm(pdia, xs)
            rec = dict(h=f"{h}s", modelo=nome, r2_mediano=pdia["r2"].median())
            for c in ("ofi_cot", "ti_lit", "qi", "ofi_hist"):
                if c in fm.index:
                    rec[f"{c}_coef"] = fm.loc[c, "media"]
                    rec[f"{c}_t"] = fm.loc[c, "t"]
            resumo.append(rec)
    tab = pd.DataFrame(resumo).set_index(["h", "modelo"])
    rel.tabela("predqi_resumo", tab, fmt="{:.4g}")
    rel.log("\n  Leitura: se ofi_cot_coef cai muito de 'base' para '+qi', a previsibilidade é mecânica de fila.")
    rel.log("  Se cai de '+qi' para '+qi+hist', reflete persistência do fluxo (ex.: fatiamento de ordens).")
    rel.log("  O que sobrar em 30 s e 300 s é o candidato a conteúdo informacional.")


# =============================================================================
# Etapa LINHA: impacto de preço por tipo de negócio no nível do milissegundo
# =============================================================================
def _nome(d: float) -> str:
    return f"{d:g}s"


def _faixa_tamanho() -> pl.Expr:
    return (pl.when(pl.col("v_cls") <= 1).then(pl.lit("1"))
            .when(pl.col("v_cls") <= 5).then(pl.lit("2-5"))
            .when(pl.col("v_cls") <= 20).then(pl.lit("6-20")).otherwise(pl.lit("21+")).alias("tamanho"))


def _fm_grupos(df: pl.DataFrame, cols_pi: list[tuple[str, str]], grupo_col: str, grupos: tuple,
               difs: tuple = (), extra: str | None = None) -> pd.DataFrame:
    """Média ponderada por volume de cada coluna por (dia, grupo[, extra]) e Fama-MacBeth entre dias."""
    chaves = ["dia", grupo_col] + ([extra] if extra else [])
    tabs = []
    for lab, col in cols_pi:
        if col not in df.columns:
            continue
        tabs.append(df.filter(pl.col(col).is_not_null()).group_by(chaves)
                    .agg(((pl.col(col) * pl.col("v_cls")).sum() / pl.col("v_cls").sum()).alias("pi"))
                    .with_columns(pl.lit(lab).alias("h")))
    if not tabs:
        return pd.DataFrame()
    largo = pl.concat(tabs).to_pandas()
    extras = [None] if not extra else sorted(largo[extra].dropna().unique())
    saida = []
    for lab, _ in cols_pi:
        for g in extras:
            m = largo["h"] == lab
            if g is not None:
                m &= largo[extra] == g
            d = largo[m].pivot_table(index="dia", columns=grupo_col, values="pi")
            for c in grupos:
                if c not in d:
                    d[c] = np.nan
            nomes = list(grupos)
            for a, b in difs:
                d[f"{a}-{b}"] = d[a] - d[b]
                nomes.append(f"{a}-{b}")
            r = resumo_fm(d, nomes)[["media", "ic95_inf", "ic95_sup", "t", "p", "n"]]
            r.insert(0, "h", lab)
            if g is not None:
                r.insert(0, extra, g)
            saida.append(r)
    return pd.concat(saida) if saida else pd.DataFrame()


def _pi_classes_fm(df: pl.DataFrame, cols_pi: list[tuple[str, str]], grupo_extra: str | None = None) -> pd.DataFrame:
    return _fm_grupos(df, cols_pi, "classe", ("Lit", "RL", "L_RL"),
                      difs=(("Lit", "RL"), ("Lit", "L_RL")), extra=grupo_extra)


def impacto_por_linha(cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[LINHA] IMPACTO DE PREÇO POR TIPO DE NEGÓCIO — nível do milissegundo\n" + "=" * 78)
    t0 = time.time()
    conds = list(cfg.cols_lit) + list(cfg.cols_rl) + list(cfg.cols_lrl)
    lf = _lazy_continuo(cfg, ["mid_simples", "negocio_qtd_agr_compra", "negocio_qtd_agr_venda",
                              "bid_execucao_qtd", "ask_execucao_qtd"] + conds)
    seg = ["dia", "seg_n"]

    # série completa (todas as linhas) para buscar mid/bid/ask em t + atraso
    serie = lf.select(seg + ["timestamp", "mid_simples", "bid_preco", "ask_preco"]).collect().sort("timestamp")
    limites = serie.group_by(seg).agg(pl.col("timestamp").min().alias("seg_ini"),
                                      pl.col("timestamp").max().alias("seg_fim"))
    bruto = {c: f"r{i}" for i, c in enumerate(conds)}

    # linhas com negócio, com o estado do livro na linha anterior
    tr = (
        lf.with_columns([pl.col(c).shift(1).over(seg).alias(f"{c}_pre") for c in ("mid_simples", "bid_preco", "ask_preco")])
        .filter(_soma(conds) > 0)
        .with_columns([
            _f0("negocio_qtd_agr_compra").alias("agr_c"),
            _f0("negocio_qtd_agr_venda").alias("agr_v"),
            _soma(cfg.cols_lit).alias("v_lit"),
            _soma(cfg.cols_rl).alias("v_rl"),
            _soma(cfg.cols_lrl).alias("v_lrl"),
            _f0("bid_execucao_qtd").alias("exe_b"),
            _f0("ask_execucao_qtd").alias("exe_a"),
        ] + [_f0(c).alias(a) for c, a in bruto.items()])
        .select(seg + ["timestamp", "mid_simples", "mid_simples_pre", "bid_preco_pre", "ask_preco_pre",
                       "agr_c", "agr_v", "v_lit", "v_rl", "v_lrl", "exe_b", "exe_a"] + list(bruto.values()))
        .rename({"mid_simples_pre": "mid_pre", "bid_preco_pre": "bid_pre", "ask_preco_pre": "ask_pre"})
        .collect()
        .join(limites, on=seg, how="left")
        .with_columns((pl.col("v_lit") + pl.col("v_rl") + pl.col("v_lrl")).alias("v_cls"))
        .filter(((pl.col("timestamp") - pl.col("seg_ini")).dt.total_seconds() >= cfg.margem_inicio_s)
                & ((pl.col("seg_fim") - pl.col("timestamp")).dt.total_seconds() >= cfg.margem_fim_s))
        .with_columns([
            ((pl.col("v_lit") > 0).cast(pl.Int8) + (pl.col("v_rl") > 0).cast(pl.Int8)
             + (pl.col("v_lrl") > 0).cast(pl.Int8)).alias("n_cls"),
            pl.sum_horizontal([(pl.col(a) > 0).cast(pl.Int8) for a in bruto.values()]).alias("n_bruto"),
            ((pl.col("agr_c") > 0) != (pl.col("agr_v") > 0)).alias("unilateral"),
            pl.when(pl.col("agr_c") > 0).then(1.0).otherwise(-1.0).alias("D"),
            pl.concat_str([pl.when(pl.col("v_lit") > 0).then(pl.lit("Lit")).otherwise(pl.lit("")),
                           pl.when(pl.col("v_rl") > 0).then(pl.lit("+RL")).otherwise(pl.lit("")),
                           pl.when(pl.col("v_lrl") > 0).then(pl.lit("+L_RL")).otherwise(pl.lit(""))]
                          ).str.strip_chars("+").alias("combo"),
        ])
    )
    rel.log(f"Linhas com negócio (após margens): {tr.height:,} | carregadas em {time.time() - t0:.1f}s")

    # ----- diagnóstico das linhas mistas -----
    vt = tr["v_cls"].sum()
    rel.log(f"Soma das condições / (agr_compra + agr_venda): {vt / max(tr['agr_c'].sum() + tr['agr_v'].sum(), 1):.4f}")
    diag = (tr.group_by([pl.col("n_cls") > 1, "unilateral"]).agg(pl.col("v_cls").sum())
            .with_columns((pl.col("v_cls") / vt).alias("frac_volume")).sort(["n_cls", "unilateral"]).to_pandas())
    diag["n_cls"] = diag["n_cls"].map({False: "condição única", True: "mista"})
    diag["unilateral"] = diag["unilateral"].map({True: "agressão de um lado só", False: "compra e venda na linha"})
    rel.log("Volume por tipo de linha:")
    rel.tabela("linha_diag_tipos", diag.set_index(["n_cls", "unilateral"]), fmt="{:.4f}")
    combos = (tr.filter(pl.col("n_cls") > 1).group_by("combo").agg(pl.col("v_cls").sum())
              .with_columns((pl.col("v_cls") / pl.col("v_cls").sum()).alias("frac_mistas")).sort("v_cls", descending=True))
    rel.log("Composição das linhas mistas:")
    rel.tabela("linha_diag_combos", combos.to_pandas().set_index("combo"), fmt="{:.4f}")

    # ----- mid/bid/ask em t + atraso, por as-of -----
    tr = tr.filter(pl.col("unilateral")).sort("timestamp")
    esg = cfg.esgotamento_s
    pontos = sorted(set(cfg.atrasos_s) | set(cfg.horizontes_s) | {esg, 1})
    for d in pontos:
        n = _nome(d)
        cols_r = ["timestamp", "mid_simples"] + (["bid_preco", "ask_preco"] if d == esg else [])
        ren = {"timestamp": "t_alvo", "mid_simples": f"mid_{n}", "bid_preco": f"bid_{n}", "ask_preco": f"ask_{n}"}
        right = serie.select(seg + cols_r).rename({k: v for k, v in ren.items() if k in cols_r})
        novas = [c for c in right.columns if c not in seg + ["t_alvo"]]
        tr = (tr.with_columns((pl.col("timestamp") + pl.duration(milliseconds=int(round(d * 1000)))).alias("t_alvo"))
              .join_asof(right, on="t_alvo", by=seg, strategy="backward")
              .with_columns([pl.when(pl.col("t_alvo") <= pl.col("seg_fim")).then(pl.col(c)).alias(c) for c in novas])
              .drop("t_alvo"))
    del serie

    ne = _nome(esg)
    bases = [b for b in sorted(set(cfg.atrasos_s) | {1}) if b > 0]
    tr = tr.with_columns(
        [(pl.col("D") * (pl.col("mid_simples") - pl.col("mid_pre")) / cfg.tick).alias("pi_0s")]
        + [(pl.col("D") * (pl.col(f"mid_{_nome(d)}") - pl.col("mid_pre")) / cfg.tick).alias(f"pi_{_nome(d)}") for d in pontos]
        + [(pl.col(c) / pl.col("v_cls")).alias(f"s_{c[2:]}") for c in ("v_lit", "v_rl", "v_lrl")]
        + [pl.when(pl.col("D") > 0).then(pl.col(f"ask_{ne}") > pl.col("ask_pre"))
             .otherwise(pl.col(f"bid_{ne}") < pl.col("bid_pre")).alias("esgotou"),
           pl.when(pl.col("D") > 0).then(pl.col("exe_a")).otherwise(pl.col("exe_b")).alias("exec_lado"),
           _faixa_tamanho()]
    ).filter(pl.col("mid_pre").is_not_null())
    tr = tr.with_columns(
        [(pl.col(f"pi_{_nome(h)}") - pl.col("pi_0s")).alias(f"drift_{_nome(h)}") for h in cfg.horizontes_s]
        + [(pl.col(f"pi_{_nome(h)}") - pl.col(f"pi_{_nome(b)}")).alias(f"desde_{_nome(b)}_{_nome(h)}")
           for b in bases for h in cfg.horizontes_s if h > b]
    )
    cols_pi = [("0s", "pi_0s")] + [(_nome(h), f"pi_{_nome(h)}") for h in cfg.horizontes_s]
    cols_drift = [(f"drift_{_nome(h)}", f"drift_{_nome(h)}") for h in cfg.horizontes_s]
    hmax = _nome(max(cfg.horizontes_s))

    # ----- A0: semântica das condições brutas -----
    rel.log("\n--- A0) Semântica das condições: linhas com UMA condição bruta, agressão unilateral ---")
    rel.log(f"  razao_exec_topo = execução no topo do lado agredido / volume do negócio (livro visível ≈ 1; RLP ≈ 0).")
    rel.log(f"  frac_esgotou = fração do volume em que o preço do lado agredido mudou em {ne}.")
    cond = pl.coalesce([pl.when(pl.col(a) > 0).then(pl.lit(c)) for c, a in bruto.items()]).alias("condicao")
    sem = (tr.filter(pl.col("n_bruto") == 1).with_columns(cond).group_by("condicao")
           .agg(pl.len().alias("linhas"), pl.col("v_cls").sum().alias("volume"),
                ((pl.col("exec_lado") > 0).cast(pl.Float64) * pl.col("v_cls")).sum().truediv(pl.col("v_cls").sum())
                .alias("frac_vol_com_exec_topo"),
                (pl.col("exec_lado").sum() / pl.col("v_cls").sum()).alias("razao_exec_topo"),
                (pl.col("esgotou").cast(pl.Float64) * pl.col("v_cls")).sum().truediv(
                    pl.col("v_cls").filter(pl.col("esgotou").is_not_null()).sum()).alias("frac_esgotou"),
                ((pl.col("pi_0s") * pl.col("v_cls")).sum() / pl.col("v_cls").sum()).alias("pi_0s"))
           .sort("volume", descending=True))
    rel.tabela("linha_semantica_condicoes", sem.to_pandas().set_index("condicao"), fmt="{:.4g}")

    # ----- A: linhas de condição única (classes agregadas) -----
    puras = tr.filter(pl.col("n_cls") == 1).with_columns(
        pl.when(pl.col("v_lit") > 0).then(pl.lit("Lit"))
        .when(pl.col("v_rl") > 0).then(pl.lit("RL")).otherwise(pl.lit("L_RL")).alias("classe"))
    cob = puras.group_by("classe").agg(pl.col("v_cls").sum()).to_pandas().set_index("classe")["v_cls"]
    tot = {c: tr[f"v_{k}"].sum() for c, k in (("Lit", "lit"), ("RL", "rl"), ("L_RL", "lrl"))}
    rel.log("\n--- A) Linhas de condição única, agressão unilateral ---")
    rel.log("Cobertura do volume de cada classe: "
            + ", ".join(f"{c}={cob.get(c, 0) / max(tot[c], 1):.1%}" for c in ("Lit", "RL", "L_RL")))
    rel.log("PI_h = sinal do agressor x (mid_{t+h} - mid antes da linha), em ticks, ponderado por volume.")
    rel.tabela("linha_impacto_fm", _pi_classes_fm(puras, cols_pi))

    dist = (puras.group_by(["classe", "tamanho"]).agg(pl.col("v_cls").sum())
            .with_columns((pl.col("v_cls") / pl.col("v_cls").sum().over("classe")).alias("frac"))
            .to_pandas().pivot_table(index="classe", columns="tamanho", values="frac"))
    rel.log("\n--- A2) Distribuição do volume por tamanho da linha (contratos) e impacto por faixa ---")
    rel.tabela("linha_dist_tamanho", dist, fmt="{:.3f}")
    hs_tam = [x for x in cols_pi if x[0] in ("0s", "60s", hmax)]
    rel.tabela("linha_impacto_tamanho", _pi_classes_fm(puras, hs_tam, grupo_extra="tamanho"))

    rel.log("\n--- A3) Deriva pós-negócio a partir do mid da própria linha (PI_h - PI_0) ---")
    rel.tabela("linha_drift_fm", _pi_classes_fm(puras, cols_drift))
    hs_d = [x for x in cols_drift if x[0] in ("drift_60s", f"drift_{hmax}")]
    rel.log("Por faixa de tamanho:")
    rel.tabela("linha_drift_tamanho", _pi_classes_fm(puras, hs_d, grupo_extra="tamanho"))

    # ----- A4: perfil fino do impacto nos primeiros instantes -----
    rel.log("\n--- A4) Perfil do impacto nos primeiros instantes (quando acontece o salto?) ---")
    finos = [("0s", "pi_0s")] + [(_nome(d), f"pi_{_nome(d)}") for d in pontos if d <= 1]
    rel.tabela("linha_perfil_fino", _pi_classes_fm(puras, finos))
    rel.log("Por faixa de tamanho:")
    rel.tabela("linha_perfil_fino_tamanho", _pi_classes_fm(puras, finos, grupo_extra="tamanho"))
    rel.log("  Se o Lit salta quase todo entre 0 s e 0,1 s, a 'deriva de 1 s' da A3 é o mesmo evento registrado")
    rel.log("  em linhas seguintes do feed, não movimento posterior.")

    # ----- A5: deriva a partir de linhas de base atrasadas -----
    rel.log("\n--- A5) Deriva a partir de linha de base atrasada (PI_h - PI_base) ---")
    for b in bases:
        cols_b = [(f"desde_{_nome(b)}_{_nome(h)}", f"desde_{_nome(b)}_{_nome(h)}") for h in cfg.horizontes_s if h > b]
        rel.log(f"Base em t + {_nome(b)}:")
        rel.tabela(f"linha_deriva_desde_{_nome(b)}", _pi_classes_fm(puras, cols_b))
        sel = [x for x in cols_b if x[0].endswith(("_60s", f"_{hmax}"))]
        rel.log(f"Base em t + {_nome(b)}, por faixa de tamanho:")
        rel.tabela(f"linha_deriva_desde_{_nome(b)}_tamanho", _pi_classes_fm(puras, sel, grupo_extra="tamanho"))

    # ----- A6: esgotamento de fila -----
    rel.log(f"\n--- A6) Esgotamento de fila (preço do lado agredido mudou em {ne}) ---")
    fr = (puras.filter(pl.col("esgotou").is_not_null()).group_by(["classe", "tamanho"])
          .agg(((pl.col("esgotou").cast(pl.Float64) * pl.col("v_cls")).sum() / pl.col("v_cls").sum()).alias("frac"))
          .to_pandas().pivot_table(index="classe", columns="tamanho", values="frac"))
    rel.log("Fração do volume que esgotou a fila, por classe e tamanho:")
    rel.tabela("linha_esgotamento_frac", fr, fmt="{:.3f}")
    lit = (puras.filter((pl.col("classe") == "Lit") & pl.col("esgotou").is_not_null())
           .with_columns(pl.when(pl.col("esgotou")).then(pl.lit("esgotou")).otherwise(pl.lit("nao_esgotou")).alias("esg")))
    cols_esg = [("0s", "pi_0s"), ("60s", "pi_60s"), (hmax, f"pi_{hmax}")]
    b0 = _nome(bases[min(1, len(bases) - 1)])
    cols_esg += [(f"desde_{b0}_{n}", f"desde_{b0}_{n}") for n in ("60s", hmax)]
    rel.log("Negócios Lit: impacto e deriva, esgotou vs. não esgotou (todos os tamanhos):")
    rel.tabela("linha_lit_esgotamento", _fm_grupos(lit, cols_esg, "esg", ("esgotou", "nao_esgotou"),
                                                   difs=(("esgotou", "nao_esgotou"),)))
    rel.log("Negócios Lit por faixa de tamanho:")
    rel.tabela("linha_lit_esgotamento_tamanho", _fm_grupos(lit, cols_esg, "esg", ("esgotou", "nao_esgotou"),
                                                           difs=(("esgotou", "nao_esgotou"),), extra="tamanho"))
    rel.log("  Leitura: impacto permanente nos Lit grandes que NÃO esgotaram a fila indica informação;")
    rel.log("  se só os que esgotaram têm impacto, é o salto de preço do esgotamento que persiste.")

    # ----- B: regressão por participação -----
    rel.log("\n--- B) Todas as linhas unilaterais: PI = b_Lit*s_Lit + b_RL*s_RL + b_LRL*s_LRL (MQP, sem constante) ---")
    rel.log("  Hipótese de identificação: o impacto de uma linha mista é a média dos impactos ponderada pelo volume.")
    sc = ["s_lit", "s_rl", "s_lrl"]
    saida = []
    for lab, col in cols_pi:
        linhas = []
        for sub in tr.select(["dia", col, "v_cls"] + sc).drop_nulls().partition_by("dia"):
            w = np.sqrt(sub["v_cls"].to_numpy())
            X = sub.select(sc).to_numpy() * w[:, None]
            b, *_ = np.linalg.lstsq(X, sub[col].to_numpy() * w, rcond=None)
            linhas.append(dict(dia=sub["dia"][0], Lit=b[0], RL=b[1], L_RL=b[2]))
        d = pd.DataFrame(linhas).set_index("dia")
        d["Lit-RL"], d["Lit-L_RL"] = d["Lit"] - d["RL"], d["Lit"] - d["L_RL"]
        r = resumo_fm(d, ["Lit", "RL", "L_RL", "Lit-RL", "Lit-L_RL"])[["media", "ic95_inf", "ic95_sup", "t", "p", "n"]]
        r.insert(0, "h", lab)
        saida.append(r)
    rel.tabela("linha_impacto_participacao", pd.concat(saida))
    rel.log(f"\n[LINHA] concluída em {time.time() - t0:.1f}s")


# =============================================================================
# Etapa FILAS: quais eventos de fila carregam informação
# =============================================================================
COMPONENTES = ["adic", "melh", "canc_tot", "canc_parc", "exec_liq"]
NOMES_COMP = {
    "adic": "ordens novas na fila do topo",
    "melh": "ordens que melhoram o preço (só com spread aberto)",
    "canc_tot": "cancelamentos totais no topo",
    "canc_parc": "cancelamentos parciais no topo",
    "exec_liq": "execuções contra o topo (agressões)",
}


def componentes_fila(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[FILAS] QUAIS EVENTOS DE FILA CARREGAM INFORMAÇÃO\n" + "=" * 78)
    rel.log("Cada componente é medido como pressão compradora: lado da compra menos lado da venda.")
    rel.log("  Ex.: canc_tot > 0 significa que saíram mais ofertas de VENDA do que de compra.")
    for c in COMPONENTES:
        rel.log(f"  {c:9s} = {NOMES_COMP[c]}")
    seg = "segmento"
    lim = cfg.filas_lim_prox_s

    # direção e tempo da PRÓXIMA mudança do mid, depois da janela t
    g = grade.with_columns([
        pl.when(pl.col("ret_ticks") != 0).then(pl.col("mid")).alias("_mid_mud"),
        pl.when(pl.col("ret_ticks") != 0).then(pl.col("janela")).alias("_t_mud"),
    ]).with_columns([
        pl.col("_mid_mud").shift(-1).backward_fill().over(seg).alias("_mid_prox"),
        pl.col("_t_mud").shift(-1).backward_fill().over(seg).alias("_t_prox"),
    ]).with_columns([
        (pl.col("_mid_prox") - pl.col("mid")).sign().alias("dir_prox"),
        ((pl.col("_t_prox") - pl.col("janela")).dt.total_milliseconds() / 1000).alias("seg_ate_prox"),
    ])
    g = g.filter(pl.col("valido")).with_columns(
        [(pl.col(c) / pl.col("depth").clip(lower_bound=1)).alias(f"{c}_rel") for c in COMPONENTES]
    )

    # ----- descrição -----
    ativ = g.select([pl.col(c).abs().sum().alias(c) for c in COMPONENTES]).row(0, named=True)
    tot = sum(ativ.values())
    rel.log("\nParticipação de cada componente no fluxo total do topo (em contratos, valor absoluto):")
    rel.log("  " + ", ".join(f"{c}={ativ[c] / tot:.1%}" for c in COMPONENTES))
    dprox = g["seg_ate_prox"].drop_nulls()
    rel.log(f"Tempo até a próxima mudança de preço: mediana {dprox.median():.2f}s, "
            f"p90 {dprox.quantile(0.9):.1f}s; usadas só janelas com mudança em até {lim}s.")

    # ----- 1) contemporâneo: qual evento acompanha a mudança de preço -----
    rel.log("\n--- 1) Contemporâneo: R² mediano entre dias da variação do mid NA PRÓPRIA janela sobre cada componente ---")
    rel.log("  Mostra qual evento coincide com a mudança de preço (é em parte mecânico).")
    r2c = {c: [] for c in COMPONENTES}
    for sub in g.select(["dia", "ret_ticks"] + COMPONENTES).drop_nulls().partition_by("dia"):
        y = sub["ret_ticks"].to_numpy()
        for c in COMPONENTES:
            r2c[c].append(ols_np(y, sub[c].to_numpy()[:, None])[1])
    rel.tabela("filas_contemporaneo_r2", pd.DataFrame({"r2_mediano": {c: np.nanmedian(v) for c, v in r2c.items()}}),
               fmt="{:.4f}")

    # ----- 2) e 3) preditivo -----
    alvos = [("dir_prox", "direção da próxima mudança de preço (+1/-1)")]
    alvos += [(f"fwd_{h}s", f"variação do mid em {h}s (ticks)") for h in cfg.horizontes_s if h in (1, 30, 300)]
    rel_cols = [f"{c}_rel" for c in COMPONENTES]
    modelos = {
        "componentes": COMPONENTES + ["ret_ticks"],
        "componentes+qi": COMPONENTES + ["qi", "ret_ticks"],
        "relativos+qi": rel_cols + ["qi", "ret_ticks"],
    }
    todas = sorted({c for xs in modelos.values() for c in xs})
    dp = g.select([pl.col(c).std().alias(c) for c in todas]).row(0, named=True)
    rel.log("\n--- 2) Previsão por dia (Fama-MacBeth). Coeficientes por 1 desvio-padrão do regressor ---")
    rel.log("  'relativos' = componente dividido pela profundidade média do topo (o mesmo cancelamento pesa mais com fila curta).")
    rel.log("  Para dir_prox, o coeficiente x 50 ≈ variação em pontos percentuais na probabilidade de a próxima mudança ser de alta.")
    coefs, tstats = [], []
    for alvo, desc in alvos:
        base = g.filter(pl.col("seg_ate_prox") <= lim) if alvo == "dir_prox" else g
        for nome, xs in modelos.items():
            linhas = []
            for sub in base.select(["dia", alvo] + xs).drop_nulls().partition_by("dia"):
                X = np.column_stack([sub[c].to_numpy() / (dp[c] or 1.0) for c in xs])
                b, r2 = ols_np(sub[alvo].to_numpy(), X)
                linhas.append(dict(dia=sub["dia"][0], r2=r2, **{c: b[i + 1] for i, c in enumerate(xs)}))
            pdia = pd.DataFrame(linhas).set_index("dia")
            rel.tabelas[f"filas_por_dia_{alvo}_{nome}"] = pdia
            fm = resumo_fm(pdia, xs)
            ch = dict(alvo=alvo, modelo=nome)
            coefs.append({**ch, "r2_mediano": pdia["r2"].median(),
                          **{c.replace("_rel", ""): fm.loc[c, "media"] for c in xs if c != "ret_ticks"}})
            tstats.append({**ch, **{c.replace("_rel", ""): fm.loc[c, "t"] for c in xs if c != "ret_ticks"}})
    cols_ord = ["r2_mediano"] + COMPONENTES + ["qi"]
    tc = pd.DataFrame(coefs).set_index(["alvo", "modelo"]).reindex(columns=cols_ord)
    tt = pd.DataFrame(tstats).set_index(["alvo", "modelo"]).reindex(columns=COMPONENTES + ["qi"])
    rel.log("\nCoeficientes:")
    rel.tabela("filas_coeficientes", tc, fmt="{:.4f}")
    rel.log("\nEstatísticas t (Fama-MacBeth entre dias):")
    rel.tabela("filas_t", tt, fmt="{:.2f}")
    rel.log("\n  Como ler:")
    rel.log("  - Sinal esperado de todos os componentes: positivo (pressão compradora -> preço sobe).")
    rel.log("  - O componente que continua significativo em 'componentes+qi' carrega informação além do tamanho das filas.")
    rel.log("  - Em fwd_300s, o que sobrevive indica informação permanente, não só mecânica de curto prazo.")
    rel.log("  - Cancelamento significativo e maior que execução sugere informação via retirada de liquidez.")


# =============================================================================
# Etapa RLP_FILAS: o RLP muda a informação das filas do livro visível?
# =============================================================================
def _dir_prox(grade: pl.DataFrame, lim_s: float) -> pl.DataFrame:
    """Direção (+1/-1) da próxima mudança do mid após a janela, se ocorrer em até lim_s segundos."""
    seg = "segmento"
    return (
        grade.with_columns([
            pl.when(pl.col("ret_ticks") != 0).then(pl.col("mid")).alias("_mid_mud"),
            pl.when(pl.col("ret_ticks") != 0).then(pl.col("janela")).alias("_t_mud"),
        ])
        .with_columns([
            pl.col("_mid_mud").shift(-1).backward_fill().over(seg).alias("_mid_prox"),
            pl.col("_t_mud").shift(-1).backward_fill().over(seg).alias("_t_prox"),
        ])
        .with_columns([
            (pl.col("_mid_prox") - pl.col("mid")).sign().alias("dir_prox"),
            ((pl.col("_t_prox") - pl.col("janela")).dt.total_milliseconds() / 1000).alias("seg_ate_prox"),
        ])
        .with_columns(pl.when(pl.col("seg_ate_prox") <= lim_s).then(pl.col("dir_prox")).alias("dir_prox"))
        .drop(["_mid_mud", "_t_mud", "_mid_prox", "_t_prox"])
    )


def rlp_informatividade(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[RLP_FILAS] O RLP MUDA A INFORMAÇÃO DAS FILAS DO LIVRO VISÍVEL?\n" + "=" * 78)
    rel.log("Hipótese: quando mais varejo é desviado para o RLP, o que sobra nas filas visíveis é mais informado;")
    rel.log("logo, desequilíbrio de filas e cancelamentos deveriam prever MELHOR o preço nesses momentos.")
    rel.log(f"Participação do RLP = volume (RL + L_RL) / volume total, em intervalos de {cfg.rlp_intervalo}.")
    seg = "segmento"
    g = _dir_prox(grade, cfg.filas_lim_prox_s).with_columns(
        pl.col("janela").dt.truncate(cfg.rlp_intervalo).alias("intervalo"))

    # ----- variáveis por intervalo -----
    iv = (
        g.group_by([seg, "dia", "intervalo"])
        .agg([
            (pl.col("v_rl") + pl.col("v_lrl")).sum().alias("v_rlp"),
            pl.col("v_cls").sum().alias("v_neg"),
            pl.col("ret_ticks").fill_null(0).pow(2).sum().alias("rv2"),
            pl.col("valido").all().alias("iv_valido"),
        ])
        .sort([seg, "intervalo"])
        .with_columns([
            pl.when(pl.col("v_neg") > 0).then(pl.col("v_rlp") / pl.col("v_neg")).alias("rlp_share"),
            (pl.col("v_neg") + 1).log().alias("log_vol_iv"),
            (pl.col("rv2").sqrt() + 1).log().alias("log_rv_iv"),
            pl.col("intervalo").dt.hour().alias("hora"),
        ])
        .with_columns(pl.col("rlp_share").shift(1).over(seg).alias("rlp_share_lag"))
    )
    # tercis DENTRO de cada dia e hora (compara intervalos do mesmo dia e mesmo horário)
    for var, nome in (("rlp_share_lag", "t_lag"), ("rlp_share", "t_cont")):
        ok = pl.when(pl.col("iv_valido")).then(pl.col(var))
        n = ok.count().over(["dia", "hora"])
        r = ok.rank("ordinal").over(["dia", "hora"])
        iv = iv.with_columns(pl.when(n >= 3).then(((r - 1) * 3 // n + 1).cast(pl.Int8)).alias(nome))

    ivv = iv.filter(pl.col("iv_valido") & pl.col("rlp_share").is_not_null())
    rel.log(f"\nIntervalos válidos: {ivv.height:,} | participação média do RLP: {ivv['rlp_share'].mean():.1%} | "
            f"desvio-padrão entre intervalos do mesmo dia (média): "
            f"{ivv.group_by('dia').agg(pl.col('rlp_share').std())['rlp_share'].mean():.1%}")
    perfil = (ivv.group_by("hora").agg(pl.col("rlp_share").mean().alias("rlp_share_media"), pl.len().alias("intervalos"))
              .sort("hora").to_pandas().set_index("hora"))
    rel.log("Perfil intradiário da participação do RLP:")
    rel.tabela("rlpf_perfil_hora", perfil, fmt="{:.3f}")

    g = g.join(iv.select([seg, "intervalo", "rlp_share", "rlp_share_lag", "log_vol_iv", "log_rv_iv", "t_lag", "t_cont"]),
               on=[seg, "intervalo"], how="left").filter(pl.col("valido"))

    xs = ["qi", "adic", "canc_tot", "exec_liq", "ret_ticks"]
    dp = g.select([pl.col(c).std().alias(c) for c in xs]).row(0, named=True)
    h_alvo = min(cfg.horizontes_s, key=lambda h: abs(h - 30))
    alvos = [("dir_prox", "direção da próxima mudança de preço"), (f"fwd_{h_alvo}s", f"variação do mid em {h_alvo}s")]

    # ----- A) comparação por tercis -----
    for tvar, rotulo in (("t_lag", "participação do RLP no intervalo ANTERIOR (principal)"),
                         ("t_cont", "participação do RLP no PRÓPRIO intervalo (robustez)")):
        rel.log(f"\n--- A) Tercis de {rotulo}, dentro de cada dia e hora ---")
        bal = (iv.filter(pl.col(tvar).is_not_null()).group_by(tvar)
               .agg(pl.col("rlp_share_lag" if tvar == "t_lag" else "rlp_share").mean().alias("rlp_share_usada"),
                    pl.col("rlp_share").mean().alias("rlp_share_no_intervalo"),
                    pl.col("log_vol_iv").mean(), pl.col("log_rv_iv").mean(), pl.len().alias("intervalos"))
               .sort(tvar).to_pandas().set_index(tvar))
        rel.log("Equilíbrio entre tercis (se volume e volatilidade diferem muito, a comparação tem confusão):")
        rel.tabela(f"rlpf_equilibrio_{tvar}", bal, fmt="{:.4f}")
        for alvo, desc in alvos:
            linhas = []
            for sub in g.select(["dia", tvar, alvo] + xs).drop_nulls().partition_by(["dia", tvar]):
                X = np.column_stack([sub[c].to_numpy() / (dp[c] or 1.0) for c in xs])
                b, r2 = ols_np(sub[alvo].to_numpy(), X)
                linhas.append(dict(dia=sub["dia"][0], tercil=int(sub[tvar][0]), r2=r2,
                                   **{c: b[i + 1] for i, c in enumerate(xs) if c != "ret_ticks"}))
            df = pd.DataFrame(linhas)
            saida = []
            for est in ("r2", "qi", "canc_tot", "adic", "exec_liq"):
                pv = df.pivot_table(index="dia", columns="tercil", values=est)
                if not {1, 3} <= set(pv.columns):
                    continue
                pv["dif"] = pv[3] - pv[1]
                fm = resumo_fm(pv, ["dif"]).iloc[0]
                saida.append(dict(estatistica=est, tercil_1_baixo=pv[1].mean(), tercil_2=pv.get(2, pd.Series(dtype=float)).mean(),
                                  tercil_3_alto=pv[3].mean(), dif_3_menos_1=fm["media"], t=fm["t"], p=fm["p"], n_dias=fm["n"]))
            rel.log(f"Alvo: {desc} — coeficientes por 1 desvio-padrão; dif = tercil alto - tercil baixo (Fama-MacBeth):")
            rel.tabela(f"rlpf_tercis_{tvar}_{alvo}", pd.DataFrame(saida).set_index("estatistica"), fmt="{:.4g}")

    # ----- B) regressão com interações e controles -----
    rel.log("\n--- B) Interações: informação das filas x participação do RLP (intervalo anterior), com controles ---")
    rel.log("  Participação do RLP, volume e volatilidade do intervalo padronizados dentro do dia.")
    rel.log("  Controles: as mesmas interações com volume e volatilidade, para não confundir RLP com atividade.")
    for alvo, desc in alvos:
        linhas = []
        for sub in g.select(["dia", alvo] + xs + ["rlp_share_lag", "log_vol_iv", "log_rv_iv"]).drop_nulls().partition_by("dia"):
            z = {c: sub[c].to_numpy() / (dp[c] or 1.0) for c in xs}
            for c, k in (("rlp_share_lag", "s"), ("log_vol_iv", "v"), ("log_rv_iv", "r")):
                x = sub[c].to_numpy()
                z[k] = (x - x.mean()) / (x.std() or 1.0)
            nomes = xs + ["s", "v", "r"]
            cols = [z[c] for c in nomes]
            for a in ("qi", "canc_tot", "adic"):
                for k in ("s", "v", "r"):
                    nomes.append(f"{a}x{k}")
                    cols.append(z[a] * z[k])
            b, _ = ols_np(sub[alvo].to_numpy(), np.column_stack(cols))
            linhas.append(dict(dia=sub["dia"][0], **{nm: b[i + 1] for i, nm in enumerate(nomes)}))
        d = pd.DataFrame(linhas).set_index("dia")
        rel.log(f"Alvo: {desc}")
        rel.tabela(f"rlpf_interacoes_{alvo}",
                   resumo_fm(d, ["qi", "canc_tot", "adic", "qixs", "canc_totxs", "adicxs", "qixv", "qixr"])
                   [["media", "ic95_inf", "ic95_sup", "t", "p", "n"]])
    rel.log("\n  Como ler:")
    rel.log("  - 'qixs' > 0 e significativo: com mais RLP, o desequilíbrio de filas prevê melhor (hipótese confirmada).")
    rel.log("  - 'canc_totxs' e 'adicxs': o mesmo para cancelamentos e ordens novas.")
    rel.log("  - Em A, o tercil alto deve ter coeficientes e R² maiores que o baixo; confira se volume e volatilidade")
    rel.log("    estão parecidos entre tercis (tabela de equilíbrio).")
    rel.log("  - Sinais opostos ou nulos: o desvio do varejo não altera a informação das filas visíveis.")


# =============================================================================
# Etapa ROMPIMENTO: como a fila se rompe (agressão x cancelamento) e o que acontece depois
# =============================================================================
def rompimento_filas(cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[ROMPIMENTO] COMO A FILA SE ROMPE: AGRESSÃO OU CANCELAMENTO?\n" + "=" * 78)
    rel.log("Rompimento = o mid muda porque a fila do lado para onde o preço andou acabou")
    rel.log("  (alta: o melhor preço de venda subiu; baixa: o melhor preço de compra caiu).")
    rel.log(f"Causa = o que tirou contratos dessa fila desde a mudança de preço anterior (no máximo "
            f"{cfg.rompimento_janela_s:g}s antes):")
    lim = cfg.rompimento_limiar
    rel.log(f"  agressão: execuções >= {lim:.0%} do retirado | cancelamento: cancelamentos >= {lim:.0%} | "
            f"misto: o resto | sem_registro: nada retirado registrado.")
    t0 = time.time()
    seg = ["dia", "seg_n"]
    ev = ["bid_execucao_qtd", "ask_execucao_qtd", "bid_cancel_total_qtd", "bid_cancel_parcial_qtd",
          "ask_cancel_total_qtd", "ask_cancel_parcial_qtd"]
    lf = (
        _lazy_continuo(cfg, ["mid_simples", "bid_qtd", "ask_qtd"] + ev)
        .with_columns([
            _f0("ask_execucao_qtd").cum_sum().over(seg).alias("cx_a"),
            _f0("bid_execucao_qtd").cum_sum().over(seg).alias("cx_b"),
            (_f0("ask_cancel_total_qtd") + _f0("ask_cancel_parcial_qtd")).cum_sum().over(seg).alias("cc_a"),
            (_f0("bid_cancel_total_qtd") + _f0("bid_cancel_parcial_qtd")).cum_sum().over(seg).alias("cc_b"),
            pl.col("mid_simples").shift(1).over(seg).alias("mid_pre"),
            pl.col("bid_preco").shift(1).over(seg).alias("bid_pre"),
            pl.col("ask_preco").shift(1).over(seg).alias("ask_pre"),
        ])
    )
    cums = ["cx_a", "cx_b", "cc_a", "cc_b"]
    serie = lf.select(seg + ["timestamp", "mid_simples", "bid_qtd", "ask_qtd"] + cums).collect().sort("timestamp")
    limites = serie.group_by(seg).agg(pl.col("timestamp").min().alias("seg_ini"), pl.col("timestamp").max().alias("seg_fim"))
    mud = (
        lf.filter(pl.col("mid_pre").is_not_null() & (pl.col("mid_simples") != pl.col("mid_pre")))
        .select(seg + ["timestamp", "mid_simples", "mid_pre", "bid_preco", "ask_preco", "bid_pre", "ask_pre"] + cums)
        .collect()
        .sort("timestamp")
        .join(limites, on=seg, how="left")
        .with_columns((pl.col("mid_simples") - pl.col("mid_pre")).sign().alias("D"))
        .with_columns([
            # próxima mudança de preço (qualquer tipo) e cumulativos na mudança anterior
            pl.col("D").shift(-1).over(seg).alias("D_prox"),
            ((pl.col("timestamp").shift(-1).over(seg) - pl.col("timestamp")).dt.total_milliseconds() / 1000).alias("seg_ate_prox"),
        ] + [pl.col(c).shift(1).over(seg).alias(f"{c}_ant") for c in cums])
    )
    rel.log(f"Mudanças de preço: {mud.height:,} | carregadas em {time.time() - t0:.1f}s")

    # cumulativos e filas em t - janela
    W = int(round(cfg.rompimento_janela_s * 1000))
    right = serie.select(seg + ["timestamp", "bid_qtd", "ask_qtd"] + cums).rename(
        {"timestamp": "t_w", "bid_qtd": "bid_qtd_w", "ask_qtd": "ask_qtd_w", **{c: f"{c}_w" for c in cums}})
    mud = (mud.with_columns((pl.col("timestamp") - pl.duration(milliseconds=W)).alias("t_w"))
           .join_asof(right, on="t_w", by=seg, strategy="backward").drop("t_w"))
    # mid em t + h
    for h in cfg.horizontes_s:
        r = serie.select(seg + ["timestamp", "mid_simples"]).rename({"timestamp": "t_alvo", "mid_simples": f"mid_{h}s"})
        mud = (mud.with_columns((pl.col("timestamp") + pl.duration(milliseconds=int(h * 1000))).alias("t_alvo"))
               .join_asof(r, on="t_alvo", by=seg, strategy="backward")
               .with_columns(pl.when(pl.col("t_alvo") <= pl.col("seg_fim")).then(pl.col(f"mid_{h}s")).alias(f"mid_{h}s"))
               .drop("t_alvo"))
    del serie, right

    up = pl.col("D") > 0

    def lado(a: str, b: str) -> pl.Expr:
        return pl.when(up).then(pl.col(a)).otherwise(pl.col(b))

    def base(c: str) -> pl.Expr:   # início da janela: o mais recente entre mudança anterior e t - janela
        return pl.max_horizontal(pl.col(f"{c}_ant").fill_null(0), pl.col(f"{c}_w").fill_null(0))

    mud = mud.with_columns([
        pl.when((up & (pl.col("ask_preco") > pl.col("ask_pre"))) | (~up & (pl.col("bid_preco") < pl.col("bid_pre"))))
          .then(pl.lit("rompimento"))
          .when((up & (pl.col("bid_preco") > pl.col("bid_pre"))) | (~up & (pl.col("ask_preco") < pl.col("ask_pre"))))
          .then(pl.lit("melhora_no_spread")).otherwise(pl.lit("outro")).alias("tipo"),
        (lado("cx_a", "cx_b") - pl.when(up).then(base("cx_a")).otherwise(base("cx_b"))).alias("exec_ret"),
        (lado("cc_a", "cc_b") - pl.when(up).then(base("cc_a")).otherwise(base("cc_b"))).alias("canc_ret"),
        lado("ask_qtd_w", "bid_qtd_w").alias("fila_antes"),
        (pl.col("D") * (pl.col("mid_simples") - pl.col("mid_pre")) / cfg.tick).alias("salto"),
        (pl.col("D_prox") == -pl.col("D")).cast(pl.Float64).alias("volta"),
        pl.col("timestamp").dt.hour().alias("hora"),
    ]).with_columns([
        (pl.col("exec_ret") / (pl.col("exec_ret") + pl.col("canc_ret"))).alias("frac_exec"),
    ]).with_columns(
        pl.when((pl.col("exec_ret") + pl.col("canc_ret")) <= 0).then(pl.lit("sem_registro"))
        .when(pl.col("frac_exec") >= lim).then(pl.lit("agressao"))
        .when(pl.col("frac_exec") <= 1 - lim).then(pl.lit("cancelamento"))
        .otherwise(pl.lit("misto")).alias("causa")
    ).with_columns(
        [(pl.col("D") * (pl.col(f"mid_{h}s") - pl.col("mid_pre")) / cfg.tick).alias(f"perm_{h}s") for h in cfg.horizontes_s]
    ).with_columns(
        [(pl.col(f"perm_{h}s") - pl.col("salto")).alias(f"rev_{h}s") for h in cfg.horizontes_s]
    ).filter(((pl.col("timestamp") - pl.col("seg_ini")).dt.total_seconds() >= cfg.margem_inicio_s)
             & ((pl.col("seg_fim") - pl.col("timestamp")).dt.total_seconds() >= cfg.margem_fim_s))

    # ----- 1) descrição -----
    tipos = (mud.group_by("tipo").agg(pl.len().alias("mudancas")).with_columns(
        (pl.col("mudancas") / pl.col("mudancas").sum()).alias("fracao")).sort("mudancas", descending=True))
    rel.log("\n--- 1) Tipos de mudança de preço ---")
    rel.tabela("romp_tipos", tipos.to_pandas().set_index("tipo"), fmt="{:.4f}")

    rp = mud.filter(pl.col("tipo") == "rompimento").with_columns(pl.lit(1.0).alias("v_cls"))
    causas = ("agressao", "cancelamento", "misto", "sem_registro")
    desc = (rp.group_by("causa").agg(
                pl.len().alias("rompimentos"),
                pl.col("fila_antes").median().alias("fila_antes_mediana"),
                pl.col("exec_ret").median().alias("execucao_mediana"),
                pl.col("canc_ret").median().alias("cancelamento_mediano"))
            .with_columns((pl.col("rompimentos") / pl.col("rompimentos").sum()).alias("fracao"))
            .sort("rompimentos", descending=True))
    rel.log("\n--- 2) Causa dos rompimentos (em contratos retirados da fila) ---")
    rel.tabela("romp_causas", desc.to_pandas().set_index("causa"), fmt="{:.4g}")
    pdia = (rp.group_by(["dia", "causa"]).agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum().over("dia")).alias("f"))
            .to_pandas().pivot_table(index="dia", columns="causa", values="f"))
    rel.log("Fração por causa, variação entre dias (mín. / máx.): "
            + ", ".join(f"{c}={pdia[c].min():.1%}/{pdia[c].max():.1%}" for c in causas if c in pdia))
    perfil = (rp.group_by(["hora", "causa"]).agg(pl.len().alias("n"))
              .with_columns((pl.col("n") / pl.col("n").sum().over("hora")).alias("f"))
              .to_pandas().pivot_table(index="hora", columns="causa", values="f"))
    rel.log("Fração por causa ao longo do dia:")
    rel.tabela("romp_causas_hora", perfil, fmt="{:.3f}")

    # ----- 3) o que acontece depois, por causa -----
    rel.log("\n--- 3) Depois do rompimento, por causa (média por evento, Fama-MacBeth entre dias) ---")
    rel.log("  salto = tamanho do salto (ticks) | perm_h = variação até t+h a partir do preço antes do rompimento")
    rel.log("  rev_h = perm_h - salto (negativo = o preço voltou) | volta = prob. de a próxima mudança ser no sentido oposto")
    medidas = [("salto", "salto"), ("volta", "volta")] + [(f"perm_{h}s", f"perm_{h}s") for h in cfg.horizontes_s] \
        + [(f"rev_{h}s", f"rev_{h}s") for h in cfg.horizontes_s]
    difs = (("cancelamento", "agressao"), ("misto", "agressao"))
    rel.tabela("romp_depois", _fm_grupos(rp, medidas, "causa", causas, difs=difs))

    # ----- 4) controlando o tamanho da fila -----
    q1, q2 = rp["fila_antes"].quantile(1 / 3), rp["fila_antes"].quantile(2 / 3)
    rp = rp.with_columns(
        pl.when(pl.col("fila_antes") <= q1).then(pl.lit(f"1_curta (<= {q1:.0f})"))
        .when(pl.col("fila_antes") <= q2).then(pl.lit(f"2_media (<= {q2:.0f})"))
        .otherwise(pl.lit("3_longa")).alias("fila"))
    rel.log(f"\n--- 4) Por tamanho da fila {cfg.rompimento_janela_s:g}s antes do rompimento (tercis) ---")
    dist = (rp.group_by(["fila", "causa"]).agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum().over("fila")).alias("f"))
            .to_pandas().pivot_table(index="fila", columns="causa", values="f"))
    rel.log("Fração de cada causa por tamanho de fila:")
    rel.tabela("romp_causas_fila", dist, fmt="{:.3f}")
    hsel = [x for x in medidas if x[0] in ("volta", "perm_60s", f"perm_{max(cfg.horizontes_s)}s",
                                           "rev_60s", f"rev_{max(cfg.horizontes_s)}s")]
    rel.log("Depois do rompimento, por causa, dentro de cada tamanho de fila:")
    rel.tabela("romp_depois_fila", _fm_grupos(rp, hsel, "causa", causas, difs=difs, extra="fila"))

    rel.log("\n  Como ler:")
    rel.log("  - Fração de 'cancelamento' em (2): quanto da descoberta de preço acontece por retirada de liquidez.")
    rel.log("  - Em (3), 'cancelamento-agressao' em perm_300s e rev_300s perto de zero: rompimentos por cancelamento")
    rel.log("    são tão permanentes quanto por agressão -> a retirada de liquidez é um canal de informação completo.")
    rel.log("  - rev negativo e 'volta' alto para cancelamento: são rompimentos que se desfazem (ruído, gestão de risco).")
    rel.log("  - (4) confere se a diferença sobrevive comparando filas de mesmo tamanho.")

    # ----- 5) fração cancelada como variável contínua -----
    rel.log("\n--- 5) Fração cancelada como variável contínua (usa todos os rompimentos, sem categorias) ---")
    rel.log("  Por dia: medida = a + b*frac_cancelada + c*log(1+fila antes) + d*log(1+contratos retirados) + horário.")
    rel.log("  b = diferença entre um rompimento 100% por cancelamento e um 100% por agressão, mesma fila e horário.")
    alvos5 = ["volta"] + [f"rev_{h}s" for h in cfg.horizontes_s] + [f"perm_{max(cfg.horizontes_s)}s"]
    rp5 = rp.with_columns([
        (1 - pl.col("frac_exec")).alias("frac_canc"),
        (pl.col("fila_antes").fill_null(0) + 1).log().alias("log_fila"),
        (pl.col("exec_ret") + pl.col("canc_ret") + 1).log().alias("log_retirado"),
    ]).filter(pl.col("frac_canc").is_not_null() & pl.col("frac_canc").is_finite()
              & pl.col("log_fila").is_finite() & pl.col("log_retirado").is_finite())
    horas = sorted(rp5["hora"].unique().to_list())

    def b_por_dia(dados: pl.DataFrame, alvo: str) -> pd.Series:
        coefs = {}
        for sub in dados.select(["dia", "hora", alvo, "frac_canc", "log_fila", "log_retirado"]).drop_nulls().partition_by("dia"):
            hh = sub["hora"].to_numpy()
            dummies = [(hh == h).astype(float) for h in horas[1:]]
            X = np.column_stack([sub["frac_canc"].to_numpy(), sub["log_fila"].to_numpy(),
                                 sub["log_retirado"].to_numpy()] + dummies)
            b, _ = ols_np(sub[alvo].to_numpy(), X)
            coefs[sub["dia"][0]] = b[1]
        return pd.Series(coefs, name=alvo)

    sem_ultima_hora = rp5.filter((pl.col("seg_fim") - pl.col("timestamp")).dt.total_seconds() >= 3600)
    linhas, por_dia, por_dia_rob = [], {}, {}
    for alvo in alvos5:
        por_dia[alvo] = b_por_dia(rp5, alvo)
        fm = resumo_fm(pd.DataFrame({"b": por_dia[alvo]}), ["b"]).iloc[0]
        por_dia_rob[alvo] = b_por_dia(sem_ultima_hora, alvo)
        fr = resumo_fm(pd.DataFrame({"b": por_dia_rob[alvo]}), ["b"]).iloc[0]
        linhas.append(dict(medida=alvo, b_frac_cancelada=fm["media"], ic95_inf=fm["ic95_inf"],
                           ic95_sup=fm["ic95_sup"], t=fm["t"], p=fm["p"], n_dias=fm["n"],
                           frac_dias_positivo=fm["frac_pos"],
                           b_sem_ultima_hora=fr["media"], t_sem_ultima_hora=fr["t"]))
    rel.tabela("romp_continuo", pd.DataFrame(linhas).set_index("medida"))
    rel.tabelas["romp_continuo_por_dia"] = pd.DataFrame(por_dia)
    rel.log("  Robustez: 'b_sem_ultima_hora' repete a estimação excluindo a última hora de cada segmento.")
    dist = rp5["frac_canc"].to_numpy()
    rel.log("Distribuição da fração cancelada (quantis 10/25/50/75/90%): "
            + " / ".join(f"{np.quantile(dist, q):.2f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))


# =============================================================================
# Etapa QID_DR: QIDres ortogonalizado contra o spread implícito (Dayri & Rosenbaum, 2015)
# =============================================================================
def qid_spread_implicito(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[QID_DR] QIDres COM SPREAD IMPLÍCITO (DAYRI & ROSENBAUM, 2015)\n" + "=" * 78)
    rel.log("Spread implícito = 2*eta*tick. eta = N_continuações / (2 * N_alternâncias), contadas nos saltos de 1 tick")
    rel.log("do preço do último negócio (modelo de zonas de incerteza). eta < 1/2: o tick é grande demais.")
    rel.log("Preço do negócio = melhor oferta agredida antes da linha (ask para compra agressora, bid para venda).")
    t0 = time.time()
    seg = ["dia", "seg_n"]
    every = cfg.qid_intervalo

    # ----- 1) preço do negócio e saltos de 1 tick -----
    lf = _lazy_continuo(cfg, ["mid_simples", "negocio_qtd_agr_compra", "negocio_qtd_agr_venda"])
    tr = (
        lf.with_columns([pl.col("bid_preco").shift(1).over(seg).alias("bid_pre"),
                         pl.col("ask_preco").shift(1).over(seg).alias("ask_pre"),
                         _f0("negocio_qtd_agr_compra").alias("agr_c"),
                         _f0("negocio_qtd_agr_venda").alias("agr_v")])
        .filter((pl.col("agr_c") > 0) != (pl.col("agr_v") > 0))
        .with_columns(pl.when(pl.col("agr_c") > 0).then(pl.col("ask_pre")).otherwise(pl.col("bid_pre")).alias("p_neg"))
        .filter(pl.col("p_neg").is_not_null())
        .select(["segmento", "dia", "timestamp", "p_neg"])
        .collect()
        .sort("timestamp")
        .with_columns([
            ((pl.col("p_neg") - pl.col("p_neg").shift(1)).over("segmento") / cfg.tick).alias("salto"),
            pl.col("timestamp").dt.truncate(every).alias("intervalo"),
        ])
    )
    n_neg = tr.group_by(["segmento", "intervalo"]).agg(pl.len().alias("n_negocios"))
    sal = (tr.filter(pl.col("salto").is_not_null() & (pl.col("salto") != 0))
           .with_columns([pl.col("salto").sign().alias("sg"), pl.col("salto").abs().alias("abs_s")])
           .with_columns([pl.col("sg").shift(1).over("segmento").alias("sg_ant"),
                          pl.col("abs_s").shift(1).over("segmento").alias("abs_ant")]))
    frac1 = (sal["abs_s"].round(6) == 1).mean()
    um = (sal.filter((pl.col("abs_s").round(6) == 1) & (pl.col("abs_ant").round(6) == 1) & pl.col("sg_ant").is_not_null())
          .with_columns((pl.col("sg") == pl.col("sg_ant")).alias("cont")))
    rel.log(f"Linhas de negócio: {tr.height:,} | saltos do preço do negócio: {sal.height:,} | "
            f"fração de saltos de 1 tick: {frac1:.1%} | carregado em {time.time() - t0:.1f}s")

    def eta_de(df: pl.DataFrame, chaves: list[str]) -> pl.DataFrame:
        return (df.group_by(chaves)
                .agg(pl.col("cont").sum().alias("n_cont"), (~pl.col("cont")).sum().alias("n_alt"))
                .with_columns(pl.when(pl.col("n_alt") > 0)
                              .then(pl.col("n_cont") / (2 * pl.col("n_alt"))).alias("eta")))

    dia_eta = eta_de(um, ["dia"]).sort("dia").to_pandas().set_index("dia")
    rel.log("\neta por dia (Dayri & Rosenbaum: ativos de tick grande têm eta bem abaixo de 1/2):")
    rel.tabela("qiddr_eta_dia", dia_eta, fmt="{:.4f}")

    iv_eta = eta_de(um, ["segmento", "intervalo"]).join(n_neg, on=["segmento", "intervalo"], how="left")
    iv_eta = iv_eta.with_columns(
        pl.when((pl.col("n_cont") + pl.col("n_alt")) >= cfg.dr_min_saltos).then(pl.col("eta")).alias("eta"))

    # taxa de continuação do mid (versão correta da medida do Gemini): mudanças do mid linha a linha,
    # fração em que a mudança segue o sentido da anterior, por intervalo (sem olhar o futuro)
    mm = (lf.with_columns((pl.col("mid_simples") - pl.col("mid_simples").shift(1)).over(seg).alias("dmid_l"))
            .filter(pl.col("dmid_l").is_not_null() & (pl.col("dmid_l") != 0))
            .select(["segmento", "timestamp", "dmid_l"]).collect().sort("timestamp")
            .with_columns(pl.col("dmid_l").sign().alias("sgm"))
            .with_columns(pl.col("sgm").shift(1).over("segmento").alias("sgm_ant"))
            .filter(pl.col("sgm_ant").is_not_null())
            .with_columns(pl.col("timestamp").dt.truncate(every).alias("intervalo"))
            .group_by(["segmento", "intervalo"])
            .agg((pl.col("sgm") == pl.col("sgm_ant")).cast(pl.Float64).mean().alias("taxa_cont_mid"),
                 pl.len().alias("n_mud_mid")))

    # ----- 2) intervalos da grade (mesma construção da etapa qid) -----
    n = cfg.reversao_horizonte
    iv = agregar_intervalo(grade, every)
    aberto = (grade.with_columns(pl.col("janela").dt.truncate(every).alias("intervalo"))
              .group_by(["segmento", "intervalo"])
              .agg((pl.col("spread_ticks") > 1.0001).mean().alias("frac_spread_aberto")))
    tot = pl.col("impr") + pl.col("deter")
    iv = (iv.join(aberto, on=["segmento", "intervalo"], how="left")
            .join(mm, on=["segmento", "intervalo"], how="left")
            .join(iv_eta.select(["segmento", "intervalo", "eta", "n_cont", "n_alt", "n_negocios"]),
                  on=["segmento", "intervalo"], how="left")
            .with_columns([
                pl.when(tot >= cfg.qid_min_eventos).then((pl.col("impr") - pl.col("deter")) / tot).alias("qid_dif"),
                pl.when((tot >= cfg.qid_min_eventos) & (pl.col("deter") > 0)).then(pl.col("impr") / pl.col("deter")).alias("qid_razao"),
                pl.when(pl.col("rv2") > 0).then(pl.col("rv2").sqrt().log()).alias("log_rv"),
                (pl.col("n_linhas") + 1).log().alias("log_eventos"),
                (pl.col("v_total") + 1).log().alias("log_volume"),
                (pl.col("spread_ticks") * cfg.tick / pl.col("mid")).log().alias("log_spread_pct"),
                pl.when(pl.col("eta") > 0).then((2 * pl.col("eta") * cfg.tick / pl.col("mid")).log()).alias("log_spread_impl_pct"),
                pl.when((pl.col("n_negocios") > 0) & (pl.col("rv2") > 0))
                  .then((pl.col("rv2") / pl.col("n_negocios")).sqrt()).alias("vol_por_negocio"),
                pl.col("intervalo").dt.hour().alias("hora"),
                pl.when(pl.col("depth") > 0).then(pl.col("depth").log()).alias("log_depth"),
                pl.when((pl.col("depth") > 0) & (pl.col("v_total") > 0) & (pl.col("n_negocios") > 0))
                  .then((pl.col("depth") / (pl.col("v_total") / pl.col("n_negocios"))).log()).alias("log_fila_rel"),
            ])
            .with_columns(pl.col("qid_dif" if cfg.qid_definicao == "diferenca" else "qid_razao").alias("QID"))
            .with_columns([
                pl.col("log_rv").shift(-1).over("segmento").alias("y_log_rv"),
                pl.col("valido").shift(-1).over("segmento").alias("valido_prox"),
                pl.col("log_rv").rolling_mean(5).over("segmento").alias("har5"),
                pl.col("log_rv").rolling_mean(30).over("segmento").alias("har30"),
            ]))
    pdf = iv.to_pandas()
    pdf = pdf[pdf["valido"]].copy()
    rel.log(f"\nIntervalos válidos de {every}: {len(pdf):,} | com eta estimado (>= {cfg.dr_min_saltos} pares de saltos): "
            f"{pdf['eta'].notna().mean():.1%} | eta mediano por intervalo: {pdf['eta'].median():.3f}")

    # ----- 3) validação: spread implícito x volatilidade por negócio -----
    v = pdf.dropna(subset=["eta", "vol_por_negocio"])
    v = v[(v["eta"] > 0) & (v["vol_por_negocio"] > 0)]
    if len(v) > 30:
        x, y = np.log(2 * v["eta"].to_numpy()), np.log(v["vol_por_negocio"].to_numpy())
        m = sm.OLS(y, sm.add_constant(x)).fit()
        rel.log("\n--- Validação (relação spread x volatilidade por negócio de Dayri & Rosenbaum) ---")
        rel.log(f"  log(vol. por negócio, ticks) = a + b*log(spread implícito, ticks): b = {m.params[1]:.3f} "
                f"(ep {m.bse[1]:.3f}), R² = {m.rsquared:.3f}, corr = {np.corrcoef(x, y)[0, 1]:.3f}")
        rel.log("  Se o spread implícito capta o custo econômico do tick, b deve ser positivo e próximo de 1.")
        sp = pdf.dropna(subset=["log_spread_pct"])
        rel.log(f"  Comparação: desvio-padrão de log(spread observado) = {sp['log_spread_pct'].std():.4f}; "
                f"de log(spread implícito) = {pdf['log_spread_impl_pct'].std():.4f}")

    # ----- 4) três versões do QIDres -----
    variantes = {
        "QIDres_spread (original)": ["log_spread_pct"],
        "QIDres_DR (spread implícito)": ["log_spread_impl_pct"],
        "QIDres_DR+aberto": ["log_spread_impl_pct", "frac_spread_aberto"],
    }
    base_cols = ["QID", "log_spread_pct", "log_spread_impl_pct", "frac_spread_aberto"]
    q = pdf.dropna(subset=base_cols).copy()
    rel.log(f"\n--- Ortogonalização (mesma amostra para as três versões: {len(q):,} intervalos) ---")
    linhas = []
    nomes_res = {}
    for i, (nome, cols) in enumerate(variantes.items()):
        m = sm.OLS(q["QID"], sm.add_constant(q[cols])).fit()
        col = f"res_{i}"
        q[col] = -(q["QID"] - m.fittedvalues) / q["QID"].std()
        nomes_res[nome] = col
        rec = dict(versao=nome, r2=m.rsquared, corr_com_QID=q[col].corr(q["QID"]))
        for c in cols:
            rec[f"coef_{c}"] = m.params[c]
            rec[f"t_{c}"] = m.tvalues[c]
        linhas.append(rec)
    rel.tabela("qiddr_ortogonalizacao", pd.DataFrame(linhas).set_index("versao"), fmt="{:.4f}")
    rel.log("  Leitura: R² perto de zero e corr perto de -1 = ortogonalização vazia (QIDres ~ -QID).")
    rel.log("  Barardehi et al. preveem QID crescente na folga do tick: coef. positivo em log(spread implícito).")

    # ----- 5) H3: poder preditivo incremental sobre HAR -----
    d = q[q["valido_prox"].fillna(False).astype(bool)].dropna(
        subset=["y_log_rv", "log_rv", "har5", "har30", "log_eventos", "log_volume", "spread_ticks"]).copy()
    if len(d) < 100:
        rel.log("Observações insuficientes para H3.")
        return
    dum = pd.get_dummies(d["hora"], prefix="h", drop_first=True, dtype=float)
    d = pd.concat([d, dum], axis=1)
    base = ["log_rv", "har5", "har30", "log_eventos", "log_volume", "spread_ticks"] + list(dum.columns)
    modelos = {"M1 HAR+controles": base, "M2 +QID": base + ["QID"]}
    for nome, col in nomes_res.items():
        modelos[f"+{nome}"] = base + [col]
    # o spread implícito como preditor próprio, e o QIDres_DR acima dele
    simp = "log_spread_impl_pct"
    modelos["M1b +spread implícito"] = base + [simp]
    modelos["M1b +QIDres_DR"] = base + [simp, nomes_res["QIDres_DR (spread implícito)"]]
    referencia = {nome: ("M1b +spread implícito" if nome == "M1b +QIDres_DR" else "M1 HAR+controles")
                  for nome in modelos}
    rel.log(f"\n--- H3: log da volatilidade realizada do próximo intervalo ({len(d):,} intervalos) ---")
    linhas = []
    for nome, xs in modelos.items():
        r = sm.OLS(d["y_log_rv"], sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": cfg.hac_lags_qid})
        var = xs[-1] if nome != "M1 HAR+controles" else None
        fm_t = np.nan
        if var:
            fm = []
            for dia, g in d.groupby("dia"):
                if len(g) >= 30:
                    b, _ = ols_np(g["y_log_rv"].to_numpy(float), g[xs].to_numpy(float))
                    fm.append(dict(dia=dia, b=b[-1]))
            if fm:
                fm_t = resumo_fm(pd.DataFrame(fm).set_index("dia"), ["b"]).loc["b", "t"]
        linhas.append(dict(modelo=nome, r2_aj=r.rsquared_adj,
                           coef=r.params.get(var, np.nan) if var else np.nan,
                           t_hac=r.tvalues.get(var, np.nan) if var else np.nan,
                           t_fama_macbeth=fm_t))
    rel.tabela("qiddr_h3_insample", pd.DataFrame(linhas).set_index("modelo"))

    dias = sorted(d["dia"].unique())
    n_tr = _n_treino(cfg, len(dias))
    tr_, te = d[d["dia"].isin(dias[:n_tr])], d[d["dia"].isin(dias[n_tr:])]
    if len(te) > 50:
        sst = ((te["y_log_rv"] - tr_["y_log_rv"].mean()) ** 2).sum()
        erros, oos = {}, []
        for nome, xs in modelos.items():
            r = sm.OLS(tr_["y_log_rv"], sm.add_constant(tr_[xs])).fit()
            e = (te["y_log_rv"] - r.predict(sm.add_constant(te[xs], has_constant="add"))).to_numpy()
            erros[nome] = e
            rec = dict(modelo=nome, r2_oos=1 - (e ** 2).sum() / sst)
            if nome != "M1 HAR+controles":
                ref = referencia[nome]
                rec["comparado_com"] = ref
                rec["dm_t"], rec["dm_p_unilateral"] = diebold_mariano(erros[ref], e, cfg.hac_lags_qid)
            oos.append(rec)
        rel.log(f"\nFora da amostra (treino {n_tr} dias, teste {len(dias) - n_tr}); DM contra o modelo de referência:")
        rel.tabela("qiddr_h3_oos", pd.DataFrame(oos).set_index("modelo"))

    rel.log("\n  Como ler: a correção funciona se (i) a validação mostra b > 0, (ii) a ortogonalização deixa de ser vazia")
    rel.log("  (R² relevante, corr bem longe de -1) e (iii) QIDres_DR melhora a previsão fora da amostra.")
    rel.log("  Se (i) e (ii) passam mas (iii) não, a medida fica bem construída, mas sem poder preditivo no WIN.")
    rel.log("  'M1b +spread implícito' testa se o próprio spread implícito prevê volatilidade além do HAR;")
    rel.log("  'M1b +QIDres_DR' testa se o QIDres acrescenta algo DEPOIS de incluir o spread implícito.")

    # ----- 6) QID condicional a medidas de folga do tick -----
    rel.log("\n--- 6) O QID acrescenta algo condicional a medidas de folga do tick? (sem construir resíduos) ---")
    rel.log("  Num modelo que já contém a variável X, incluir QIDres(X) equivale a incluir o QID.")
    rel.log("  Por isso o teste é direto: HAR+X contra HAR, e HAR+X+QID contra HAR+X, fora da amostra.")
    rel.log("  Quando X inclui a fração com spread aberto, ela substitui o spread médio na base (são quase colineares).")
    folgas = {
        "fila no topo (log)": ["log_depth"],
        "fila / negócio médio (log)": ["log_fila_rel"],
        "fração com spread aberto": ["frac_spread_aberto"],
        "taxa de continuação do mid": ["taxa_cont_mid"],
        "spread implícito D&R (log)": ["log_spread_impl_pct"],
    }
    folgas["todas"] = [c for cols in folgas.values() for c in cols]
    todas_x = folgas["todas"]
    d6 = d.dropna(subset=todas_x).copy()
    d6 = d6[np.isfinite(d6[todas_x]).all(axis=1)]
    rel.log(f"  Amostra comum: {len(d6):,} intervalos.")
    base = [c for c in base if d6[c].std() > 0]   # descarta colunas constantes nesta amostra (ex.: hora sem dados)
    dias6 = sorted(d6["dia"].unique())
    n_tr6 = _n_treino(cfg, len(dias6))
    tr6, te6 = d6[d6["dia"].isin(dias6[:n_tr6])], d6[d6["dia"].isin(dias6[n_tr6:])]

    def oos_erro(xs):
        r = sm.OLS(tr6["y_log_rv"], sm.add_constant(tr6[xs])).fit()
        return (te6["y_log_rv"] - r.predict(sm.add_constant(te6[xs], has_constant="add"))).to_numpy()

    sst6 = ((te6["y_log_rv"] - tr6["y_log_rv"].mean()) ** 2).sum() if len(te6) else np.nan
    e_base = oos_erro(base) if len(te6) > 50 else None
    linhas = []
    for nome, xs in folgas.items():
        rec = dict(folga=nome)
        # com spread de 1 ou 2 ticks, o spread médio do intervalo é quase 1 + fração aberta: evita colinearidade
        bx = [c for c in base if not (c == "spread_ticks" and "frac_spread_aberto" in xs)]
        rec["r2_QID_explicado_por_X"] = sm.OLS(d6["QID"], sm.add_constant(d6[xs])).fit().rsquared
        r_x = sm.OLS(d6["y_log_rv"], sm.add_constant(d6[bx + xs])).fit(cov_type="HAC", cov_kwds={"maxlags": cfg.hac_lags_qid})
        r_xq = sm.OLS(d6["y_log_rv"], sm.add_constant(d6[bx + xs + ["QID"]])).fit(cov_type="HAC", cov_kwds={"maxlags": cfg.hac_lags_qid})
        rec["t_X"] = r_x.tvalues[xs[0]] if len(xs) == 1 else np.nan
        rec["t_QID_dado_X"] = r_xq.tvalues["QID"]
        fm = []
        for dia, g in d6.groupby("dia"):
            if len(g) >= 30:
                b, _ = ols_np(g["y_log_rv"].to_numpy(float), g[bx + xs + ["QID"]].to_numpy(float))
                fm.append(dict(dia=dia, b=b[-1]))
        rec["t_QID_dado_X_FM"] = resumo_fm(pd.DataFrame(fm).set_index("dia"), ["b"]).loc["b", "t"] if fm else np.nan
        if e_base is not None:
            e_x, e_xq = oos_erro(bx + xs), oos_erro(bx + xs + ["QID"])
            rec["r2oos_HAR"] = 1 - (e_base ** 2).sum() / sst6
            rec["r2oos_HAR+X"] = 1 - (e_x ** 2).sum() / sst6
            rec["r2oos_HAR+X+QID"] = 1 - (e_xq ** 2).sum() / sst6
            rec["dm_p_X_vs_HAR"] = diebold_mariano(e_base, e_x, cfg.hac_lags_qid)[1]
            rec["dm_p_QID_dado_X"] = diebold_mariano(e_x, e_xq, cfg.hac_lags_qid)[1]
        linhas.append(rec)
    rel.tabela("qiddr_qid_condicional", pd.DataFrame(linhas).set_index("folga"), fmt="{:.4f}")
    rel.log("  Como ler: 'dm_p_QID_dado_X' < 0,05 em alguma linha seria evidência de que o QID tem conteúdo")
    rel.log("  condicional àquela medida de folga. 'dm_p_X_vs_HAR' mostra se a própria medida de folga prevê volatilidade.")



# =============================================================================
# Etapa QID_CURTO: reconciliação com a especificação do Gemini (horizonte de 0,5 s)
# =============================================================================
def qid_curto_prazo(grade: pl.DataFrame, cfg: Config, rel: Relatorio) -> None:
    rel.log("\n" + "=" * 78 + "\n[QID_CURTO] RECONCILIAÇÃO: SPREAD IMPLÍCITO E QID NO HORIZONTE DE 0,5 s\n" + "=" * 78)
    rel.log("Alvo (igual ao do Gemini): desvio-padrão da variação do mid nas 5 janelas de 100 ms seguintes (ticks).")
    rel.log("Pergunta: o spread implícito e o QID preveem esse alvo quando a volatilidade passada é bem controlada,")
    rel.log("com inferência entre dias e avaliação fora da amostra?")
    t0 = time.time()
    seg2 = ["dia", "seg_n"]
    dt = cfg.dt_ms
    k5, k50, k600 = 5, int(round(5000 / dt)), int(round(60000 / dt))

    # ----- spread implícito em janela móvel de N saltos de 1 tick do preço do negócio -----
    lf = _lazy_continuo(cfg, ["negocio_qtd_agr_compra", "negocio_qtd_agr_venda"])
    tr = (
        lf.with_columns([pl.col("bid_preco").shift(1).over(seg2).alias("bid_pre"),
                         pl.col("ask_preco").shift(1).over(seg2).alias("ask_pre"),
                         _f0("negocio_qtd_agr_compra").alias("agr_c"),
                         _f0("negocio_qtd_agr_venda").alias("agr_v")])
        .filter((pl.col("agr_c") > 0) != (pl.col("agr_v") > 0))
        .with_columns(pl.when(pl.col("agr_c") > 0).then(pl.col("ask_pre")).otherwise(pl.col("bid_pre")).alias("p_neg"))
        .filter(pl.col("p_neg").is_not_null())
        .select(["segmento", "timestamp", "p_neg"]).collect().sort("timestamp")
        .with_columns(((pl.col("p_neg") - pl.col("p_neg").shift(1)).over("segmento") / cfg.tick).alias("salto"))
        .filter(pl.col("salto").is_not_null() & (pl.col("salto") != 0))
        .with_columns([pl.col("salto").sign().alias("sg"), pl.col("salto").abs().round(6).alias("abs_s")])
        .with_columns([pl.col("sg").shift(1).over("segmento").alias("sg_ant"),
                       pl.col("abs_s").shift(1).over("segmento").alias("abs_ant")])
        .filter((pl.col("abs_s") == 1) & (pl.col("abs_ant") == 1))
        .with_columns((pl.col("sg") == pl.col("sg_ant")).cast(pl.Float64).alias("cont"))
    )
    N = cfg.curto_janela_saltos
    tr = (tr.with_columns([pl.col("cont").rolling_sum(N).over("segmento").alias("n_c"),
                           (1 - pl.col("cont")).rolling_sum(N).over("segmento").alias("n_a")])
            .with_columns(((pl.col("n_c") + 1) / (2 * (pl.col("n_a") + 1))).log().alias("log_eta"))
            .filter(pl.col("log_eta").is_not_null())
            .select(["segmento", "timestamp", "log_eta"]))
    rel.log(f"Pares de saltos de 1 tick: {tr.height:,} | janela móvel de {N} pares | {time.time() - t0:.1f}s")

    # ----- variáveis na grade de 100 ms -----
    tot = pl.col("impr") + pl.col("deter")
    g = (grade.sort(["segmento", "janela"])
         .with_columns([pl.col("ret_ticks").fill_null(0).alias("r"),
                        (pl.col("janela") + pl.duration(milliseconds=dt)).alias("t_fim")])
         .with_columns([
             pl.col("r").rolling_std(k5).over("segmento").alias("vol_atual"),
             pl.col("r").rolling_std(k5).shift(-k5).over("segmento").alias("vol_futura"),
             pl.col("valido").shift(-k5).over("segmento").alias("valido_fut"),
             (pl.col("r").pow(2).rolling_mean(k5).over("segmento")).sqrt().alias("rv_05s"),
             (pl.col("r").pow(2).rolling_mean(k50).over("segmento")).sqrt().alias("rv_5s"),
             (pl.col("r").pow(2).rolling_mean(k600).over("segmento")).sqrt().alias("rv_60s"),
             (pl.col("n_linhas").cast(pl.Float64).rolling_sum(k5).over("segmento") + 1).log().alias("log_ativ"),
             (pl.col("v_total") + 1).log().alias("log_vol_qtd"),
             pl.when(tot > 0).then((pl.col("impr") - pl.col("deter")) / tot).otherwise(0.0).alias("QID_100ms"),
             pl.col("impr").rolling_sum(k50).over("segmento").alias("_i5"),
             pl.col("deter").rolling_sum(k50).over("segmento").alias("_d5"),
             pl.col("janela").dt.hour().alias("hora"),
         ])
         .with_columns(pl.when((pl.col("_i5") + pl.col("_d5")) > 0)
                       .then((pl.col("_i5") - pl.col("_d5")) / (pl.col("_i5") + pl.col("_d5"))).otherwise(0.0)
                       .alias("QID_5s"))
         .sort("t_fim")
         .join_asof(tr.rename({"timestamp": "t_fim"}), on="t_fim", by="segmento", strategy="backward")
         .filter(pl.col("valido") & pl.col("valido_fut").fill_null(False)))
    xs_all = ["vol_atual", "rv_05s", "rv_5s", "rv_60s", "log_ativ", "log_vol_qtd", "spread_ticks",
              "log_eta", "QID_100ms", "QID_5s"]
    g = g.select(["dia", "hora", "vol_futura"] + xs_all).drop_nulls()
    horas = sorted(g["hora"].unique().to_list())
    g = g.with_columns([(pl.col("hora") == h).cast(pl.Float64).alias(f"h{h}") for h in horas[1:]])
    hd = [f"h{h}" for h in horas[1:]]
    rel.log(f"Janelas utilizadas: {g.height:,}")

    har = ["rv_05s", "rv_5s", "rv_60s", "log_ativ", "spread_ticks"] + hd
    modelos = {
        "G réplica do Gemini": ["vol_atual", "log_vol_qtd", "log_eta", "QID_100ms"],
        "H HAR curto": har,
        "H + spread implícito": har + ["log_eta"],
        "H + spread implícito + QID": har + ["log_eta", "QID_100ms", "QID_5s"],
    }

    # ----- Fama-MacBeth por dia -----
    destaque = ["vol_atual", "rv_05s", "rv_5s", "rv_60s", "log_eta", "QID_100ms", "QID_5s"]
    linhas = []
    dias_frames = g.partition_by("dia")
    for nome, xs in modelos.items():
        coefs = []
        for sub in dias_frames:
            X = sub.select(xs).to_numpy()
            ok = X.std(0) > 0
            b, r2 = ols_np(sub["vol_futura"].to_numpy(), X[:, ok])
            cols_ok = [c for c, o in zip(xs, ok) if o]
            coefs.append(dict(dia=sub["dia"][0], r2=r2, **{c: b[i + 1] for i, c in enumerate(cols_ok)}))
        cd = pd.DataFrame(coefs).set_index("dia")
        rec = dict(modelo=nome, r2_mediano_dia=cd["r2"].median())
        for c in destaque:
            if c in cd:
                f = resumo_fm(cd, [c]).iloc[0]
                rec[f"{c}_coef"], rec[f"{c}_t"] = f["media"], f["t"]
        linhas.append(rec)
    rel.log("\n--- Dentro da amostra: coeficientes e t de Fama-MacBeth entre dias ---")
    rel.tabela("qidcurto_fm", pd.DataFrame(linhas).set_index("modelo").T, fmt="{:.4g}")
    rel.log("  Compare a réplica (uma escala de volatilidade) com o HAR curto (três escalas): se o coeficiente do")
    rel.log("  spread implícito encolhe muito no HAR, a maior parte do efeito na réplica era persistência de volatilidade.")

    # ----- fora da amostra -----
    dias = sorted(g["dia"].unique().to_list())
    n_tr = _n_treino(cfg, len(dias))
    trn = g.filter(pl.col("dia").is_in(dias[:n_tr]))
    tst = g.filter(pl.col("dia").is_in(dias[n_tr:]))
    if tst.height > 1000:
        y_tr, y_te = trn["vol_futura"].to_numpy(), tst["vol_futura"].to_numpy()
        sst = ((y_te - y_tr.mean()) ** 2).sum()
        erros = {}
        for nome, xs in modelos.items():
            Xtr, Xte = trn.select(xs).to_numpy(), tst.select(xs).to_numpy()
            b, _ = ols_np(y_tr, Xtr)
            erros[nome] = y_te - (b[0] + Xte @ b[1:])
        comparacoes = [("G réplica do Gemini", "H HAR curto"),
                       ("H HAR curto", "G réplica do Gemini"),
                       ("H + spread implícito", "H HAR curto"),
                       ("H + spread implícito + QID", "H + spread implícito")]
        linhas = [dict(modelo=n, r2_oos=1 - (e ** 2).sum() / sst) for n, e in erros.items()]
        lag = max(cfg.hac_lags_qid, 2 * k5)
        dm = []
        for novo, ref in comparacoes:
            t, p = diebold_mariano(erros[ref], erros[novo], lag)
            dm.append(dict(modelo=novo, contra=ref, dm_t=t, p_unilateral_novo_melhor=p))
        rel.log(f"\n--- Fora da amostra (treino {n_tr} dias, teste {len(dias) - n_tr}) ---")
        rel.tabela("qidcurto_oos", pd.DataFrame(linhas).set_index("modelo"), fmt="{:.5f}")
        rel.tabela("qidcurto_dm", pd.DataFrame(dm).set_index("modelo"), fmt="{:.4f}")

    # ----- validação deixando um dia de fora (cada dia é teste uma vez) -----
    rel.log("\n--- Validação deixando um dia de fora: ganho de erro quadrático dia a dia ---")
    rel.log("  Para cada dia: estima nos outros dias, prevê o dia deixado de fora. Ganho = queda relativa do")
    rel.log("  erro quadrático médio do modelo novo contra o de referência. Inferência entre dias (n = número de dias).")

    def mat(sub, xs):
        return np.column_stack([np.ones(sub.height), sub.select(xs).to_numpy()])

    acum = {}
    for nome, xs in modelos.items():
        acum[nome] = []
        for sub in dias_frames:
            X, y = mat(sub, xs), sub["vol_futura"].to_numpy()
            acum[nome].append((X.T @ X, X.T @ y))
    mse = {nome: [] for nome in modelos}
    rotulos = [sub["dia"][0] for sub in dias_frames]
    for i, sub in enumerate(dias_frames):
        y = sub["vol_futura"].to_numpy()
        for nome, xs in modelos.items():
            A = sum(a for j, (a, _) in enumerate(acum[nome]) if j != i)
            bvec = sum(b for j, (_, b) in enumerate(acum[nome]) if j != i)
            beta = np.linalg.lstsq(A, bvec, rcond=None)[0]
            e = y - mat(sub, xs) @ beta
            mse[nome].append(float(np.mean(e ** 2)))
    comps_loo = [("H HAR curto", "G réplica do Gemini"),
                 ("H + spread implícito", "H HAR curto"),
                 ("H + spread implícito + QID", "H + spread implícito")]
    linhas, por_dia = [], {}
    for novo, ref in comps_loo:
        ganho = 1 - np.array(mse[novo]) / np.array(mse[ref])
        por_dia[f"{novo} vs {ref}"] = ganho
        nd, npos = len(ganho), int((ganho > 0).sum())
        m, ep = ganho.mean(), ganho.std(ddof=1) / math.sqrt(nd)
        linhas.append(dict(comparacao=f"{novo} vs {ref}", dias_com_ganho=f"{npos}/{nd}",
                           ganho_medio_pct=100 * m, ganho_min_pct=100 * ganho.min(), ganho_max_pct=100 * ganho.max(),
                           t_entre_dias=m / ep if ep > 0 else np.nan,
                           p_sinal_unilateral=float(stats.binomtest(npos, nd, 0.5, alternative="greater").pvalue)))
    rel.tabela("qidcurto_loo", pd.DataFrame(linhas).set_index("comparacao"), fmt="{:.4g}")
    rel.tabelas["qidcurto_loo_por_dia"] = pd.DataFrame(por_dia, index=rotulos)
    rel.log("  O resultado do spread implícito fica de pé se ganhar na grande maioria dos dias (teste do sinal),")
    rel.log("  independentemente do t inflado do Diebold-Mariano agrupado.")

    rel.log("\n  Como ler:")
    rel.log("  - Se 'H + spread implícito' vence 'H HAR curto' fora da amostra, o spread implícito prevê o ruído")
    rel.log("    de microestrutura de 0,5 s além da volatilidade passada (resultado de dinâmica de preço, não do QID).")
    rel.log("  - Se não vence, o t alto do Gemini era persistência de volatilidade mal controlada.")
    rel.log("  - A última comparação diz se o QID acrescenta algo neste horizonte, depois do spread implícito.")
    rel.log(f"[QID_CURTO] concluída em {time.time() - t0:.1f}s")


# =============================================================================
# Modo confirmatório: pré-condições, decisões das hipóteses primárias e manifesto
# =============================================================================
ETAPAS_CONFIRMATORIAS = ("diag", "h1", "pred_qi", "filas", "rompimento", "linha", "rlp_filas", "qid", "qid_dr", "qid_curto")
MARGEM_EQUIVALENCIA = 0.15   # ticks (H3)
ALFA = 0.05


def _sha256(caminho: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gravar_manifesto(cfg: Config, pasta: Path, etapas: list[str]) -> None:
    import platform
    import scipy
    from datetime import datetime
    man = {
        "protocolo_versao": PROTOCOLO_VERSAO,
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "dados": str(Path(cfg.parquet_path).resolve()),
        "dados_sha256": _sha256(Path(cfg.parquet_path)),
        "etapas": etapas,
        "parametros": asdict(cfg),
        "margem_equivalencia_ticks": MARGEM_EQUIVALENCIA,
        "alfa_familia": ALFA,
        "versoes": {"python": platform.python_version(), "polars": pl.__version__, "numpy": np.__version__,
                    "pandas": pd.__version__, "statsmodels": sm.__version__ if hasattr(sm, "__version__") else "",
                    "scipy": scipy.__version__},
    }
    (pasta / "manifesto.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")


def _holm(pvals: list[float]) -> list[float]:
    k = len(pvals)
    ordem = sorted(range(k), key=lambda i: pvals[i])
    ajust, maximo = [np.nan] * k, 0.0
    for pos, i in enumerate(ordem):
        if not np.isfinite(pvals[i]):
            continue
        maximo = max(maximo, min(1.0, (k - pos) * pvals[i]))
        ajust[i] = maximo
    return ajust


def avaliar_confirmatorio(grade: pl.DataFrame, cfg: Config, rel: Relatorio, pasta: Path) -> None:
    rel.log("\n" + "=" * 78 + f"\n[CONFIRMATÓRIO] DECISÕES DO PROTOCOLO v{PROTOCOLO_VERSAO}\n" + "=" * 78)
    tab = rel.tabelas

    # ----- pré-condições -----
    pre = []
    sem = tab.get("linha_semantica_condicoes")
    ok_sem = sem is not None
    for cond in list(cfg.cols_lit) + list(cfg.cols_rl) + list(cfg.cols_lrl):
        lit = cond in cfg.cols_lit
        val = float(sem.loc[cond, "razao_exec_topo"]) if (sem is not None and cond in sem.index) else np.nan
        passa = bool(np.isfinite(val) and (val >= 0.90 if lit else val <= 0.05))
        ok_sem &= passa
        pre.append(dict(precondicao=f"semântica {cond}", valor=val, criterio=">= 0,90" if lit else "<= 0,05", passa=passa))
    v = grade.filter(pl.col("valido"))
    frac1 = float((v["spread_ticks"] <= 1.0001).mean())
    ok_tick = frac1 >= 0.90
    pre.append(dict(precondicao="spread em 1 tick (fração do tempo)", valor=frac1, criterio=">= 0,90", passa=ok_tick))
    n_dias = grade["dia"].n_unique()
    ok_dias = n_dias >= 12
    pre.append(dict(precondicao="pregões na amostra", valor=n_dias, criterio=">= 12", passa=ok_dias))
    pre_df = pd.DataFrame(pre).set_index("precondicao")
    rel.log("Pré-condições:")
    rel.tabela("precondicoes", pre_df, fmt="{:.4f}")
    pre_df.to_csv(pasta / "precondicoes.csv")

    # ----- estatísticas das hipóteses primárias -----
    def direcional(serie: pd.Series | None, sinal: int) -> dict:
        if serie is None:
            return dict(estimativa=np.nan, t=np.nan, gl=np.nan, p_bruto=np.nan, frac_dias_sinal=np.nan, n_dias=0)
        x = serie.dropna().to_numpy(float)
        n = len(x)
        if n < 2:
            return dict(estimativa=np.nan, t=np.nan, gl=np.nan, p_bruto=np.nan, frac_dias_sinal=np.nan, n_dias=n)
        m, ep = x.mean(), x.std(ddof=1) / math.sqrt(n)
        t = m / ep if ep > 0 else np.nan
        p = stats.t.sf(t, n - 1) if sinal > 0 else stats.t.cdf(t, n - 1)
        return dict(estimativa=m, t=t, gl=n - 1, p_bruto=float(p), frac_dias_sinal=float(np.mean(np.sign(x) == sinal)), n_dias=n)

    def equivalencia(linha_fm: pd.Series | None) -> dict:
        if linha_fm is None:
            return dict(estimativa=np.nan, p_bruto=np.nan, n_dias=0)
        m, n = float(linha_fm["media"]), int(linha_fm["n"])
        tc = stats.t.ppf(0.975, n - 1)
        ep = (float(linha_fm["ic95_sup"]) - m) / tc
        d = MARGEM_EQUIVALENCIA
        p_inf = stats.t.sf((m + d) / ep, n - 1)
        p_sup = stats.t.cdf((m - d) / ep, n - 1)
        t90 = stats.t.ppf(0.95, n - 1)
        return dict(estimativa=m, ic90_inf=m - t90 * ep, ic90_sup=m + t90 * ep, gl=n - 1,
                    p_bruto=float(max(p_inf, p_sup)), n_dias=n)

    rc = tab.get("romp_continuo_por_dia")
    fq = tab.get(f"filas_por_dia_dir_prox_componentes+qi")
    imp = tab.get("linha_impacto_fm")

    def linha_rl(h: str):
        if imp is None:
            return None
        sel = imp[(imp.index == "RL") & (imp["h"] == h)]
        return sel.iloc[0] if len(sel) else None

    testes = [
        ("H1a", "reversão do próximo movimento: b > 0", "dir", direcional(rc["volta"] if rc is not None and "volta" in rc else None, +1), +1),
        ("H1b", "rev_1s: b < 0", "dir", direcional(rc["rev_1s"] if rc is not None and "rev_1s" in rc else None, -1), -1),
        ("H2a", "canc_tot prevê o próximo movimento: coef. > 0", "dir", direcional(fq["canc_tot"] if fq is not None else None, +1), +1),
        ("H2b", "adic prevê o próximo movimento: coef. > 0", "dir", direcional(fq["adic"] if fq is not None else None, +1), +1),
        ("H3a", f"RL em 60 s dentro de ±{MARGEM_EQUIVALENCIA} tick", "eq", equivalencia(linha_rl("60s")), 0),
        ("H3b", f"RL em 300 s dentro de ±{MARGEM_EQUIVALENCIA} tick", "eq", equivalencia(linha_rl("300s")), 0),
    ]
    p_adj = _holm([t[3]["p_bruto"] for t in testes])
    linhas = []
    for (tid, desc, tipo, res, sinal), pa in zip(testes, p_adj):
        rec = dict(teste=tid, descricao=desc, **res, p_holm=pa)
        if not (ok_sem and ok_tick and ok_dias):
            decisao = "não avaliado: pré-condição falhou"
        elif not np.isfinite(pa):
            decisao = "não avaliado: estatística ausente"
        elif tipo == "dir":
            minimo = math.ceil(2 * res["n_dias"] / 3) / res["n_dias"]
            certo = np.sign(res["estimativa"]) == sinal
            decisao = "confirmado" if (certo and pa < ALFA and res["frac_dias_sinal"] >= minimo - 1e-12) else "não confirmado"
        else:
            decisao = "confirmado" if pa < ALFA else "não confirmado"
        rec["decisao"] = decisao
        linhas.append(rec)

    # ----- secundárias -----
    dm = tab.get("h3_dm")
    p_s1 = float(dm.loc["M2 HAR+controles+QID", "p_unilateral_novo_melhor"]) if dm is not None and "M2 HAR+controles+QID" in dm.index else np.nan
    linhas.append(dict(teste="S1", descricao="QID em 1 min: ganho fora da amostra (DM)", p_bruto=p_s1,
                       decisao=("sem ganho" if p_s1 >= ALFA else "ganho") if np.isfinite(p_s1) else "não avaliado"))
    orto = tab.get("qiddr_ortogonalizacao")
    if orto is not None:
        c_obs = float(orto.loc["QIDres_spread (original)", "corr_com_QID"])
        c_dr = float(orto.loc["QIDres_DR (spread implícito)", "corr_com_QID"])
        linhas.append(dict(teste="S2", descricao="ortogonalização do QID vazia (corr QIDres-QID < -0,9, observado e implícito)",
                           estimativa=max(c_obs, c_dr),
                           decisao="vazia" if (c_obs < -0.9 and c_dr < -0.9) else "não vazia"))
    s3 = []
    for h in ("60s", "300s"):
        if imp is not None:
            sel = imp[(imp.index == "L_RL") & (imp["h"] == h)]
            if len(sel):
                r = equivalencia(sel.iloc[0])
                linhas.append(dict(teste=f"S3 ({h})", descricao=f"L RL em {h} dentro de ±{MARGEM_EQUIVALENCIA} tick (sem correção)",
                                   **r, decisao="equivalente" if r["p_bruto"] < ALFA else "não equivalente"))
    dec = pd.DataFrame(linhas).set_index("teste")
    rel.log("\nDecisões (primárias com correção de Holm sobre H1a–H3b; secundárias sem correção):")
    rel.tabela("decisoes_confirmatorias", dec, fmt="{:.4g}")
    dec.to_csv(pasta / "decisoes_confirmatorias.csv")


# =============================================================================
# Execução
# =============================================================================
ETAPAS = ("diag", "h1", "pred", "pred_qi", "filas", "rlp_filas", "rompimento", "impacto", "linha", "var", "regimes", "qid", "qid_dr", "qid_curto")


def main() -> None:
    ap = argparse.ArgumentParser(description="Protocolo HFT WIN v2")
    ap.add_argument("--parquet", default=Config.parquet_path)
    ap.add_argument("--saida", default=Config.saida_dir)
    ap.add_argument("--dt-ms", type=int, default=Config.dt_ms)
    ap.add_argument("--qid-intervalo", default=Config.qid_intervalo)
    ap.add_argument("--qid-definicao", choices=["diferenca", "razao"], default=Config.qid_definicao)
    ap.add_argument("--margem-s", type=int, default=Config.margem_inicio_s)
    ap.add_argument("--etapas", default=",".join(ETAPAS), help="lista separada por vírgulas: " + ",".join(ETAPAS))
    ap.add_argument("--usar-cache", action="store_true", help="reaproveita a grade salva em <saida>/grade.parquet")
    ap.add_argument("--rompimento-limiar", type=float, default=Config.rompimento_limiar)
    ap.add_argument("--confirmatorio", action="store_true",
                    help=f"executa o protocolo congelado v{PROTOCOLO_VERSAO}: parâmetros fixos, etapas fixas, decisões e manifesto")
    a = ap.parse_args()

    if not Path(a.parquet).is_file():
        raise SystemExit(f"Arquivo de dados não encontrado: {Path(a.parquet).resolve()}\n"
                         "Rode a partir da pasta onde está o parquet ou passe o caminho completo em --parquet.")

    if a.confirmatorio:
        ignorados = [n for n, v, d in (("--dt-ms", a.dt_ms, Config.dt_ms), ("--qid-intervalo", a.qid_intervalo, Config.qid_intervalo),
                                       ("--qid-definicao", a.qid_definicao, Config.qid_definicao), ("--margem-s", a.margem_s, Config.margem_inicio_s),
                                       ("--rompimento-limiar", a.rompimento_limiar, Config.rompimento_limiar),
                                       ("--etapas", a.etapas, ",".join(ETAPAS)), ("--usar-cache", a.usar_cache, False)) if v != d]
        cfg = Config(parquet_path=a.parquet, saida_dir=a.saida, dias_treino=0)
        pasta = Path(cfg.saida_dir)
        pasta.mkdir(parents=True, exist_ok=True)
        rel = Relatorio(pasta)
        rel.log(f"[CONFIRMATÓRIO] Protocolo v{PROTOCOLO_VERSAO} | dados: {cfg.parquet_path}")
        if ignorados:
            rel.log(f"[CONFIRMATÓRIO] Opções ignoradas (o protocolo fixa os parâmetros): {', '.join(ignorados)}")
        etapas = list(ETAPAS_CONFIRMATORIAS)
        gravar_manifesto(cfg, pasta, etapas)
        grade = construir_grade(cfg, rel)
        for etapa in etapas:
            try:
                {"diag": lambda: diagnosticos(grade, cfg, rel), "h1": lambda: h1_contemporaneo(grade, cfg, rel),
                 "pred_qi": lambda: preditivo_qi(grade, cfg, rel), "filas": lambda: componentes_fila(grade, cfg, rel),
                 "rompimento": lambda: rompimento_filas(cfg, rel), "linha": lambda: impacto_por_linha(cfg, rel),
                 "rlp_filas": lambda: rlp_informatividade(grade, cfg, rel), "qid": lambda: qid_h3_eq7(grade, cfg, rel),
                 "qid_dr": lambda: qid_spread_implicito(grade, cfg, rel), "qid_curto": lambda: qid_curto_prazo(grade, cfg, rel)}[etapa]()
            except Exception:
                rel.log(f"\n!!! Erro na etapa '{etapa}':\n{traceback.format_exc()}")
        try:
            avaliar_confirmatorio(grade, cfg, rel, pasta)
        except Exception:
            rel.log(f"\n!!! Erro na avaliação confirmatória:\n{traceback.format_exc()}")
        rel.salvar(cfg, "relatorio_confirmatorio.txt")
        rel.log(f"\nResultados salvos em {pasta.resolve()}")
        return

    cfg = Config(parquet_path=a.parquet, saida_dir=a.saida, dt_ms=a.dt_ms, qid_intervalo=a.qid_intervalo,
                 qid_definicao=a.qid_definicao, margem_inicio_s=a.margem_s, margem_fim_s=a.margem_s,
                 rompimento_limiar=a.rompimento_limiar)
    pasta = Path(cfg.saida_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    rel = Relatorio(pasta)
    etapas = [e.strip() for e in a.etapas.split(",") if e.strip()]

    cache = pasta / "grade.parquet"
    grade = None
    if a.usar_cache and cache.exists():
        grade = pl.read_parquet(cache)
        if not {"qi", "adic"} <= set(grade.columns):
            rel.log("[grade] Cache de versão anterior do script (faltam colunas); reconstruindo a grade.")
            grade = None
        else:
            rel.log(f"[grade] Usando cache {cache}")
    if grade is None:
        grade = construir_grade(cfg, rel)
        grade.write_parquet(cache)

    var_dia, imp = None, None
    for etapa in etapas:
        try:
            if etapa == "diag":
                diagnosticos(grade, cfg, rel)
            elif etapa == "h1":
                h1_contemporaneo(grade, cfg, rel)
            elif etapa == "pred":
                preditivo_fm(grade, cfg, rel)
            elif etapa == "qid_curto":
                qid_curto_prazo(grade, cfg, rel)
            elif etapa == "qid_dr":
                qid_spread_implicito(grade, cfg, rel)
            elif etapa == "rompimento":
                rompimento_filas(cfg, rel)
            elif etapa == "rlp_filas":
                rlp_informatividade(grade, cfg, rel)
            elif etapa == "filas":
                componentes_fila(grade, cfg, rel)
            elif etapa == "pred_qi":
                preditivo_qi(grade, cfg, rel)
            elif etapa == "linha":
                impacto_por_linha(cfg, rel)
            elif etapa == "impacto":
                imp = impacto_por_tipo(grade, cfg, rel)
            elif etapa == "var":
                var_dia = hasbrouck_var(grade, cfg, rel)
            elif etapa == "regimes":
                regimes(grade, var_dia, imp, cfg, rel)
            elif etapa == "qid":
                qid_h3_eq7(grade, cfg, rel)
            else:
                rel.log(f"Etapa desconhecida: {etapa}")
        except Exception:
            rel.log(f"\n!!! Erro na etapa '{etapa}':\n{traceback.format_exc()}")

    nome_rel = "relatorio.txt" if set(etapas) == set(ETAPAS) else f"relatorio_{'_'.join(etapas)}.txt"
    rel.salvar(cfg, nome_rel)
    rel.log(f"\nResultados salvos em {pasta.resolve()}")


if __name__ == "__main__":
    main()