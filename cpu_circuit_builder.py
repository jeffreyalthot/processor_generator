from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def setup_logger(out_dir: str, level: int = logging.INFO) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    logger = logging.getLogger("cpu_eda_ai")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def now_ms() -> int:
    return int(time.time() * 1000)


def fmt_hz(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.3f} GHz"
    if x >= 1e6:
        return f"{x / 1e6:.3f} MHz"
    if x >= 1e3:
        return f"{x / 1e3:.3f} kHz"
    return f"{x:.3f} Hz"


def mask(width: int) -> int:
    return (1 << width) - 1


def u(width: int, value: int) -> int:
    return value & mask(width)


def bits_of(width: int, value: int) -> List[int]:
    value = u(width, value)
    return [(value >> i) & 1 for i in range(width)]


def from_bits(bits: List[int]) -> int:
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out


def popcount(x: int) -> int:
    return int(x).bit_count()


WireId = str


@dataclass
class Gate:
    name: str
    inputs: List[WireId]
    output: WireId
    delay_ps: float
    cap_f: float
    eval_fn: Callable[[List[int]], int]

    def eval(self, ins: List[int]) -> int:
        return self.eval_fn(ins) & 1


def G_AND(name: str, a: WireId, b: WireId, y: WireId, delay_ps=25.0, cap_f=2.0) -> Gate:
    return Gate(name, [a, b], y, delay_ps, cap_f, lambda ins: ins[0] & ins[1])


def G_OR(name: str, a: WireId, b: WireId, y: WireId, delay_ps=25.0, cap_f=2.0) -> Gate:
    return Gate(name, [a, b], y, delay_ps, cap_f, lambda ins: ins[0] | ins[1])


def G_XOR(name: str, a: WireId, b: WireId, y: WireId, delay_ps=35.0, cap_f=3.0) -> Gate:
    return Gate(name, [a, b], y, delay_ps, cap_f, lambda ins: ins[0] ^ ins[1])


@dataclass
class Netlist:
    name: str
    gates: List[Gate] = field(default_factory=list)
    wires: Set[WireId] = field(default_factory=set)
    inputs: List[WireId] = field(default_factory=list)
    outputs: List[WireId] = field(default_factory=list)

    def add_gate(self, g: Gate) -> None:
        self.gates.append(g)
        self.wires.add(g.output)
        for w in g.inputs:
            self.wires.add(w)

    def add_input(self, w: WireId) -> None:
        self.inputs.append(w)
        self.wires.add(w)

    def add_output(self, w: WireId) -> None:
        self.outputs.append(w)
        self.wires.add(w)

    def topo_sort(self) -> List[Gate]:
        producers: Dict[WireId, Gate] = {}
        for g in self.gates:
            producers[g.output] = g

        deps: Dict[str, Set[str]] = {}
        by_name: Dict[str, Gate] = {g.name: g for g in self.gates}
        for g in self.gates:
            s = set()
            for w in g.inputs:
                if w in producers:
                    s.add(producers[w].name)
            deps[g.name] = s

        ordered: List[Gate] = []
        ready = sorted([n for n, s in deps.items() if not s])

        while ready:
            n = ready.pop(0)
            ordered.append(by_name[n])
            for m in list(deps.keys()):
                if n in deps[m]:
                    deps[m].remove(n)
                    if not deps[m]:
                        ready.append(m)
                        ready.sort()
            deps.pop(n, None)

        if deps:
            raise ValueError(f"Cycle détecté dans netlist {self.name}: {list(deps.keys())[:10]} ...")
        return ordered

    def simulate(self, in_values: Dict[WireId, int]) -> Dict[WireId, int]:
        values: Dict[WireId, int] = {w: (in_values.get(w, 0) & 1) for w in self.wires}
        for g in self.topo_sort():
            ins = [values[i] for i in g.inputs]
            values[g.output] = g.eval(ins)
        return {w: values[w] for w in self.outputs}

    def critical_path_ps(self) -> float:
        arrival: Dict[WireId, float] = {w: 0.0 for w in self.wires}
        for g in self.topo_sort():
            t_in = max((arrival.get(i, 0.0) for i in g.inputs), default=0.0)
            arrival[g.output] = t_in + g.delay_ps
        return max((arrival.get(w, 0.0) for w in self.outputs), default=0.0)

    def total_cap_f(self) -> float:
        return sum(g.cap_f for g in self.gates)

    def gate_count(self) -> int:
        return len(self.gates)


def build_full_adder(prefix: str, a: WireId, b: WireId, cin: WireId, s: WireId, cout: WireId) -> Netlist:
    nl = Netlist(name=f"{prefix}_FA")
    for w in [a, b, cin]:
        nl.add_input(w)
    for w in [s, cout]:
        nl.add_output(w)

    x1, o1, a1, a2, a3 = [f"{prefix}_{s}" for s in ["x1", "o1", "a1", "a2", "a3"]]
    nl.add_gate(G_XOR(f"{prefix}_xor1", a, b, x1))
    nl.add_gate(G_XOR(f"{prefix}_xor2", x1, cin, s))
    nl.add_gate(G_AND(f"{prefix}_and1", a, b, a1))
    nl.add_gate(G_AND(f"{prefix}_and2", a, cin, a2))
    nl.add_gate(G_AND(f"{prefix}_and3", b, cin, a3))
    nl.add_gate(G_OR(f"{prefix}_or1", a1, a2, o1))
    nl.add_gate(G_OR(f"{prefix}_or2", o1, a3, cout))
    return nl


def build_ripple_adder(prefix: str, width: int, a_bits: List[WireId], b_bits: List[WireId], cin: WireId, s_bits: List[WireId], cout: WireId) -> Netlist:
    nl = Netlist(name=f"{prefix}_RCA{width}")
    for w in a_bits + b_bits + [cin]:
        nl.add_input(w)
    for w in s_bits + [cout]:
        nl.add_output(w)

    c = cin
    for i in range(width):
        si = s_bits[i]
        co = cout if i == width - 1 else f"{prefix}_c{i + 1}"
        fa = build_full_adder(f"{prefix}_fa{i}", a_bits[i], b_bits[i], c, si, co)
        for g in fa.gates:
            nl.add_gate(g)
        c = co
    return nl


def build_bitwise_logic(prefix: str, op: str, width: int, a_bits: List[WireId], b_bits: List[WireId], y_bits: List[WireId]) -> Netlist:
    nl = Netlist(name=f"{prefix}_{op}{width}")
    for w in a_bits + b_bits:
        nl.add_input(w)
    for w in y_bits:
        nl.add_output(w)

    for i in range(width):
        if op == "AND":
            nl.add_gate(G_AND(f"{prefix}_and{i}", a_bits[i], b_bits[i], y_bits[i]))
        elif op == "OR":
            nl.add_gate(G_OR(f"{prefix}_or{i}", a_bits[i], b_bits[i], y_bits[i]))
        else:
            nl.add_gate(G_XOR(f"{prefix}_xor{i}", a_bits[i], b_bits[i], y_bits[i]))
    return nl


@dataclass
class Register:
    name: str
    width: int
    value: int = 0

    def read(self) -> int:
        return u(self.width, self.value)

    def write(self, v: int) -> None:
        self.value = u(self.width, v)


@dataclass
class RegisterFile:
    name: str
    nregs: int
    width: int
    regs: List[Register] = field(init=False)

    def __post_init__(self) -> None:
        self.regs = [Register(f"{self.name}_r{i}", self.width, 0) for i in range(self.nregs)]

    def read(self, idx: int) -> int:
        return self.regs[idx % self.nregs].read()

    def write(self, idx: int, val: int) -> None:
        self.regs[idx % self.nregs].write(val)


@dataclass
class Memory:
    name: str
    width: int
    depth: int
    mem: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.mem = [0] * self.depth

    def load(self, addr: int) -> int:
        return u(self.width, self.mem[addr % self.depth])

    def store(self, addr: int, val: int) -> None:
        self.mem[addr % self.depth] = u(self.width, val)


OP_NOP, OP_ADD, OP_ADDI, OP_AND, OP_OR, OP_XOR, OP_LD, OP_ST, OP_BEQ, OP_J = range(10)


def signext4(x: int) -> int:
    x &= 0xF
    return x - 16 if (x & 0x8) else x


@dataclass
class CPUConfig:
    width: int = 16
    nregs: int = 16
    mem_depth: int = 256
    vdd: float = 1.0
    freq_hz: float = 50e6
    leakage_w: float = 0.02
    r_th: float = 8.0
    c_th: float = 10.0
    t_amb: float = 25.0
    use_fast_adder: bool = False
    pipeline_stages: int = 1
    branch_rate: float = 0.12
    mem_rate: float = 0.18


@dataclass
class CPUDynamicResult:
    cycles: int
    insn_retired: int
    freq_hz: float
    fmax_hz: float
    avg_power_w: float
    peak_power_w: float
    final_temp_c: float
    avg_temp_c: float
    crit_path_ps: float
    cap_total_f: float
    toggles: int


class ToyCPU:
    def __init__(self, cfg: CPUConfig):
        self.cfg = cfg
        self.rf = RegisterFile("rf", cfg.nregs, cfg.width)
        self.mem = Memory("mem", cfg.width, cfg.mem_depth)
        self.pc = Register("pc", cfg.width, 0)
        self._build_datapath()
        self.reset()

    def reset(self) -> None:
        self.pc.write(0)
        for i in range(self.cfg.nregs):
            self.rf.write(i, 0)
        self.cycle = 0
        self.insn_retired = 0
        self._last_alu_y = 0
        self._last_rf_write = 0
        self._toggle_count = 0

    def load_program(self, words: List[int], base_addr: int = 0) -> None:
        for i, w in enumerate(words):
            self.mem.store(base_addr + i, w)

    def _build_datapath(self) -> None:
        w = self.cfg.width
        self.A = [f"A{i}" for i in range(w)]
        self.B = [f"B{i}" for i in range(w)]
        self.S = [f"S{i}" for i in range(w)]
        self.CIN, self.COUT = "CIN", "COUT"

        self.nl_adder = build_ripple_adder("alu_add", w, self.A, self.B, self.CIN, self.S, self.COUT)
        if self.cfg.use_fast_adder:
            for g in self.nl_adder.gates:
                g.delay_ps *= 0.7
                g.cap_f *= 1.35

        self.Y_and = [f"Yand{i}" for i in range(w)]
        self.Y_or = [f"Yor{i}" for i in range(w)]
        self.Y_xor = [f"Yxor{i}" for i in range(w)]
        self.nl_and = build_bitwise_logic("alu", "AND", w, self.A, self.B, self.Y_and)
        self.nl_or = build_bitwise_logic("alu", "OR", w, self.A, self.B, self.Y_or)
        self.nl_xor = build_bitwise_logic("alu", "XOR", w, self.A, self.B, self.Y_xor)

        self.crit_path_ps = max(self.nl_adder.critical_path_ps(), self.nl_and.critical_path_ps(), self.nl_or.critical_path_ps(), self.nl_xor.critical_path_ps())
        self.cap_total_f = self.nl_adder.total_cap_f() + self.nl_and.total_cap_f() + self.nl_or.total_cap_f() + self.nl_xor.total_cap_f()
        self.gate_counts = {"adder": self.nl_adder.gate_count(), "and": self.nl_and.gate_count(), "or": self.nl_or.gate_count(), "xor": self.nl_xor.gate_count()}

    def fmax_hz(self) -> float:
        tcrit_s = self.crit_path_ps * 1e-12
        pipe = max(1, int(self.cfg.pipeline_stages))
        return 1.0 / ((tcrit_s / pipe) + 120e-12)

    def effective_freq_hz(self) -> float:
        return min(self.cfg.freq_hz, self.fmax_hz())

    def fetch(self) -> int:
        return self.mem.load(self.pc.read() % self.cfg.mem_depth)

    def _alu_add(self, a: int, b: int, cin: int = 0) -> int:
        w = self.cfg.width
        inmap = {self.A[i]: bits_of(w, a)[i] for i in range(w)}
        inmap.update({self.B[i]: bits_of(w, b)[i] for i in range(w)})
        inmap[self.CIN] = cin & 1
        outs = self.nl_adder.simulate(inmap)
        return from_bits([outs[self.S[i]] for i in range(w)])

    def _alu_logic(self, nl: Netlist, a: int, b: int, out_bus: List[WireId]) -> int:
        w = self.cfg.width
        inmap = {self.A[i]: bits_of(w, a)[i] for i in range(w)}
        inmap.update({self.B[i]: bits_of(w, b)[i] for i in range(w)})
        outs = nl.simulate(inmap)
        return from_bits([outs[out_bus[i]] for i in range(w)])

    def step(self) -> None:
        instr = self.fetch()
        self.pc.write(self.pc.read() + 1)
        opcode, rd, rs, imm4 = ((instr >> 12) & 0xF, (instr >> 8) & 0xF, (instr >> 4) & 0xF, instr & 0xF)
        rt = imm4
        a, b = self.rf.read(rs), self.rf.read(rt)
        alu_y = 0
        rf_write_val: Optional[int] = None
        rf_write_idx: Optional[int] = None

        if opcode == OP_ADD:
            alu_y = self._alu_add(a, b, 0)
            rf_write_idx, rf_write_val = rd, alu_y
        elif opcode == OP_ADDI:
            alu_y = self._alu_add(a, imm4, 0)
            rf_write_idx, rf_write_val = rd, alu_y
        elif opcode == OP_AND:
            alu_y = self._alu_logic(self.nl_and, a, b, self.Y_and)
            rf_write_idx, rf_write_val = rd, alu_y
        elif opcode == OP_OR:
            alu_y = self._alu_logic(self.nl_or, a, b, self.Y_or)
            rf_write_idx, rf_write_val = rd, alu_y
        elif opcode == OP_XOR:
            alu_y = self._alu_logic(self.nl_xor, a, b, self.Y_xor)
            rf_write_idx, rf_write_val = rd, alu_y
        elif opcode == OP_LD:
            rf_write_idx, rf_write_val = rd, self.mem.load((a + imm4) % self.cfg.mem_depth)
        elif opcode == OP_ST:
            self.mem.store((a + imm4) % self.cfg.mem_depth, self.rf.read(rd))
        elif opcode == OP_BEQ:
            if self.rf.read(rd) == self.rf.read(rs):
                self.pc.write(u(self.cfg.width, self.pc.read() + signext4(imm4)))
        elif opcode == OP_J:
            self.pc.write((self.pc.read() & 0xF000) | (instr & 0x0FFF))

        if rf_write_idx is not None and rf_write_val is not None:
            self.rf.write(rf_write_idx, rf_write_val)

        self._toggle_count += popcount(self._last_alu_y ^ alu_y)
        self._last_alu_y = alu_y
        if rf_write_val is not None:
            self._toggle_count += popcount(self._last_rf_write ^ rf_write_val)
            self._last_rf_write = rf_write_val

        self.cycle += 1
        self.insn_retired += 1

    def evaluate_dynamic(self, cycles: int = 2500, seed: int = 0) -> CPUDynamicResult:
        rng = random.Random(seed)
        prog_len = min(128, self.cfg.mem_depth // 2)
        prog: List[int] = []
        for _ in range(prog_len):
            op = rng.choices(population=[OP_NOP, OP_ADD, OP_ADDI, OP_AND, OP_OR, OP_XOR, OP_LD, OP_ST, OP_BEQ, OP_J], weights=[5, 10, 10, 7, 7, 7, 8, 8, 6, 2], k=1)[0]
            rd, rs, imm = rng.randrange(self.cfg.nregs), rng.randrange(self.cfg.nregs), rng.randrange(16)
            prog.append(((op & 0xF) << 12) | ((rd & 0xF) << 8) | ((rs & 0xF) << 4) | (imm & 0xF))

        self.reset()
        self.load_program(prog, 0)
        for i in range(self.cfg.nregs):
            self.rf.write(i, rng.randrange(1 << self.cfg.width))

        freq = self.effective_freq_hz()
        dt = 1.0 / max(1.0, freq)
        T = self.cfg.t_amb
        T_acc = p_acc = p_peak = 0.0
        toggles_prev = 0
        c_total = self.cap_total_f * 1e-15

        for _ in range(cycles):
            self.step()
            toggles_cycle = max(0, self._toggle_count - toggles_prev)
            toggles_prev = self._toggle_count
            alpha = min(1.0, toggles_cycle / max(1, 2 * self.cfg.width))
            p_total = (alpha * c_total * (self.cfg.vdd ** 2) * freq) + self.cfg.leakage_w
            T += (((self.cfg.t_amb - T) / (self.cfg.r_th * self.cfg.c_th)) + (p_total / self.cfg.c_th)) * dt
            p_acc += p_total
            p_peak = max(p_peak, p_total)
            T_acc += T

        stall_frac = min(0.75, self.cfg.branch_rate * (max(1, self.cfg.pipeline_stages) - 1) * 0.35 + self.cfg.mem_rate * 0.20)
        return CPUDynamicResult(
            cycles=cycles,
            insn_retired=int(self.insn_retired * (1.0 - stall_frac)),
            freq_hz=freq,
            fmax_hz=self.fmax_hz(),
            avg_power_w=p_acc / cycles,
            peak_power_w=p_peak,
            final_temp_c=T,
            avg_temp_c=T_acc / cycles,
            crit_path_ps=self.crit_path_ps,
            cap_total_f=self.cap_total_f,
            toggles=self._toggle_count,
        )


@dataclass
class Block:
    name: str
    layer: int
    w: float
    h: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class Floorplan:
    name: str
    layers: int
    blocks: List[Block]
    die_w: float
    die_h: float
    layer_heights: List[float]


def _approx_area_from_gates(gates: int, scale: float = 1.0) -> float:
    return max(10.0, gates * scale)


def place_blocks_multilayer(name: str, layers: int, blocks: List[Block]) -> Floorplan:
    pad, shelf_h_gap, shelf_w_limit = 10.0, 10.0, 380.0
    layer_heights: List[float] = [0.0] * layers
    layer_widths: List[float] = [0.0] * layers

    for L in range(layers):
        x = y = pad
        shelf_h = max_x = 0.0
        bl = sorted([b for b in blocks if b.layer == L], key=lambda b: b.w * b.h, reverse=True)
        for b in bl:
            if x + b.w + pad > shelf_w_limit and x > pad:
                x = pad
                y += shelf_h + shelf_h_gap
                shelf_h = 0.0
            b.x, b.y = x, y
            x += b.w + pad
            shelf_h = max(shelf_h, b.h)
            max_x = max(max_x, x)
        layer_widths[L] = max(max_x + pad, shelf_w_limit * 0.85)
        layer_heights[L] = y + shelf_h + pad

    die_w = max(layer_widths) if layer_widths else 500.0
    die_h = sum(layer_heights) + pad * (layers + 1)
    y_offset = pad
    for L in range(layers):
        for b in blocks:
            if b.layer == L:
                b.x += (die_w - layer_widths[L]) * 0.5
                b.y += y_offset
        y_offset += layer_heights[L] + pad

    return Floorplan(name, layers, blocks, die_w, die_h, layer_heights)


def build_floorplan_for_cpu(cpu: ToyCPU, name: str = "cpu") -> Floorplan:
    cfg = cpu.cfg

    def dims(area: float, aspect: float) -> Tuple[float, float]:
        w = math.sqrt(area * aspect)
        return w, area / max(1e-6, w)

    blocks: List[Block] = []
    for n, l, a, asp in [
        ("Adder", 0, _approx_area_from_gates(cpu.gate_counts["adder"], 1.1), 1.8 if cfg.use_fast_adder else 1.4),
        ("Logic(AND/OR/XOR)", 0, _approx_area_from_gates(cpu.gate_counts["and"] + cpu.gate_counts["or"] + cpu.gate_counts["xor"], 0.9), 1.6),
        ("RegFile", 1, max(120.0, cfg.nregs * cfg.width * 0.7), 1.2),
        ("PC", 1, max(40.0, cfg.width * 1.2), 1.0),
        ("Memory", 2, max(200.0, cfg.mem_depth * cfg.width * 0.03), 1.1),
        ("Control", 2, 140.0 + 20.0 * max(1, cfg.pipeline_stages), 1.3),
    ]:
        w, h = dims(a, asp)
        blocks.append(Block(n, l, w, h))
    return place_blocks_multilayer(name, 3, blocks)


def floorplan_to_svg(fp: Floorplan, path: str) -> None:
    layer_colors = ["#E8F4FF", "#E8FFE8", "#FFF0E8"]
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fp.die_w:.0f}" height="{fp.die_h:.0f}" viewBox="0 0 {fp.die_w:.0f} {fp.die_h:.0f}">',
        f'<rect x="0" y="0" width="{fp.die_w:.0f}" height="{fp.die_h:.0f}" fill="#FFFFFF" stroke="#333" stroke-width="2"/>',
        f'<text x="12" y="22" font-family="monospace" font-size="14">Floorplan: {fp.name}</text>',
    ]
    y_offset = 10.0
    for L in range(fp.layers):
        h = fp.layer_heights[L]
        col = layer_colors[L % len(layer_colors)]
        svg.append(f'<rect x="6" y="{y_offset:.1f}" width="{fp.die_w - 12:.1f}" height="{h:.1f}" fill="{col}" stroke="#333" stroke-width="1" opacity="0.55"/>')
        svg.append(f'<text x="12" y="{y_offset + 18:.1f}" font-family="monospace" font-size="12">Layer {L}</text>')
        y_offset += h + 10.0

    for b in fp.blocks:
        svg.append(f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" fill="{layer_colors[b.layer % len(layer_colors)]}" stroke="#333" stroke-width="1.5"/>')
        svg.append(f'<text x="{b.x + 6:.1f}" y="{b.y + 16:.1f}" font-family="monospace" font-size="12">{b.name}</text>')

    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def floorplan_to_png_optional(fp: Floorplan, path_png: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    fig = plt.figure(figsize=(fp.die_w / 120.0, fp.die_h / 120.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, fp.die_w)
    ax.set_ylim(fp.die_h, 0)
    ax.set_aspect("equal")
    for b in fp.blocks:
        ax.add_patch(plt.Rectangle((b.x, b.y), b.w, b.h, fill=False, linewidth=1.5))
        ax.text(b.x + 4, b.y + 14, b.name, fontsize=8, family="monospace")
    ax.axis("off")
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)
    return True


@dataclass
class PDK:
    name: str = "toy28"
    metal_layers: int = 6
    max_utilization: float = 0.75
    r_ohm_per_um: float = 0.08
    c_ff_per_um: float = 0.20


@dataclass
class EDAChecks:
    require_synthesis: bool = True
    require_floorplan: bool = True
    require_place: bool = True
    require_route: bool = True
    require_extraction: bool = True
    require_sta: bool = True
    require_drc: bool = True
    require_lvs: bool = True
    require_power_thermal_signoff: bool = True


@dataclass
class EDAArtifacts:
    rtl_json: Optional[str] = None
    gate_netlist_json: Optional[str] = None
    floorplan_svg: Optional[str] = None


@dataclass
class EDAResult:
    ok: bool
    reasons: List[str]
    artifacts: EDAArtifacts
    est_added_delay_ps: float = 0.0
    est_added_cap_f: float = 0.0
    util: float = 0.0
    wirelength_um: float = 0.0
    sta_fmax_hz: float = 0.0
    signoff_temp_ss_c: float = 0.0


class EDAFlow:
    def __init__(self, pdk: PDK, checks: EDAChecks, out_dir: str, logger: logging.Logger):
        self.pdk, self.checks, self.out_dir, self.logger = pdk, checks, out_dir, logger
        os.makedirs(out_dir, exist_ok=True)

    def run(self, cpu: ToyCPU, tag: str) -> EDAResult:
        reasons: List[str] = []
        art = EDAArtifacts()

        rtl_path = os.path.join(self.out_dir, f"{tag}_rtl.json")
        with open(rtl_path, "w", encoding="utf-8") as f:
            json.dump({"cpu_cfg": asdict(cpu.cfg), "isa": "toy16"}, f, indent=2)
        art.rtl_json = rtl_path

        nl_path = os.path.join(self.out_dir, f"{tag}_gate_netlist.json")
        gates = cpu.nl_adder.gates + cpu.nl_and.gates + cpu.nl_or.gates + cpu.nl_xor.gates
        with open(nl_path, "w", encoding="utf-8") as f:
            json.dump({"gate_count": len(gates)}, f, indent=2)
        art.gate_netlist_json = nl_path

        fp = build_floorplan_for_cpu(cpu, name=f"{tag}_{self.pdk.name}")
        fp_svg = os.path.join(self.out_dir, f"{tag}_floorplan.svg")
        floorplan_to_svg(fp, fp_svg)
        art.floorplan_svg = fp_svg

        die_area = fp.die_w * fp.die_h
        util = sum(b.w * b.h for b in fp.blocks) / max(1e-9, die_area)
        if util > self.pdk.max_utilization:
            reasons.append(f"Placement utilization too high: {util:.3f} > {self.pdk.max_utilization:.3f}")

        nets = [("Adder", "RegFile"), ("Logic(AND/OR/XOR)", "RegFile"), ("RegFile", "Control")]
        by_name = {b.name: b for b in fp.blocks}
        total_wl = 0.0
        for a, b in nets:
            if a in by_name and b in by_name:
                aa, bb = by_name[a], by_name[b]
                total_wl += abs((aa.x + aa.w / 2) - (bb.x + bb.w / 2)) + abs((aa.y + aa.h / 2) - (bb.y + bb.h / 2))

        r_total = total_wl * self.pdk.r_ohm_per_um
        c_total_ff = total_wl * self.pdk.c_ff_per_um
        added_delay_ps = 0.25 * (r_total * (c_total_ff * 1e-15)) * 1e12
        sta_fmax = 1.0 / ((((cpu.crit_path_ps + added_delay_ps) / max(1, cpu.cfg.pipeline_stages)) + 120.0) * 1e-12)

        dyn = cpu.evaluate_dynamic(cycles=1000, seed=123)
        temp_ss = cpu.cfg.t_amb + ((dyn.avg_power_w + cpu.cfg.leakage_w) * cpu.cfg.r_th)
        if temp_ss >= 95.0:
            reasons.append("Power/Thermal signoff failed (temp too high).")

        return EDAResult(
            ok=len(reasons) == 0,
            reasons=reasons,
            artifacts=art,
            est_added_delay_ps=added_delay_ps,
            est_added_cap_f=c_total_ff,
            util=util,
            wirelength_um=total_wl,
            sta_fmax_hz=sta_fmax,
            signoff_temp_ss_c=temp_ss,
        )


@dataclass
class DesignScore:
    score: float
    perf_ips: float
    avg_temp_c: float
    avg_power_w: float
    fmax_hz: float
    freq_hz: float
    timing_ratio: float


def compute_design_score(dyn: CPUDynamicResult, w_perf: float, w_temp: float, w_power: float, w_timing: float) -> DesignScore:
    perf_ips = (dyn.insn_retired / max(1, dyn.cycles)) * dyn.freq_hz
    ratio = dyn.freq_hz / max(1e-9, dyn.fmax_hz)
    score = (w_perf * math.log10(max(1.0, perf_ips))) - (w_temp * (dyn.avg_temp_c / 100.0)) - (w_power * dyn.avg_power_w) - (w_timing * max(0.0, ratio - 0.92))
    return DesignScore(score, perf_ips, dyn.avg_temp_c, dyn.avg_power_w, dyn.fmax_hz, dyn.freq_hz, ratio)


def mutate_cfg(cfg: CPUConfig, rng: random.Random) -> CPUConfig:
    c = CPUConfig(**asdict(cfg))
    if rng.random() < 0.35:
        c.vdd = min(1.25, max(0.75, c.vdd + rng.uniform(-0.08, 0.08)))
    if rng.random() < 0.4:
        c.freq_hz = min(600e6, max(5e6, c.freq_hz * rng.uniform(0.78, 1.28)))
    if rng.random() < 0.3:
        c.pipeline_stages = int(min(7, max(1, c.pipeline_stages + rng.choice([-1, 1]))))
    if rng.random() < 0.25:
        c.use_fast_adder = not c.use_fast_adder
    return c


def crossover(a: CPUConfig, b: CPUConfig, rng: random.Random) -> CPUConfig:
    return CPUConfig(**{k: (getattr(a, k) if rng.random() < 0.5 else getattr(b, k)) for k in asdict(a).keys()})


@dataclass
class Candidate:
    run_id: str
    cfg: CPUConfig
    dyn: CPUDynamicResult
    ds: DesignScore
    eda: EDAResult


class DesignerAI:
    def __init__(self, seed_cfg: CPUConfig, pdk: PDK, checks: EDAChecks, logger: logging.Logger, eval_cycles: int = 2500, design_budget_sec: int = 1800, rng_seed: int = 1234, out_dir: str = "out_design"):
        self.seed_cfg = seed_cfg
        self.eval_cycles = eval_cycles
        self.design_budget_sec = design_budget_sec
        self.rng = random.Random(rng_seed)
        self.out_dir = out_dir
        self.logger = logger
        os.makedirs(self.out_dir, exist_ok=True)
        self.runs_dir = os.path.join(self.out_dir, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self.eda = EDAFlow(pdk=pdk, checks=checks, out_dir=os.path.join(self.out_dir, "eda"), logger=logger)
        self.log_path = os.path.join(self.out_dir, "training_log.csv")
        self.best_svg = os.path.join(self.out_dir, "floorplan_best.svg")
        self.best_png = os.path.join(self.out_dir, "floorplan_best.png")
        self.best: Optional[Candidate] = None
        self.pool: List[Candidate] = []

    def _init_csv(self) -> None:
        if os.path.exists(self.log_path):
            return
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "iter", "run_id", "timestamp_ms", "valid_eda", "reasons", "score", "perf_ips", "avg_temp_c", "avg_power_w", "timing_ratio", "freq_hz", "fmax_hz", "crit_path_ps", "cap_total_f", "toggles", "vdd", "freq_target_hz", "leakage_w", "r_th", "c_th", "use_fast_adder", "pipeline_stages", "branch_rate", "mem_rate", "eda_util", "eda_wirelength_um", "eda_sta_fmax_hz", "eda_temp_ss_c",
            ])

    def _append_csv(self, it: int, cand: Candidate) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                it, cand.run_id, now_ms(), int(cand.eda.ok), ";".join(cand.eda.reasons), cand.ds.score, cand.ds.perf_ips, cand.ds.avg_temp_c, cand.ds.avg_power_w, cand.ds.timing_ratio, cand.dyn.freq_hz, cand.dyn.fmax_hz, cand.dyn.crit_path_ps, cand.dyn.cap_total_f, cand.dyn.toggles, cand.cfg.vdd, cand.cfg.freq_hz, cand.cfg.leakage_w, cand.cfg.r_th, cand.cfg.c_th, int(cand.cfg.use_fast_adder), cand.cfg.pipeline_stages, cand.cfg.branch_rate, cand.cfg.mem_rate, cand.eda.util, cand.eda.wirelength_um, cand.eda.sta_fmax_hz, cand.eda.signoff_temp_ss_c,
            ])

    def _save_best_floorplan(self, cand: Candidate) -> None:
        fp = build_floorplan_for_cpu(ToyCPU(cand.cfg), name=f"best_score={cand.ds.score:.4f}")
        floorplan_to_svg(fp, self.best_svg)
        floorplan_to_png_optional(fp, self.best_png)

    def _dump_run_json(self, it: int, cand: Candidate) -> None:
        with open(os.path.join(self.runs_dir, f"{cand.run_id}.json"), "w", encoding="utf-8") as f:
            json.dump({"iter": it, "run_id": cand.run_id, "cfg": asdict(cand.cfg), "dyn": asdict(cand.dyn), "score": asdict(cand.ds), "eda": {"ok": cand.eda.ok, "reasons": cand.eda.reasons, "metrics": {"util": cand.eda.util, "wirelength_um": cand.eda.wirelength_um, "sta_fmax_hz": cand.eda.sta_fmax_hz, "signoff_temp_ss_c": cand.eda.signoff_temp_ss_c, "added_delay_ps": cand.eda.est_added_delay_ps, "added_cap_f": cand.eda.est_added_cap_f}}}, f, indent=2)

    def _evaluate_cfg(self, cfg: CPUConfig, seed: int, run_id: str) -> Candidate:
        cpu = ToyCPU(cfg)
        eda = self.eda.run(cpu, run_id)
        dyn = cpu.evaluate_dynamic(cycles=self.eval_cycles, seed=seed)
        ds = compute_design_score(dyn, 1.0, 1.25, 0.35, 0.6)
        self.logger.info(f"[AI:{run_id}] score={ds.score:.5f} EDA={'PASS' if eda.ok else 'FAIL'} perf={ds.perf_ips / 1e6:.2f}MIPS T={ds.avg_temp_c:.2f}C")
        return Candidate(run_id, cfg, dyn, ds, eda)

    def learn(self) -> Candidate:
        self._init_csv()
        start = time.time()
        it = 0
        seed = self._evaluate_cfg(self.seed_cfg, self.rng.randrange(1 << 30), "iter000000_seed")
        self.pool = [seed]
        self.best = seed if seed.eda.ok else None
        self._append_csv(it, seed)
        self._dump_run_json(it, seed)
        if self.best:
            self._save_best_floorplan(self.best)

        while (time.time() - start) < self.design_budget_sec:
            it += 1
            run_id = f"iter{it:06d}"
            top = sorted(self.pool, key=lambda c: c.ds.score, reverse=True)[: max(3, len(self.pool) // 3)]
            child = mutate_cfg(crossover(self.rng.choice(top).cfg, self.rng.choice(top).cfg, self.rng), self.rng)
            cand = self._evaluate_cfg(child, self.rng.randrange(1 << 30), run_id)
            self.pool.append(cand)
            if len(self.pool) > 40:
                self.pool = sorted(self.pool, key=lambda c: c.ds.score, reverse=True)[:40]

            self._append_csv(it, cand)
            self._dump_run_json(it, cand)
            if cand.eda.ok and (self.best is None or cand.ds.score > self.best.ds.score):
                self.best = cand
                self._save_best_floorplan(cand)
                self.logger.info(f"[AI] NEW BEST {run_id} score={cand.ds.score:.5f}")

        if self.best is None:
            self.best = sorted(self.pool, key=lambda c: c.ds.score, reverse=True)[0]
        return self.best


def pretty_candidate(c: Candidate) -> str:
    return f"run={c.run_id} | EDA={'PASS' if c.eda.ok else 'FAIL'} | score={c.ds.score:.5f} | perf={c.ds.perf_ips / 1e6:.3f} MIPS | Tavg={c.ds.avg_temp_c:.2f}C"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU toy builder + EDA + AI")
    p.add_argument("--out-dir", default="out_design")
    p.add_argument("--design-budget-sec", type=int, default=30 * 60)
    p.add_argument("--eval-cycles", type=int, default=2500)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR"])
    p.add_argument("--rng-seed", type=int, default=42)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = setup_logger(args.out_dir, level=getattr(logging, args.log_level))

    ai = DesignerAI(
        seed_cfg=CPUConfig(freq_hz=120e6),
        pdk=PDK(name="toy28", metal_layers=6),
        checks=EDAChecks(),
        logger=logger,
        eval_cycles=args.eval_cycles,
        design_budget_sec=args.design_budget_sec,
        rng_seed=args.rng_seed,
        out_dir=args.out_dir,
    )
    best = ai.learn()
    print("=== BEST DESIGN FOUND ===")
    print(pretty_candidate(best))


if __name__ == "__main__":
    main()
