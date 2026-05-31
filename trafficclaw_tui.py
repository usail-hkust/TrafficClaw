"""Terminal UI for launching TrafficClaw LLM control simulations."""

import contextlib
import curses
import getpass
import math
import os
import re
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List, Optional, Sequence, Tuple

from trafficclaw_runner import (
    AVAILABLE_CONTROL_MODULES,
    SimulationRunOptions,
    TrafficClawSimulationEntrypoint,
    default_profile_for_modules,
    fill_defaults,
)
from utils.path_utils import resolve_config_path
from utils.prompt_utils import set_runtime_query_provider


MODULE_DESCRIPTIONS = {
    "signal_timing": "Traffic signal phases and timing plans.",
    "highway_speed_limit": "Variable speed limits. Uses highway zone configs.",
    "ramp_metering": "Ramp meter release rates. Uses highway zone configs.",
    "bus_scheduling": "Bus departures, headways, and service adjustment.",
    "subway_scheduling": "Subway departures, headways, and service adjustment.",
    "taxi_scheduling": "Taxi dispatch and repositioning.",
}

HIGHWAY_REGIONS = [
    ("manhattan", "Manhattan", "Data/sumo_config_highway/Manhattan/Manhattan.sumocfg"),
    ("queens", "Queens", "Data/sumo_config_highway/Queens/Queens.sumocfg"),
    ("brooklyn", "Brooklyn", "Data/sumo_config_highway/Brooklyn/Brooklyn.sumocfg"),
]

REGIONS = [
    ("manhattan", "Manhattan", "Data/sumo_config/Manhattan/Manhattan.sumocfg"),
    ("queens", "Queens", "Data/sumo_config/Queens/Queens.sumocfg"),
    ("brooklyn", "Brooklyn", "Data/sumo_config/Brooklyn/Brooklyn.sumocfg"),
]

HIGHWAY_CONFIG_MODULES = {"highway_speed_limit", "ramp_metering"}

PROVIDER_API_KEY_ENVS = {
    "siliconflow": ["SILICONFLOW_API_KEY", "SiliconFlow_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "bailian": ["DASHSCOPE_API_KEY", "BAILIAN_API_KEY"],
    "dashscope": ["DASHSCOPE_API_KEY", "BAILIAN_API_KEY"],
}

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


LOBSTER = r"""
 ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗
 ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝
    ██║   ██████╔╝███████║█████╗  █████╗  ██║██║     
    ██║   ██╔══██╗██╔══██║██╔══╝  ██╔══╝  ██║██║     
    ██║   ██║  ██║██║  ██║██║     ██║     ██║╚██████╗
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ ╚═════╝
     
   ██████╗██╗      █████╗ ██╗    ██╗
  ██╔════╝██║     ██╔══██╗██║    ██║
  ██║     ██║     ███████║██║ █╗ ██║
  ██║     ██║     ██╔══██║██║███╗██║
  ╚██████╗███████╗██║  ██║╚███╔███╔╝
   ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
"""


def main() -> None:
    _clear_screen()
    _print_header()
    _wait_for_enter()

    modules = _ask_modules()
    query = _ask_query()
    options = fill_defaults(SimulationRunOptions(control_modules=modules, user_query=query))
    options.duration = _ask_float("Simulation duration seconds", options.duration)
    options.checkpoint_interval = _ask_float("Checkpoint interval seconds", options.checkpoint_interval)
    options.llm_model = _ask_str("LLM model", options.llm_model)
    _ensure_api_key_for_model(options.llm_model)
    options.config = _ask_zone_config(modules)
    options.use_wandb = _ask_wandb_enabled(bool(options.use_wandb))
    if options.use_wandb:
        options.wandb_project = _ask_str("Wandb project", options.wandb_project)

    print()
    _section("Summary")
    _print_summary(options)
    if not _config_exists(options.config):
        print(f"{YELLOW}Warning: config file was not found after path resolution.{RESET}")
        print(f"{DIM}You can still go back by answering no at confirmation.{RESET}")

    if not _ask_bool("Start simulation now", False):
        print(f"{DIM}Cancelled.{RESET}")
        return

    print()
    _section("Checkpoint Metrics")
    metric_writer = CheckpointMetricWriter(
        total_duration=float(options.duration),
        checkpoint_interval=float(options.checkpoint_interval),
    )
    metric_writer.start()
    instruction_controller = RuntimeInstructionController(metric_writer)
    instruction_controller.start()
    set_runtime_query_provider(instruction_controller.consume_query_at_checkpoint)
    try:
        with contextlib.redirect_stdout(metric_writer), contextlib.redirect_stderr(SilentWriter()):
            results = TrafficClawSimulationEntrypoint().run(options)
    finally:
        set_runtime_query_provider(None)
        instruction_controller.cleanup()
        metric_writer.finish()

    print()
    _section("Finished")
    status = results.get("status", "unknown") if isinstance(results, dict) else "unknown"
    if status == "error":
        print(f"{RED}Status: error{RESET}")
        print(results.get("message", "No error message returned."))
        raise SystemExit(1)
    print(f"{GREEN}Status: {status}{RESET}")
    if isinstance(results, dict):
        simulation_id = results.get("simulation_id")
        if simulation_id:
            print(f"│ Simulation ID: {simulation_id}")


def _print_header() -> None:
    border = "#" * 72
    print(f"{CYAN}{border}{RESET}")
    print(f"{CYAN}#{RESET} {BOLD}TrafficClaw LLM Control TUI{RESET}".ljust(80) + f"{CYAN}#{RESET}")
    print(f"{CYAN}{border}{RESET}")
    for line in LOBSTER.splitlines():
        if not line:
            continue
        colored = (
            line.replace("R", f"{RED}R{CYAN}")
            .replace("Y", f"{YELLOW}Y{CYAN}")
            .replace("G", f"{GREEN}G{CYAN}")
        )
        print(f"{CYAN}{colored}{RESET}")
    print(f"{DIM}Select modules first, then provide an optional user query.{RESET}")
    print(f"{DIM}Press Enter to configure.{RESET}")


def _wait_for_enter() -> None:
    try:
        input()
    except EOFError:
        pass


def _ask_modules() -> List[str]:
    selected = _checkbox_select(
        title="Modules",
        subtitle="Choose which control modules the LLM can operate.",
        items=[
            (module, module, MODULE_DESCRIPTIONS.get(module, ""))
            for module in AVAILABLE_CONTROL_MODULES
        ],
        default_selected=["signal_timing"],
        allow_multiple=True,
    )
    if not selected:
        return ["signal_timing"]
    return selected


def _ask_query() -> Optional[str]:
    print()
    _section("Query")
    print(f"{DIM}│ Optional. Leave empty to use the default query from the selected runner.{RESET}")
    print(f"{DIM}│ During simulation, the query input stays open for next-checkpoint instructions.{RESET}")
    query = input("│ User query [default]: ").strip()
    return query or None


def _ask_zone_config(modules: Sequence[str]) -> str:
    regions = HIGHWAY_REGIONS if _needs_highway_config(modules) else REGIONS
    title = "Zone"
    subtitle = (
        "Select SUMO region. Highway/ramp modules use highway config."
        if _needs_highway_config(modules)
        else "Select SUMO region. Standard Data/sumo_config will be used."
    )
    selected = _checkbox_select(
        title=title,
        subtitle=subtitle,
        items=[
            (region_key, region_label, config)
            for region_key, region_label, config in regions
        ],
        default_selected=[regions[0][0]],
        allow_multiple=False,
    )
    selected_key = selected[0] if selected else regions[0][0]
    for region_key, _, config in regions:
        if region_key == selected_key:
            return config
    return regions[0][2]


def _ask_wandb_enabled(default: bool) -> bool:
    selected = _checkbox_select(
        title="Wandb",
        subtitle="Choose whether this run should log metrics to wandb.",
        items=[
            ("yes", "Use wandb logging", "Record checkpoint and final metrics."),
            ("no", "Do not use wandb", "Run locally without remote logging."),
        ],
        default_selected=["yes" if default else "no"],
        allow_multiple=False,
    )
    return (selected[0] if selected else ("yes" if default else "no")) == "yes"


def _print_summary(options: SimulationRunOptions) -> None:
    modules = ", ".join(options.control_modules)
    profile = default_profile_for_modules(options.control_modules)
    mode = "single-module runner" if len(options.control_modules) == 1 else "joint runner"
    print(f"│ Mode: {BOLD}{mode}{RESET}")
    print(f"│ Modules: {modules}")
    print(f"│ Config: {options.config}")
    print(f"│ Duration: {options.duration}s")
    print(f"│ Checkpoint interval: {options.checkpoint_interval}s")
    print(f"│ LLM model: {options.llm_model}")
    print(f"│ Max agent turns: {options.max_agent_turns} (runner default)")
    print(f"│ Step seconds: {options.step_seconds}s (runner default)")
    print(f"│ Traffic state interval: {options.traffic_state_interval}s (runner default)")
    print(f"│ Wandb: {options.use_wandb} ({options.wandb_project or profile.wandb_project})")
    print(f"│ SUMO GUI: {options.use_gui}")
    print(f"│ Query: {options.user_query or '[default prompt only]'}")


def _config_exists(config: Optional[str]) -> bool:
    if not config:
        return False
    workspace_root = Path(__file__).resolve().parent
    return resolve_config_path(config, workspace_root).exists()


def _section(title: str) -> None:
    print(f"{MAGENTA}{BOLD}◆  {title}{RESET}")


def _ask_str(label: str, default: Optional[str]) -> str:
    value = input(f"│ {label} [{default}]: ").strip()
    return value or str(default or "")


def _ask_float(label: str, default: Optional[float]) -> float:
    value = input(f"│ {label} [{default}]: ").strip()
    return float(value) if value else float(default)


def _ask_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"│ {label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true", "t"}


def _ensure_api_key_for_model(llm_model: Optional[str]) -> None:
    provider = _llm_provider_prefix(llm_model)
    if not provider or provider == "local_llm":
        return

    env_names = PROVIDER_API_KEY_ENVS.get(provider)
    if not env_names:
        return

    existing_key = next((os.environ.get(name) for name in env_names if os.environ.get(name)), None)
    if existing_key:
        primary = env_names[0]
        os.environ.setdefault(primary, existing_key)
        return

    primary = env_names[0]
    aliases = ", ".join(env_names)
    print()
    _section("API Key")
    print(f"{YELLOW}│ {provider} model selected, but none of {aliases} is set.{RESET}")
    print(f"{DIM}│ This key is only stored in the current process for this run.{RESET}")
    print(f"{DIM}│ Next time, set it before launching, for example:{RESET}")
    print(f"{DIM}│   export {primary}=\"your-api-key\"{RESET}")

    api_key = ""
    while not api_key:
        api_key = getpass.getpass(f"│ Enter {primary}: ").strip()
        if not api_key:
            print(f"{YELLOW}│ API key cannot be empty for provider '{provider}'.{RESET}")

    for env_name in env_names:
        os.environ[env_name] = api_key
    print(f"{GREEN}│ API key loaded for this TUI session.{RESET}")


def _llm_provider_prefix(llm_model: Optional[str]) -> Optional[str]:
    if not llm_model:
        return None
    model = llm_model.strip()
    if model.lower() == "local_llm":
        return "local_llm"
    if "/" not in model:
        return None
    return model.split("/", 1)[0].strip().lower()


def _needs_highway_config(modules: Iterable[str]) -> bool:
    return any(module in HIGHWAY_CONFIG_MODULES for module in modules)


def _runtime_input_listener(
    queue: Queue,
    writer: "CheckpointMetricWriter",
    stop_event: threading.Event,
) -> None:
    stdin = getattr(sys, "__stdin__", None)
    if stdin is not None and not getattr(stdin, "closed", True):
        if _read_runtime_queries(stdin, queue, writer, stop_event):
            return

    try:
        with open("/dev/tty", "r+", encoding="utf-8", buffering=1) as tty_file:
            _read_runtime_queries(tty_file, queue, writer, stop_event)
    except BaseException as exc:
        writer.mark_runtime_instruction_unavailable(str(exc))
        return


def _read_runtime_queries(
    input_file,
    queue: Queue,
    writer: "CheckpointMetricWriter",
    stop_event: threading.Event,
) -> bool:
    try:
        while not stop_event.is_set():
            query = input_file.readline()
            if query == "":
                return False
            if stop_event.is_set():
                return True
            queue.put(query.strip())
            writer.mark_runtime_instruction_captured(has_query=bool(query.strip()))
            if not stop_event.is_set():
                writer.mark_runtime_instruction_input_ready()
        return True
    except BaseException:
        return False


class CheckpointMetricWriter:
    """Filter noisy runner stdout and render only checkpoint metrics."""

    def __init__(self, total_duration: float, checkpoint_interval: float) -> None:
        self._buffer = ""
        self._current_checkpoint: Optional[str] = None
        self._section: Optional[str] = None
        self._out = sys.__stdout__
        self._total_duration = max(1.0, total_duration)
        self._checkpoint_interval = max(1.0, checkpoint_interval)
        self._total_checkpoints = max(1, int(math.ceil(self._total_duration / self._checkpoint_interval)))
        self._llm_preview_pending = False
        self._printed_llm_preview = False
        self._runtime_input_prompt_visible = False
        self._runtime_input_intro_printed = False
        self._output_lock = threading.RLock()
        self._prompt_restore_timer: Optional[threading.Timer] = None

    def start(self) -> None:
        self._emit_progress(simulated_seconds=0.0)
        self._emit(
            f"{DIM}│ Type a query at any time, then press Enter to apply it at the next checkpoint; "
            f"Ctrl+C exits.{RESET}"
        )
        self.mark_runtime_instruction_input_ready()

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(text)

    def flush(self) -> None:
        self._out.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._out, "encoding", "utf-8")

    def finish(self) -> None:
        self._cancel_prompt_restore_timer()
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        self.flush()

    def _handle_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        running_match = re.search(r"\[Checkpoint\s+(\d+)\]\s+Running simulation for", line)
        if running_match:
            checkpoint_number = int(running_match.group(1))
            self._emit("")
            self._emit(
                f"{DIM}│ simulating outer interval {checkpoint_number}/{self._total_checkpoints}..."
                f"{RESET}"
            )
            self._emit(
                f"{DIM}│ runtime query input is active · Ctrl+C: exit{RESET}"
            )
            return

        checkpoint_match = re.search(r"\[Checkpoint\s+(\d+)\]\s+Checkpoint reached!", line)
        if checkpoint_match:
            self._current_checkpoint = checkpoint_match.group(1)
            self._section = None
            simulated_seconds = min(
                self._total_duration,
                int(self._current_checkpoint) * self._checkpoint_interval,
            )
            self._emit("")
            self._emit(f"{MAGENTA}{BOLD}◆  Checkpoint {self._current_checkpoint}{RESET}")
            self._emit_progress(simulated_seconds=simulated_seconds)
            return

        if "LLM Agent Optimization" in line or "LLM AGENT OPTIMIZATION SESSION" in line:
            self._section = None
            self._printed_llm_preview = False
            self._llm_preview_pending = False
            self._emit(f"{DIM}│ thinking... preparing LLM optimization{RESET}")
            return

        turn_match = re.search(r"---\s+(?:Reflection\s+)?Turn\s+([0-9]+/[0-9]+)\s+---", line)
        if turn_match:
            self._llm_preview_pending = False
            self._printed_llm_preview = False
            self._emit(f"{DIM}│ thinking... turn {turn_match.group(1)}{RESET}")
            return

        if line.startswith("LLM Response:"):
            self._llm_preview_pending = True
            self._printed_llm_preview = False
            return

        if self._llm_preview_pending and not self._printed_llm_preview:
            preview = self._preview_text(line)
            if preview:
                self._emit(f"{DIM}│   {preview}{RESET}")
                self._printed_llm_preview = True
                self._llm_preview_pending = False
            return

        if line.startswith("Parsed Action:"):
            action = line.split(":", 1)[1].strip() if ":" in line else ""
            if action:
                self._emit(f"{DIM}│ thinking... action={action}{RESET}")
            return

        if line.startswith("Elapsed time:") or line.startswith("Remaining duration:"):
            self._emit(f"│ {line}")
            return

        if line.startswith("Control Module Performance Metrics"):
            self._section = "module"
            self._emit(f"│ {BOLD}Control module metrics{RESET}")
            return

        if line.startswith("Travel Time Metrics"):
            self._section = "travel"
            self._emit(f"│ {BOLD}Travel time{RESET}")
            return

        if line.startswith("Waiting Time Metrics"):
            self._section = "waiting"
            self._emit(f"│ {BOLD}Waiting time{RESET}")
            return

        if line.startswith("Baseline Simulation Module Performance Metrics"):
            self._section = "baseline"
            return

        if line.startswith("ERROR") or line.startswith("Error:"):
            self._emit(f"│ {RED}{line}{RESET}")
            return

        if self._section == "module" and line.endswith(":") and not line.startswith("-"):
            label = line[:-1].replace("_", " ").title()
            self._emit(f"│  ● {CYAN}{label}{RESET}")
            return

        if self._section in {"module", "travel", "waiting"} and line.startswith("- "):
            metric = line[2:]
            if ":" in metric:
                name, value = metric.split(":", 1)
                self._emit(f"│     {name:<42} {GREEN}{value.strip()}{RESET}")
            else:
                self._emit(f"│     {metric}")

    def _emit(self, line: str) -> None:
        with self._output_lock:
            self._cancel_prompt_restore_timer_locked()
            if self._runtime_input_prompt_visible:
                self._out.write("\n")
                self._runtime_input_prompt_visible = False
            self._out.write(f"{line}\n")
            self._out.flush()
            self._schedule_prompt_restore_locked()

    def _prompt_runtime_query(self) -> None:
        with self._output_lock:
            self._cancel_prompt_restore_timer_locked()
            if self._runtime_input_prompt_visible:
                return
            self._out.write(
                f"│ User query for next checkpoint "
                f"{DIM}(leave blank to execute the current request){RESET}: "
            )
            self._out.flush()
            self._runtime_input_prompt_visible = True

    def _schedule_prompt_restore_locked(self) -> None:
        if not self._runtime_input_intro_printed:
            return
        timer = threading.Timer(0.05, self._prompt_runtime_query)
        timer.daemon = True
        self._prompt_restore_timer = timer
        timer.start()

    def _cancel_prompt_restore_timer(self) -> None:
        with self._output_lock:
            self._cancel_prompt_restore_timer_locked()

    def _cancel_prompt_restore_timer_locked(self) -> None:
        if self._prompt_restore_timer is not None:
            self._prompt_restore_timer.cancel()
            self._prompt_restore_timer = None

    def _emit_progress(self, simulated_seconds: float) -> None:
        done = min(max(simulated_seconds, 0.0), self._total_duration)
        ratio = done / self._total_duration
        width = 28
        filled = int(round(width * ratio))
        bar = "█" * filled + "░" * (width - filled)
        percent = ratio * 100
        self._emit(
            f"│ Progress [{CYAN}{bar}{RESET}] "
            f"{self._format_seconds(done)}/{self._format_seconds(self._total_duration)} "
            f"{percent:5.1f}%"
        )

    def _format_seconds(self, seconds: float) -> str:
        seconds_int = int(round(seconds))
        if seconds_int >= 3600:
            return f"{seconds_int / 3600:.1f}h"
        return f"{seconds_int}s"

    def _preview_text(self, line: str) -> str:
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned or set(cleaned) <= {"=", "-", "`"}:
            return ""
        if len(cleaned) > 120:
            cleaned = f"{cleaned[:117]}..."
        return cleaned

    def mark_runtime_instruction_input_ready(self) -> None:
        if not self._runtime_input_intro_printed:
            self._emit("")
            self._emit(f"{MAGENTA}{BOLD}◆  Runtime Query{RESET}")
            self._emit(f"{DIM}│ Input stays active while simulation runs.{RESET}")
            self._emit(f"{DIM}│ Press Enter to queue a query for the next checkpoint.{RESET}")
            self._runtime_input_intro_printed = True
        self._prompt_runtime_query()

    def mark_runtime_instruction_result(self, has_query: bool) -> None:
        if has_query:
            self._emit(f"{GREEN}│ Instruction applied to this checkpoint prompt.{RESET}")
        else:
            self._emit(f"{DIM}│ No new instruction. Continuing with the existing prompt.{RESET}")

    def mark_runtime_instruction_captured(self, has_query: bool) -> None:
        if has_query:
            self._emit("")
            self._emit(f"{GREEN}│ Instruction received; it will apply at the next checkpoint.{RESET}")
        else:
            self._emit("")
            self._emit(f"{DIM}│ Empty instruction received; current prompt will be kept.{RESET}")

    def mark_runtime_instruction_unavailable(self, reason: str) -> None:
        self._emit("")
        self._emit(f"{YELLOW}│ Runtime query input is unavailable: {reason or 'unknown error'}{RESET}")


class RuntimeInstructionController:
    """Listen for runtime instruction requests during a TUI simulation run."""

    def __init__(self, writer: CheckpointMetricWriter) -> None:
        self._writer = writer
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._input_queue: Queue = Queue()

    def start(self) -> None:
        self._listener_thread = threading.Thread(
            target=_runtime_input_listener,
            args=(self._input_queue, self._writer, self._stop_event),
            daemon=True,
        )
        self._listener_thread.start()

    def consume_query_at_checkpoint(self) -> Optional[str]:
        query = ""
        while True:
            try:
                query = self._input_queue.get_nowait()
            except Empty:
                break

        query = (query or "").strip()
        if query:
            self._writer.mark_runtime_instruction_result(has_query=True)
        return query or None

    def cleanup(self) -> None:
        self._stop_event.set()
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=0.5)
        self._listener_thread = None


class SilentWriter:
    """Discard noisy stderr from SUMO/LLM libraries during TUI runs."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def _checkbox_select(
    title: str,
    items: Sequence[Tuple[str, ...]],
    default_selected: Optional[Sequence[str]] = None,
    allow_multiple: bool = True,
    subtitle: str = "",
) -> List[str]:
    if not items:
        return []
    try:
        return curses.wrapper(
            _checkbox_select_screen,
            title,
            subtitle,
            items,
            set(default_selected or []),
            allow_multiple,
        )
    except curses.error:
        return _checkbox_select_fallback(title, items, default_selected, allow_multiple)


def _checkbox_select_screen(
    stdscr,
    title: str,
    subtitle: str,
    items: Sequence[Tuple[str, ...]],
    default_selected: set,
    allow_multiple: bool,
) -> List[str]:
    curses.curs_set(0)
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        pass
    current = 0
    selected = {item[0] for item in items if item[0] in default_selected}
    if not selected and not allow_multiple:
        selected.add(items[0][0])

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        _add_clipped(stdscr, 1, 2, f"◆  {title}", curses.A_BOLD)
        if subtitle:
            _add_clipped(stdscr, 2, 2, f"│  {subtitle}", curses.A_DIM)
        _add_clipped(stdscr, 4, 2, "│  Space select   Enter confirm   ↑/↓ move", curses.A_DIM)

        start_line = 6
        marker_selected = "◼" if allow_multiple else "●"
        marker_empty = "◻" if allow_multiple else "○"
        for index, item in enumerate(items):
            value, label, description = _item_parts(item)
            marker = marker_selected if value in selected else marker_empty
            attrs = curses.A_REVERSE if index == current else curses.A_NORMAL
            prefix = "│  "
            line = f"{prefix}{marker} {label}"
            _add_clipped(stdscr, start_line + index, 2, line, attrs)
            if description and start_line + index < height - 3:
                desc_x = min(2 + len(line) + 2, max(2, width - 20))
                _add_clipped(stdscr, start_line + index, desc_x, description, curses.A_DIM)

        footer_y = min(height - 2, start_line + len(items) + 2)
        footer = (
            "└─ multi-select; empty selection falls back to signal_timing"
            if allow_multiple
            else "└─ single-select; press Space to change selection"
        )
        _add_clipped(stdscr, footer_y, 2, footer, curses.A_DIM)

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            current = (current - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j")):
            current = (current + 1) % len(items)
        elif key == ord(" "):
            value = items[current][0]
            if allow_multiple:
                if value in selected:
                    selected.remove(value)
                else:
                    selected.add(value)
            else:
                selected = {value}
        elif key in (curses.KEY_ENTER, 10, 13):
            if selected:
                return [item[0] for item in items if item[0] in selected]
            if allow_multiple:
                return []
            return [items[0][0]]


def _checkbox_select_fallback(
    title: str,
    items: Sequence[Tuple[str, ...]],
    default_selected: Optional[Sequence[str]],
    allow_multiple: bool,
) -> List[str]:
    print()
    _section(title)
    for index, item in enumerate(items, start=1):
        _, label, description = _item_parts(item)
        suffix = f" ({description})" if description else ""
        print(f"│ {index}. {label}{DIM}{suffix}{RESET}")
    default_indexes = [
        str(index)
        for index, item in enumerate(items, start=1)
        if item[0] in set(default_selected or [])
    ]
    default_text = " ".join(default_indexes) if default_indexes else "1"
    raw = input(f"│ Select item numbers [{default_text}]: ").strip() or default_text
    selected_indexes = []
    for token in raw.replace(",", " ").split():
        if token.isdigit():
            selected_indexes.append(int(token))
    if not allow_multiple and selected_indexes:
        selected_indexes = selected_indexes[:1]
    values = []
    for index in selected_indexes:
        if 1 <= index <= len(items):
            values.append(items[index - 1][0])
    return values


def _item_parts(item: Tuple[str, ...]) -> Tuple[str, str, str]:
    value = item[0]
    label = item[1] if len(item) > 1 else item[0]
    description = item[2] if len(item) > 2 else ""
    return value, label, description


def _add_clipped(stdscr, y: int, x: int, text: str, attrs: int = curses.A_NORMAL) -> None:
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width - 1:
        return
    available = max(0, width - x - 1)
    if available <= 0:
        return
    try:
        stdscr.addstr(y, x, text[:available], attrs)
    except curses.error:
        pass


def _clear_screen() -> None:
    print("\033[2J\033[H", end="")


if __name__ == "__main__":
    main()
