#!/usr/bin/env python3
"""
RL-SDN Interactive Terminal User Interface (TUI).
Production-grade dashboard and execution environment supporting all 10 architectural enhancements:
  1. Hierarchical Multi-Agent Coordination
  2. Prioritized Experience Replay (PER)
  3. Dueling DQN Architecture
  4. Link Failure Detection & Recovery (MTTR)
  5. Multi-Traffic Types (VoIP, IoT, Bursty, Bulk, Poisson)
  6. Progressive Difficulty Curriculum Learning
  7. Double DQN with Soft Updates
  8. Real-Time Performance Dashboard & Plots
  9. Transfer Learning & Topology Adaptation
 10. Policy Distillation for Edge Switches
"""
import sys
import os
import copy
import yaml
import logging
import pytest
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.ai.hierarchical_agent import HierarchicalSDNController
from rl_sdn_controller.ai.curriculum import CurriculumTrainer
from rl_sdn_controller.ai.transfer import TransferLearningManager
from rl_sdn_controller.ai.distillation import PolicyDistiller
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator
from rl_sdn_controller.network.routing_engine import RLRoutingEngine, OSPFRoutingEngine, RoundRobinRoutingEngine
from rl_sdn_controller.network.state_manager import StateManager
from rl_sdn_controller.cli.visualizer import TerminalVisualizer
from rl_sdn_controller.sdn_api.stats_provider import LinkStats

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = Console()
viz = TerminalVisualizer()



def packet_weighted_latency(telem):
    """Compute packet-count-weighted average latency (physically correct metric)."""
    total_lat_w = sum(st.avg_latency_ms * st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    total_pkt = sum(st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    return total_lat_w / total_pkt if total_pkt > 0 else 0.0


def load_configs(chaos_enabled: bool = True):
    top_path = "configs/topology.yaml"
    traffic_path = "configs/traffic_profiles.yaml"
    rl_path = "configs/rl_config.yaml"
    chaos_path = "configs/chaos_config.yaml" if chaos_enabled else None

    with open(traffic_path, "r") as f:
        traffic_configs = yaml.safe_load(f).get("flows", [])
    with open(rl_path, "r") as f:
        rl_config = yaml.safe_load(f)
    chaos_config = None
    if chaos_path and os.path.exists(chaos_path):
        with open(chaos_path, "r") as f:
            chaos_config = yaml.safe_load(f)

    return top_path, traffic_configs, rl_config, chaos_config


def aggregate_telemetry(history):
    if not history:
        return {}
    aggregated = {}
    link_keys = history[0].keys()
    for key in link_keys:
        sample = history[0][key]
        total_tx_bytes = sum(h[key].tx_bytes for h in history if key in h)
        total_tx_p = sum(h[key].tx_packets for h in history if key in h)
        total_drop_p = sum(h[key].dropped_packets for h in history if key in h)
        total_arrived = total_tx_p + total_drop_p
        drop_rate = (total_drop_p / total_arrived * 100.0) if total_arrived > 0 else 0.0

        avg_util = float(np.mean([h[key].utilization_pct for h in history if key in h]))
        valid_lat = [h[key].avg_latency_ms for h in history if key in h and h[key].avg_latency_ms > 0]
        avg_lat = float(np.mean(valid_lat)) if valid_lat else 0.0
        avg_q = int(np.mean([h[key].queue_depth for h in history if key in h]))

        aggregated[key] = LinkStats(
            src=sample.src,
            dst=sample.dst,
            capacity_mbps=sample.capacity_mbps,
            tx_bytes=total_tx_bytes,
            tx_packets=total_tx_p,
            dropped_bytes=total_drop_p * 1000,
            dropped_packets=total_drop_p,
            queue_depth=avg_q,
            max_queue_packets=sample.max_queue_packets,
            utilization_pct=avg_util,
            drop_rate_pct=drop_rate,
            avg_latency_ms=avg_lat
        )
    return aggregated


def evaluate_rl_agent(use_dueling: bool, episodes: int, chaos: bool, save_plots: bool = False):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    model_config = copy.deepcopy(rl_config)
    model_config["agent"]["use_dueling"] = use_dueling

    model_name = "Dueling DQN" if use_dueling else "Standard DQN"
    console.print(f"\n[bold green]⚙️  Training {model_name} for {episodes} episodes...[/bold green]")
    controller = SDNController(top_path, traffic_configs, model_config, chaos_config=chaos_config)
    controller.train_episodes(num_episodes=episodes, verbose=False)

    console.print(f"[bold green]📊 Evaluating {model_name} policy...[/bold green]")
    metrics, last_telem = controller.evaluate(max_steps=600)

    if save_plots:
        ospf_m = evaluate_ospf(chaos=chaos)
        rr_m = evaluate_round_robin(chaos=chaos)
        controller.metrics_tracker.set_baselines(ospf_m, rr_m)
        plot_path = controller.save_dashboard_plots(output_dir="plots", filename=f"{model_name.lower().replace(' ', '_')}_dashboard.png")
        console.print(f"[bold green]📈 Diagnostic plot saved to: {plot_path}[/bold green]")

    return metrics, last_telem, controller


def evaluate_ospf(chaos: bool):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = OSPFRoutingEngine(topo, rt_api)
    engine.update_routes(flow_ids, sim.flow_src_dst)

    tp_list, drop_list, lat_list = [], [], []
    for _ in range(600):
        for lq in sim.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim.step(0.01)
        telem = sim.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        tp_list.append(tp)
        drop_list.append(dr)
        lat_list.append(lat)

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def evaluate_round_robin(chaos: bool):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = RoundRobinRoutingEngine(topo, rt_api)

    tp_list, drop_list, lat_list = [], [], []
    for _ in range(600):
        engine.update_routes(flow_ids, sim.flow_src_dst)
        for lq in sim.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim.step(0.01)
        telem = sim.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        tp_list.append(tp)
        drop_list.append(dr)
        lat_list.append(lat)

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def run_full_comparison(episodes: int, chaos: bool):
    console.print("\n[bold cyan]🚀 RUNNING FULL MULTI-ALGORITHM BENCHMARK...[/bold cyan]")
    dueling_m, dueling_telem, ctrl = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos, save_plots=True)
    standard_m, _, _ = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos)
    ospf_m = evaluate_ospf(chaos=chaos)
    rr_m = evaluate_round_robin(chaos=chaos)

    table = Table(title=f"📊 BENCHMARK COMPARISON ({'CHAOS ACTIVE' if chaos else 'NORMAL NETWORK'})")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Dueling DQN (Proposed)", style="bold green")
    table.add_column("Standard DQN", style="bold cyan")
    table.add_column("Static OSPF", style="bold magenta")
    table.add_column("Round-Robin", style="bold blue")

    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_m['throughput_mbps']:.1f}",
        f"{standard_m['throughput_mbps']:.1f}",
        f"{ospf_m['throughput_mbps']:.1f}",
        f"{rr_m['throughput_mbps']:.1f}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_m['drop_rate_pct']:.2f}%",
        f"{standard_m['drop_rate_pct']:.2f}%",
        f"{ospf_m['drop_rate_pct']:.2f}%",
        f"{rr_m['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_m['avg_latency_ms']:.2f} ms",
        f"{standard_m['avg_latency_ms']:.2f} ms",
        f"{ospf_m['avg_latency_ms']:.2f} ms",
        f"{rr_m['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "Tail Latency P99 (ms)",
        f"{dueling_m['p99_latency_ms']:.2f} ms",
        f"{standard_m['p99_latency_ms']:.2f} ms",
        f"{ospf_m['p99_latency_ms']:.2f} ms",
        f"{rr_m['p99_latency_ms']:.2f} ms"
    )

    console.print(table)
    if dueling_telem:
        viz.print_link_telemetry_table(dueling_telem)


from rl_sdn_controller.network.dynamic_scenario import generate_random_production_scenario


def evaluate_model_on_packets(
    model_type: str,
    trained_controller: Optional[SDNController],
    target_packets: int,
    chaos: bool,
    custom_scenario: Optional[Tuple[NetworkTopology, List[Dict[str, Any]], Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates a specific model (Dueling DQN, Standard DQN, OSPF, or Round-Robin)
    on an exact target number of packets under a specific or randomized network scenario.
    """
    if custom_scenario is not None:
        topo, traffic_configs, chaos_config = custom_scenario
        if not chaos:
            chaos_config = None
    else:
        top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
        topo = NetworkTopology(top_path)

    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())

    if model_type == "ospf":
        engine = OSPFRoutingEngine(topo, rt_api)
        engine.update_routes(flow_ids, sim.flow_src_dst)
    elif model_type == "round_robin":
        engine = RoundRobinRoutingEngine(topo, rt_api)
    elif model_type in ["dueling_dqn", "standard_dqn"]:
        engine = RLRoutingEngine(topo, rt_api, flow_ids, sim.flow_src_dst)
        state_mgr = StateManager(topo)
        engine.apply_action(0)

    # Run packet processing loop — collect TRUE end-to-end latency per delivered packet
    total_tx_bytes = 0
    total_tx_pkts = 0
    total_drop_pkts = 0
    e2e_latencies_ms = []   # True end-to-end latency per delivered packet

    sim.reset()
    if model_type == "ospf":
        engine.update_routes(flow_ids, sim.flow_src_dst)
    elif model_type in ["dueling_dqn", "standard_dqn"]:
        engine.apply_action(0)

    step_dt = 0.01          # 10ms micro-step
    rl_step_interval = 0.1  # 100ms RL control interval
    time_since_rl = 0.0
    last_telem = {}

    while total_tx_pkts + total_drop_pkts < target_packets and sim.current_time < 300.0:

        # Reset window stats so this step's telemetry is fresh
        for lq in sim.links.values():
            lq.reset_window_stats()

        sim.step(step_dt)

        # Collect fresh telemetry AFTER the step so it reflects current link state
        step_telem = sim.stats_provider.collect_window_telemetry(step_dt)

        # RL / routing decisions based on CURRENT post-step link state
        if model_type == "round_robin":
            engine.update_routes(flow_ids, sim.flow_src_dst)
        elif model_type in ["dueling_dqn", "standard_dqn"]:
            time_since_rl += step_dt
            if time_since_rl >= rl_step_interval:
                time_since_rl = 0.0
                obs = state_mgr.get_observation_vector(step_telem)  # Fresh post-step obs
                action = trained_controller.agent.select_action(obs, evaluate=True)
                engine.apply_action(action)

        # Collect TRUE end-to-end latency from packets delivered this step
        for pkt in sim.delivered_packets:
            total_tx_bytes += pkt.size_bytes
            total_tx_pkts += 1
            if pkt.e2e_latency_ms > 0:
                e2e_latencies_ms.append(pkt.e2e_latency_ms)

        # Use simulator's precise per-step drop counter (tracks each enqueue() failure)
        total_drop_pkts += sim.step_dropped_packets

        last_telem = step_telem

    total_arrived = total_tx_pkts + total_drop_pkts
    drop_pct = (total_drop_pkts / total_arrived * 100.0) if total_arrived > 0 else 0.0
    throughput = (total_tx_bytes * 8.0 / 1_000_000.0) / max(0.001, sim.current_time)
    avg_lat = float(np.mean(e2e_latencies_ms)) if e2e_latencies_ms else 0.0
    p99_lat = float(np.percentile(e2e_latencies_ms, 99)) if e2e_latencies_ms else 0.0

    return {
        "target_packets": target_packets,
        "total_packets_sent": total_arrived,
        "delivered_packets": total_tx_pkts,
        "dropped_packets": total_drop_pkts,
        "drop_rate_pct": drop_pct,
        "throughput_mbps": throughput,
        "avg_latency_ms": avg_lat,
        "p99_latency_ms": p99_lat,
        "sim_time_sec": sim.current_time,
        "telemetry": last_telem
    }




def run_custom_train_and_packet_eval():
    console.print("\n[bold cyan]🎯 CUSTOM WORKFLOW: TRAIN ON N CYCLES ➔ EVALUATE ON M PACKETS[/bold cyan]\n")
    
    # 1. Manually select train cycles
    train_cycles_str = Prompt.ask("Enter number of training cycles (episodes)", default="20")
    train_cycles = int(train_cycles_str)

    # 2. Manually select test packets
    test_packets_str = Prompt.ask("Enter number of test packets to evaluate against all models", default="5000")
    test_packets = int(test_packets_str)

    # 3. Chaos toggle
    chaos = Confirm.ask("Enable Network Chaos Engine (flapping links, BER drops, delay jitter)?", default=True)

    # 4. Randomized Real-World Scenario toggle
    random_scenario = Confirm.ask("Generate Randomized Real-World Production Scenario (asymmetric link capacities & mixed traffic)?", default=True)

    # Generate or load scenario
    if random_scenario:
        scenario_tuple = generate_random_production_scenario()
        console.print("[dim italic]🎲 Generated fresh randomized real-world production topology & traffic mix.[/dim italic]")
    else:
        scenario_tuple = None

    # Phase 1: Training Dueling DQN
    console.print(f"\n[bold green]🏋️ [PHASE 1] Training Proposed Dueling DQN for {train_cycles} cycles...[/bold green]")
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    dueling_cfg = copy.deepcopy(rl_config)
    dueling_cfg["agent"]["use_dueling"] = True
    
    # Train controller
    if scenario_tuple:
        # Train on dynamic scenario
        topo_obj, flows, ch_cfg = scenario_tuple
        import tempfile
        # Write temporary topology yaml for controller initialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(topo_obj.topology_dict, tf)
            temp_top_path = tf.name
        dueling_controller = SDNController(temp_top_path, flows, dueling_cfg, chaos_config=ch_cfg if chaos else None)
    else:
        dueling_controller = SDNController(top_path, traffic_configs, dueling_cfg, chaos_config=chaos_config)

    dueling_controller.train_episodes(num_episodes=train_cycles, verbose=True)

    # Train Standard DQN
    console.print(f"\n[bold green]🏋️ Training Baseline Standard DQN for {train_cycles} cycles...[/bold green]")
    std_cfg = copy.deepcopy(rl_config)
    std_cfg["agent"]["use_dueling"] = False
    if scenario_tuple:
        std_controller = SDNController(temp_top_path, flows, std_cfg, chaos_config=ch_cfg if chaos else None)
    else:
        std_controller = SDNController(top_path, traffic_configs, std_cfg, chaos_config=chaos_config)
    std_controller.train_episodes(num_episodes=train_cycles, verbose=False)

    # Clean up temp file if needed
    if scenario_tuple and os.path.exists(temp_top_path):
        os.remove(temp_top_path)

    # Phase 2: Evaluation on exact M packets
    console.print(f"\n[bold cyan]🧪 [PHASE 2] Evaluating all 4 models on {test_packets:,} test packets...[/bold cyan]")
    dueling_res = evaluate_model_on_packets("dueling_dqn", dueling_controller, test_packets, chaos, custom_scenario=scenario_tuple)
    std_res = evaluate_model_on_packets("standard_dqn", std_controller, test_packets, chaos, custom_scenario=scenario_tuple)
    ospf_res = evaluate_model_on_packets("ospf", None, test_packets, chaos, custom_scenario=scenario_tuple)
    rr_res = evaluate_model_on_packets("round_robin", None, test_packets, chaos, custom_scenario=scenario_tuple)


    # Phase 3: Display Comparison Results
    table = Table(title=f"\n🏆 PERFORMANCE COMPARISON ON {test_packets:,} TEST PACKETS ({'CHAOS ACTIVE' if chaos else 'NORMAL'})", border_style="bright_blue")
    table.add_column("Evaluation Metric", style="bold yellow")
    table.add_column("Dueling DQN (Proposed)", style="bold green", justify="right")
    table.add_column("Standard DQN", style="bold cyan", justify="right")
    table.add_column("Static OSPF Baseline", style="bold magenta", justify="right")
    table.add_column("Round-Robin Baseline", style="bold blue", justify="right")

    table.add_row(
        "Packets Sent",
        f"{dueling_res['total_packets_sent']:,}",
        f"{std_res['total_packets_sent']:,}",
        f"{ospf_res['total_packets_sent']:,}",
        f"{rr_res['total_packets_sent']:,}"
    )
    table.add_row(
        "Packets Delivered",
        f"{dueling_res['delivered_packets']:,}",
        f"{std_res['delivered_packets']:,}",
        f"{ospf_res['delivered_packets']:,}",
        f"{rr_res['delivered_packets']:,}"
    )
    table.add_row(
        "Packets Dropped",
        f"{dueling_res['dropped_packets']:,}",
        f"{std_res['dropped_packets']:,}",
        f"{ospf_res['dropped_packets']:,}",
        f"{rr_res['dropped_packets']:,}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_res['drop_rate_pct']:.2f}%",
        f"{std_res['drop_rate_pct']:.2f}%",
        f"{ospf_res['drop_rate_pct']:.2f}%",
        f"{rr_res['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_res['throughput_mbps']:.2f} Mbps",
        f"{std_res['throughput_mbps']:.2f} Mbps",
        f"{ospf_res['throughput_mbps']:.2f} Mbps",
        f"{rr_res['throughput_mbps']:.2f} Mbps"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_res['avg_latency_ms']:.2f} ms",
        f"{std_res['avg_latency_ms']:.2f} ms",
        f"{ospf_res['avg_latency_ms']:.2f} ms",
        f"{rr_res['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "Tail Latency P99 (ms)",
        f"{dueling_res['p99_latency_ms']:.2f} ms",
        f"{std_res['p99_latency_ms']:.2f} ms",
        f"{ospf_res['p99_latency_ms']:.2f} ms",
        f"{rr_res['p99_latency_ms']:.2f} ms"
    )

    console.print(table)

    if dueling_res.get("telemetry"):
        viz.print_link_telemetry_table(dueling_res["telemetry"])

    # Save diagnostic plots
    dueling_controller.metrics_tracker.set_baselines(ospf_res, rr_res)
    plot_file = dueling_controller.save_dashboard_plots(output_dir="plots", filename=f"eval_{test_packets}_packets.png")
    console.print(f"\n[bold green]📈 Performance diagnostic plots saved to: [underline]{plot_file}[/underline][/bold green]")


def main():
    os.system("clear" if os.name == "posix" else "cls")
    
    banner = Panel.fit(
        "[bold cyan]⚡ RL-SDN AUTOMATED CONTROL PLANE & PACKET SIMULATOR TUI ⚡[/bold cyan]\n"
        "[dim]Production-Grade Enhancements: Hierarchical RL | PER | DDQN | Multi-Traffic | Distillation | Curriculum[/dim]",
        border_style="bright_blue"
    )
    console.print(banner)

    console.print("\n[bold yellow]Select Mode to Run:[/bold yellow]")
    console.print("  [1] [bold green]Train on N Cycles ➔ Test on M Packets (Custom Comparison)[/bold green]")
    console.print("  [2] [bold cyan]Full Comparison Benchmark[/bold cyan] (Dueling DQN vs Standard DQN vs OSPF vs Round-Robin)")
    console.print("  [3] [bold magenta]Hierarchical Multi-Agent RL[/bold magenta] (Regional Agents + Global Coordinator)")
    console.print("  [4] [bold blue]Curriculum Learning[/bold blue] (3 Progressive Difficulty Stages)")
    console.print("  [5] [bold yellow]Policy Distillation[/bold yellow] (Edge Model Compression Benchmark)")
    console.print("  [6] [bold green]Transfer Learning[/bold green] (Topology Generalization & Layer Freezing)")
    console.print("  [7] [bold white]Dueling DQN Agent Only[/bold white]")
    console.print("  [8] [bold magenta]Standard DQN Agent Only[/bold magenta]")
    console.print("  [9] [bold blue]Static OSPF (Dijkstra Shortest Path)[/bold blue]")
    console.print("  [10] [bold yellow]Round-Robin (ECMP Load Balancer)[/bold yellow]")
    console.print("  [11] [bold red]Train Model & Export ONNX Policy[/bold red]")
    console.print("  [12] [bold cyan]Run Automated PyTest Suite[/bold cyan]")
    console.print("  [13] Exit\n")

    choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"], default="1")
    if choice == "13":
        console.print("[yellow]Exiting RL-SDN TUI. Goodbye![/yellow]")
        return

    if choice == "1":
        run_custom_train_and_packet_eval()
        return

    if choice == "12":
        console.print("\n[bold cyan]🧪 Running Automated Test Suite...[/bold cyan]\n")
        pytest.main(["tests/", "-v"])
        return

    if choice == "4":
        run_curriculum_demo()
        return

    if choice == "5":
        run_distillation_demo()
        return

    if choice == "6":
        run_transfer_learning_demo()
        return

    chaos = Confirm.ask("Enable Network Chaos Engine (flapping links, BER drops, delay jitter)?", default=True)
    episodes_str = Prompt.ask("Number of training / evaluation episodes", default="15")
    episodes = int(episodes_str)

    if choice == "2":
        run_full_comparison(episodes=episodes, chaos=chaos)

    elif choice == "3":
        run_hierarchical_demo(episodes=episodes, chaos=chaos)

    elif choice == "7":
        m, telem, _ = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos, save_plots=True)
        viz.print_benchmark_table(m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})
        if telem:
            viz.print_link_telemetry_table(telem)

    elif choice == "8":
        m, telem, _ = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos, save_plots=True)
        viz.print_benchmark_table(m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})
        if telem:
            viz.print_link_telemetry_table(telem)

    elif choice == "9":
        m = evaluate_ospf(chaos=chaos)
        viz.print_benchmark_table({"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})

    elif choice == "10":
        m = evaluate_round_robin(chaos=chaos)
        viz.print_benchmark_table({"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, m)

    elif choice == "11":
        onnx_file = Prompt.ask("Enter output ONNX filename", default="model.onnx")
        top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
        controller = SDNController(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
        controller.train_episodes(num_episodes=episodes, verbose=True)
        from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx
        export_policy_to_onnx(controller.agent.policy_net, controller.state_dim, onnx_file)
        console.print(f"[bold green]✅ Model exported to {onnx_file}[/bold green]")



if __name__ == "__main__":
    main()

