import matplotlib.pyplot as plt
import numpy as np

def plot_all_metrics(options_files, output_dir, test_name):
    metrics_data = {
        'p99': [],
        'ops_per_sec': [],
        'write_amp': [],
        'read_amp': []
    }
    for config in options_files:
        if len(config) >= 5:
            _, results, _, _, is_failed_tuning = config
        else:
            _, results, _, _ = config
            
        if results:
            metrics_data['p99'].append(results.get('p99'))
            metrics_data['ops_per_sec'].append(results.get('ops_per_sec'))
            metrics_data['write_amp'].append(results.get('write_amp'))
            metrics_data['read_amp'].append(results.get('read_amp'))
    print(f"[DEBUG][plot_all_metrics] p99: {metrics_data['p99']}")
    print(f"[DEBUG][plot_all_metrics] ops_per_sec: {metrics_data['ops_per_sec']}")
    print(f"[DEBUG][plot_all_metrics] write_amp: {metrics_data['write_amp']}")
    print(f"[DEBUG][plot_all_metrics] read_amp: {metrics_data['read_amp']}")

    plot_metrics_comparison(options_files, output_dir, test_name)
    plot_metrics_heatmap(options_files, output_dir, test_name)
    plot_metrics_radar(options_files, output_dir, test_name)
    plot_metrics_trends(options_files, output_dir, test_name)
    
def plot(values, title, file):
    y_limit = 1.5*max(values)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values, label=title)

    for i, value in enumerate(values):
        plt.text(i, value + (y_limit*.02), str(value), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel("Throughput (ops/sec)")
    plt.legend()

    plt.ylim(0, y_limit)

    plt.savefig(file)
    plt.close()


def plot_2axis(keys, values, title, file):
    plt.figure(figsize=(12, 6))
    plt.plot(keys, values, label=title, linestyle='-')

    plt.title(title)
    plt.legend()
    plt.xlabel("Time (sec)")
    plt.ylabel("Throughput (ops/sec)")
    plt.grid(True)

    plt.ylim(0, 1.5*max(values))

    plt.savefig(file)
    plt.close()


def plot_multiple(data, title, file):
    plt.figure(figsize=(12, 6))
    for i, iteration in enumerate(data):
        keys, values = iteration[1]["ops_per_second_graph"]
        plt.plot(keys, values, label=f"Iteration-{i}", linestyle='-')

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.ylim(0, 1.5*max(max(row) for row in [x[1]["ops_per_second_graph"][1] for x in data]))

    plt.savefig(file)
    plt.close()

def plot_multiple_manual(data, file):
    plt.figure(figsize=(16.5, 8))
    labels = ["Default file", "Iteration 2", "Iteration 4", "Iteration 6"]
    colors = ['red', 'orange', 'royalblue', 'green'] 
    for i, ops in enumerate(data):
        plt.plot(ops, label=f"{labels[i]}", linestyle='-',color=colors[i])
    plt.xlabel("Time (seconds)")  
    plt.ylabel("Throughput (kops/s)")  
    plt.legend()


    plt.ylim(0, 400)
    plt.tight_layout()

    plt.savefig(file)
    plt.close()

def plot_metrics_comparison(options_files, output_dir, test_name, suffix=""):
    metrics_data = {
        'p99': [],
        'ops_per_sec': [],
        'write_amp': [],
        'read_amp': []
    }
    failed_indices = []
    
    for idx, config in enumerate(options_files):
        if len(config) >= 5:
            _, results, _, _, is_failed_tuning = config
            if is_failed_tuning:
                failed_indices.append(idx)
        else:
            _, results, _, _ = config
            
        if results:
            metrics_data['p99'].append(results.get('p99'))
            metrics_data['ops_per_sec'].append(results.get('ops_per_sec'))
            metrics_data['write_amp'].append(results.get('write_amp'))
            metrics_data['read_amp'].append(results.get('read_amp'))
    
    print(f"[DEBUG][plot_metrics_comparison] p99: {metrics_data['p99']}")
    print(f"[DEBUG][plot_metrics_comparison] ops_per_sec: {metrics_data['ops_per_sec']}")
    print(f"[DEBUG][plot_metrics_comparison] write_amp: {metrics_data['write_amp']}")
    print(f"[DEBUG][plot_metrics_comparison] read_amp: {metrics_data['read_amp']}")
    print(f"[DEBUG][plot_metrics_comparison] failed_indices: {failed_indices}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Metrics Comparison - {test_name}', fontsize=16)
    
    ax1 = axes[0, 0]
    p99_values = [x for x in metrics_data['p99'] if x is not None]
    if p99_values:
        colors = ['red' if i in failed_indices else 'blue' for i in range(len(p99_values))]
        ax1.plot(range(len(p99_values)), p99_values, 'b-', linewidth=2, alpha=0.7)
        for i, v in enumerate(p99_values):
            ax1.plot(i, v, 'o', color=colors[i], markersize=8)
        ax1.set_title('P99 Latency')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Latency (ms)')
        ax1.grid(True, alpha=0.3)
        for i, v in enumerate(p99_values):
            ax1.annotate(f'{v:.3f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax2 = axes[0, 1]
    ops_values = [x for x in metrics_data['ops_per_sec'] if x is not None]
    if ops_values:
        colors = ['red' if i in failed_indices else 'green' for i in range(len(ops_values))]
        ax2.plot(range(len(ops_values)), ops_values, 'g-', linewidth=2, alpha=0.7)
        for i, v in enumerate(ops_values):
            ax2.plot(i, v, 'o', color=colors[i], markersize=8)
        ax2.set_title('Throughput')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Ops/sec')
        ax2.grid(True, alpha=0.3)
        for i, v in enumerate(ops_values):
            ax2.annotate(f'{v:.0f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax3 = axes[1, 0]
    write_amp_values = [x for x in metrics_data['write_amp'] if x is not None]
    if write_amp_values:
        colors = ['red' if i in failed_indices else 'orange' for i in range(len(write_amp_values))]
        ax3.plot(range(len(write_amp_values)), write_amp_values, 'r-', linewidth=2, alpha=0.7)
        for i, v in enumerate(write_amp_values):
            ax3.plot(i, v, 'o', color=colors[i], markersize=8)
        ax3.set_title('Write Amplification')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Write Amplification')
        ax3.grid(True, alpha=0.3)
        for i, v in enumerate(write_amp_values):
            ax3.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax4 = axes[1, 1]
    read_amp_values = [x for x in metrics_data['read_amp'] if x is not None]
    if read_amp_values:
        colors = ['red' if i in failed_indices else 'purple' for i in range(len(read_amp_values))]
        ax4.plot(range(len(read_amp_values)), read_amp_values, 'm-', linewidth=2, alpha=0.7)
        for i, v in enumerate(read_amp_values):
            ax4.plot(i, v, 'o', color=colors[i], markersize=8)
        ax4.set_title('Read Amplification')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Read Amplification')
        ax4.grid(True, alpha=0.3)
        for i, v in enumerate(read_amp_values):
            ax4.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Successful Tuning'),
        Patch(facecolor='red', label='Failed Tuning (Best of 4 attempts)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_heatmap(options_files, output_dir, test_name):
    metrics_data = {
        'p99': [],
        'ops_per_sec': [],
        'write_amp': [],
        'read_amp': []
    }
    
    for config in options_files:
        if len(config) >= 5:
            _, results, _, _, is_failed_tuning = config
        else:
            _, results, _, _ = config
            
        if results:
            metrics_data['p99'].append(results.get('p99'))
            metrics_data['ops_per_sec'].append(results.get('ops_per_sec'))
            metrics_data['write_amp'].append(results.get('write_amp'))
            metrics_data['read_amp'].append(results.get('read_amp'))
    print(f"[DEBUG][plot_metrics_heatmap] p99: {metrics_data['p99']}")
    print(f"[DEBUG][plot_metrics_heatmap] ops_per_sec: {metrics_data['ops_per_sec']}")
    print(f"[DEBUG][plot_metrics_heatmap] write_amp: {metrics_data['write_amp']}")
    print(f"[DEBUG][plot_metrics_heatmap] read_amp: {metrics_data['read_amp']}")
    
    max_iterations = max(len(values) for values in metrics_data.values())
    
    for metric in metrics_data:
        while len(metrics_data[metric]) < max_iterations:
            metrics_data[metric].append(None)
    
    valid_data = {}
    for metric, values in metrics_data.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            valid_data[metric] = valid_values
    
    if not valid_data:
        return
    
    normalized_data = {}
    for metric, values in valid_data.items():
        if metric in ['p99', 'write_amp', 'read_amp']:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized_data[metric] = [1.0] * len(values)
            else:
                normalized_data[metric] = [(max_val - v) / (max_val - min_val) for v in values]
        else:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized_data[metric] = [1.0] * len(values)
            else:
                normalized_data[metric] = [(v - min_val) / (max_val - min_val) for v in values]
    
    metrics_list = list(normalized_data.keys())
    max_length = max(len(values) for values in normalized_data.values())
    
    heatmap_matrix = []
    for metric in metrics_list:
        row = normalized_data[metric].copy()
        while len(row) < max_length:
            row.append(0.0)
        heatmap_matrix.append(row)
    
    heatmap_matrix = np.array(heatmap_matrix)
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized Performance (1=Best, 0=Worst)', rotation=270, labelpad=20)
    
    plt.xlabel('Iteration')
    plt.ylabel('Metric')
    plt.title(f'Performance Heatmap - {test_name}')
    
    iterations = list(range(max_length))
    plt.xticks(iterations, [f'Iter {i}' for i in iterations])
    plt.yticks(range(len(metrics_list)), metrics_list)
    
    for i in range(len(metrics_list)):
        for j in range(max_length):
            if j < len(normalized_data[metrics_list[i]]):
                text = plt.text(j, i, f'{heatmap_matrix[i][j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
            else:
                text = plt.text(j, i, 'N/A',
                               ha="center", va="center", color="gray", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_radar(options_files, output_dir, test_name):
    metrics_data = {
        'p99': [],
        'ops_per_sec': [],
        'write_amp': [],
        'read_amp': []
    }
    
    for config in options_files:
        if len(config) >= 5:
            _, results, _, _, is_failed_tuning = config
        else:
            _, results, _, _ = config
            
        if results:
            metrics_data['p99'].append(results.get('p99'))
            metrics_data['ops_per_sec'].append(results.get('ops_per_sec'))
            metrics_data['write_amp'].append(results.get('write_amp'))
            metrics_data['read_amp'].append(results.get('read_amp'))
    print(f"[DEBUG][plot_metrics_radar] p99: {metrics_data['p99']}")
    print(f"[DEBUG][plot_metrics_radar] ops_per_sec: {metrics_data['ops_per_sec']}")
    print(f"[DEBUG][plot_metrics_radar] write_amp: {metrics_data['write_amp']}")
    print(f"[DEBUG][plot_metrics_radar] read_amp: {metrics_data['read_amp']}")
    
    max_iterations = max(len(values) for values in metrics_data.values())
    
    for metric in metrics_data:
        while len(metrics_data[metric]) < max_iterations:
            metrics_data[metric].append(None)
    
    valid_data = {}
    for metric, values in metrics_data.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            valid_data[metric] = valid_values
    
    if not valid_data:
        return
    
    normalized_data = {}
    for metric, values in valid_data.items():
        if metric in ['p99', 'write_amp', 'read_amp']:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized_data[metric] = [1.0] * len(values)
            else:
                normalized_data[metric] = [(max_val - v) / (max_val - min_val) for v in values]
        else:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized_data[metric] = [1.0] * len(values)
            else:
                normalized_data[metric] = [(v - min_val) / (max_val - min_val) for v in values]
    
    metrics_list = list(normalized_data.keys())
    num_metrics = len(metrics_list)
    
    if num_metrics == 0:
        return
    
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    min_iterations = min(len(values) for values in normalized_data.values())
    
    if min_iterations == 0:
        plt.close()
        return
    
    colors = plt.cm.Set3(np.linspace(0, 1, min_iterations))
    
    for i in range(min_iterations):
        values = []
        for metric in metrics_list:
            if i < len(normalized_data[metric]):
                values.append(normalized_data[metric][i])
            else:
                values.append(0.0)
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Iteration {i}', color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list)
    
    ax.set_title(f'Performance Radar Chart - {test_name}', size=16, y=1.08)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_trends(options_files, output_dir, test_name):
    metrics_data = {
        'p99': [],
        'ops_per_sec': [],
        'write_amp': [],
        'read_amp': []
    }
    
    for config in options_files:
        if len(config) >= 5:
            _, results, _, _, is_failed_tuning = config
        else:
            _, results, _, _ = config
            
        if results:
            metrics_data['p99'].append(results.get('p99'))
            metrics_data['ops_per_sec'].append(results.get('ops_per_sec'))
            metrics_data['write_amp'].append(results.get('write_amp'))
            metrics_data['read_amp'].append(results.get('read_amp'))
    print(f"[DEBUG][plot_metrics_trends] p99: {metrics_data['p99']}")
    print(f"[DEBUG][plot_metrics_trends] ops_per_sec: {metrics_data['ops_per_sec']}")
    print(f"[DEBUG][plot_metrics_trends] write_amp: {metrics_data['write_amp']}")
    print(f"[DEBUG][plot_metrics_trends] read_amp: {metrics_data['read_amp']}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Metrics Trends Analysis - {test_name}', fontsize=16)
    
    metrics_info = {
        'p99': {'title': 'P99 Latency Trend', 'ylabel': 'Latency (ms)', 'color': 'blue', 'lower_better': True},
        'ops_per_sec': {'title': 'Throughput Trend', 'ylabel': 'Ops/sec', 'color': 'green', 'lower_better': False},
        'write_amp': {'title': 'Write Amplification Trend', 'ylabel': 'Write Amp', 'color': 'red', 'lower_better': True},
        'read_amp': {'title': 'Read Amplification Trend', 'ylabel': 'Read Amp', 'color': 'purple', 'lower_better': True}
    }
    
    for idx, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[idx // 2, idx % 2]
        values = [v for v in metrics_data[metric] if v is not None]
        
        if values:
            iterations = list(range(len(values)))
            
            ax.plot(iterations, values, '-o', color=info["color"], linewidth=2, markersize=6)
            
            if len(values) > 1:
                z = np.polyfit(iterations, values, 1)
                p = np.poly1d(z)
                ax.plot(iterations, p(iterations), "--", color=info['color'], alpha=0.7, linewidth=1)
                
                slope = z[0]
                if info['lower_better']:
                    trend = "Improving" if slope < 0 else "Degrading" if slope > 0 else "Stable"
                else:
                    trend = "Improving" if slope > 0 else "Degrading" if slope < 0 else "Stable"
                
                ax.text(0.05, 0.95, f'Trend: {trend}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            for i, v in enumerate(values):
                if info['title'] in ['Write Amplification Trend', 'Read Amplification Trend']:
                    ax.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
                elif info['title'] == 'P99 Latency Trend':
                    ax.annotate(f'{v:.3f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
                else:
                    ax.annotate(f'{v:.2f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
            
            ax.set_title(info['title'])
            ax.set_xlabel('Iteration')
            ax.set_ylabel(info['ylabel'])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

