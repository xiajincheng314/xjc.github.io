from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import curve_fit
import io
import base64

# ==================== Flask 配置 ====================
app = Flask(__name__)
CORS(app)

def configure_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    zh_fonts = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei',
                'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'STHeiti']
    for path in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            name = font_manager.FontProperties(fname=path).get_name()
            if name in zh_fonts:
                font_manager.fontManager.addfont(path)
                plt.rcParams['font.sans-serif'] = [name]
                return
        except Exception:
            pass
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

configure_chinese_font()
plt.rcParams['figure.facecolor'] = '#060b19'
plt.rcParams['axes.facecolor'] = '#060b19'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'

active_figures = []
COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c',
          '#e67e22', '#2c3e50', '#e91e63', '#00bcd4']

# ==================== 候选函数库（μ–FN）====================
CANDIDATE_FUNCTIONS_FN = {
    "μ = a·FN + b": {
        "func": lambda x, a, b: a * x + b,
        "p0": [0.01, 0.2]
    },
    "μ = a·exp(-b·FN) + c": {
        "func": lambda x, a, b, c: a * np.exp(-b * x) + c,
        "p0": [0.5, 0.1, 0.2]
    },
    "μ = a·FN^b + c": {
        "func": lambda x, a, b, c: a * np.power(np.maximum(x, 1e-6), b) + c,
        "p0": [0.01, 0.5, 0.2]
    },
    "μ = a/(FN + b) + c": {
        "func": lambda x, a, b, c: a / (np.maximum(x, 1e-6) + b) + c,
        "p0": [0.1, 0.5, 0.2]
    },
    "μ = a - b·ln(FN)": {
        "func": lambda x, a, b: a - b * np.log(np.maximum(x, 1e-6)),
        "p0": [0.3, 0.02]
    },
    "μ = a + b·exp(-c·FN)": {
        "func": lambda x, a, b, c: a + b * np.exp(-c * x),
        "p0": [0.2, 0.1, 0.2]
    },
}

# ==================== 数据解析 ====================
def parse_three_column_file(content):
    vy_list, ay_list = [], []
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('质量') or line.startswith('y'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                vy_list.append(float(parts[1]))
                ay_list.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(vy_list), np.array(ay_list)

def fit_with_function(x, y, func, p0):
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
        return popt, r2
    except Exception:
        return None, -np.inf

def format_params(popt, precision=4):
    return '[' + ', '.join(f"{float(p):.{precision}f}" for p in popt) + ']'

# ==================== 单组预处理（改为筛选 |ay|∈(2,3) 取平均）====================
def preprocess_one_group(vy_raw, ay_raw, phi=25, alpha=45):
    """
    1. 剔除 ay >= 0 → |ay|
    2. 筛选 2 < |ay| < 3 的值取平均
    3. 用平均 |ay| 计算 μ
    返回 {'mu': 单个μ值, 'ay_avg': 平均|ay|, 'n_filtered': 样本数, 'n_total': 总负值点数}
    """
    mask = ay_raw < 0
    ay_neg = ay_raw[mask]
    n_total = len(ay_neg)
    if n_total < 5:
        return None

    ay_abs = np.abs(ay_neg)

    # 筛选 2 < |ay| < 3
    ay_filtered = ay_abs[(ay_abs > 2) & (ay_abs < 3)]
    n_filtered = len(ay_filtered)
    if n_filtered == 0:
        return None

    ay_avg = float(np.mean(ay_filtered))

    g = 9.8
    phi_rad, alpha_rad = math.radians(phi), math.radians(alpha)
    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)
    sin_alpha = math.sin(alpha_rad)

    mu = (g * sin_phi - ay_avg) * sin_alpha / (g * cos_phi)

    return {
        'mu': mu,
        'ay_avg': ay_avg,
        'n_filtered': n_filtered,
        'n_total': n_total
    }

def encode_image(buf):
    if buf is None:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

# ==================== 核心端点 ====================
@app.route('/pressure-friction/multi', methods=['POST'])
def pressure_friction_multi():
    try:
        data = request.json
        groups = data.get('groups', [])
        if not groups:
            return jsonify({"status": "error", "error": "至少需要一组实验数据。"}), 400

        # -------- 第一步：每组计算 (FN, μ) --------
        points = []       # [{name, FN, mu, ay_avg, n_filtered, color, ...}, ...]
        for i, grp in enumerate(groups):
            content = grp.get('content', '')
            name = grp.get('name', f'组{i+1}')
            params = grp.get('params', {})
            m = float(params.get('weight', 48))
            alpha = float(params.get('vAngle', 45))
            phi = float(params.get('rampAngle', 25))
            metal = params.get('metal', '铜')

            vy_raw, ay_raw = parse_three_column_file(content)
            pd = preprocess_one_group(vy_raw, ay_raw, phi=phi, alpha=alpha)
            if pd is None:
                continue

            g = 9.8
            phi_rad, alpha_rad = math.radians(phi), math.radians(alpha)
            FN = m / 1000 * g * math.cos(phi_rad) / math.sin(alpha_rad)

            points.append({
                "name": name,
                "FN": round(FN, 4),
                "mu": round(pd['mu'], 6),
                "ay_avg": round(pd['ay_avg'], 4),
                "n_filtered": pd['n_filtered'],
                "n_total": pd['n_total'],
                "color": COLORS[i % len(COLORS)],
                "metal": metal,
                "m": m, "alpha": alpha, "phi": phi
            })

        if len(points) < 2:
            return jsonify({"status": "error", "error": f"有效数据组仅 {len(points)} 组，至少需要 2 组进行拟合分析。"}), 400

        # -------- 第二步：提取 (FN, μ) 数组 --------
        fn_array = np.array([p['FN'] for p in points], dtype=float)
        mu_array = np.array([p['mu'] for p in points], dtype=float)

        # -------- 第三步：候选函数拟合，筛选最优 --------
        # 要求：R² ≥ 0.75 且 μ 随 FN 增大而减小（负相关）
        best_name = None
        best_popt = None
        best_r2 = -np.inf
        degraded = False

        for fname, fspec in CANDIDATE_FUNCTIONS_FN.items():
            popt, r2 = fit_with_function(fn_array, mu_array, fspec['func'], fspec['p0'])
            if popt is None or r2 < 0.75:
                continue
            # 验证负相关：在 FN 范围内，拟合函数值随 FN 增大而减小
            fn_test = np.linspace(fn_array.min(), fn_array.max(), 50)
            mu_test = fspec['func'](fn_test, *popt)
            if mu_test[0] <= mu_test[-1]:      # 非负相关
                continue
            if r2 > best_r2:
                best_r2 = r2
                best_name = fname
                best_popt = popt

        if best_name is None:
            # 降级：仅按 R² 选最优
            degraded = True
            for fname, fspec in CANDIDATE_FUNCTIONS_FN.items():
                popt, r2 = fit_with_function(fn_array, mu_array, fspec['func'], fspec['p0'])
                if popt is not None and r2 > best_r2:
                    best_r2 = r2
                    best_name = fname
                    best_popt = popt

        # -------- 第四步：绘图 --------
        fig, ax = plt.subplots(figsize=(14, 8))

        # 各组数据点（不同颜色）
        for p in points:
            ax.scatter(p['FN'], p['mu'],
                       c=p['color'], s=80, alpha=0.9,
                       edgecolors='white', linewidth=1.5, zorder=5,
                       label=f"{p['name']} (n={p['n_filtered']}/{p['n_total']}, μ={p['mu']:.4f})")
            # 标注
            ax.annotate(f"μ={p['mu']:.4f}",
                        (p['FN'], p['mu']),
                        textcoords="offset points",
                        xytext=(8, 8),
                        fontsize=8, color=p['color'])

        # 最优拟合曲线
        if best_popt is not None:
            fn_dense = np.linspace(fn_array.min() * 0.9, fn_array.max() * 1.1, 200)
            best_func = CANDIDATE_FUNCTIONS_FN[best_name]['func']
            mu_line = best_func(fn_dense, *best_popt)
            label = f'最优拟合：{best_name}  R²={best_r2:.4f}'
            if degraded:
                label += '  [⚠降级]'
            ax.plot(fn_dense, mu_line, 'w-', linewidth=3, label=label)

        ax.set_xlabel('法向压力 FN (N)', fontsize=14)
        ax.set_ylabel('摩擦系数 μ', fontsize=14)
        title = f'μ – FN 关系 (筛选 |ay|∈(2,3) 取平均)  [{best_name}]'
        if degraded:
            title += '  (降级)'
        ax.set_title(title, fontsize=16, fontweight='bold', color='#f59e0b')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
        fig.tight_layout()
        plt.show()

        # 弹出 Figure2
        try:
            plt.ion()
            fig.canvas.manager.set_window_title("Figure2 - μ–FN 关系 (|ay|∈(2,3))")
            fig.show()
            plt.draw()
            plt.pause(0.3)
            active_figures.append(fig)
        except Exception:
            pass

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)

        return jsonify({
            "status": "success",
            "image1": encode_image(buf),
            "best_function": best_name,
            "best_r2": round(float(best_r2), 4),
            "degraded": degraded,
            "points": [{
                "name": p['name'], "FN": p['FN'], "mu": p['mu'],
                "ay_avg": p['ay_avg'], "n_filtered": p['n_filtered'],
                "metal": p['metal'], "m": p['m'], "alpha": p['alpha'], "phi": p['phi']
            } for p in points],
            "message": f"成功分析 {len(points)} 组数据。最优拟合：{best_name}，R²={best_r2:.4f}"
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5003, debug=True, use_reloader=False)