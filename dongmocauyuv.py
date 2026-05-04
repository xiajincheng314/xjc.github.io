"""
dongmocauyuv3.py — Flask 后端（端口 5004）+ 共同函数筛选
功能：接收前端上传的多组三列格式 (y, vy, ay) 实验数据，
      剔除 ay≥0，|ay| → μ，剔除 μ<0，剔除 μ>0.2，
      保序回归 → SG平滑 → 生成控制点，
      测试 10 种共同函数 → 选出三组 R² 均 ≥ 0.75 的最佳共同函数，
      若均不满足则回退独立最优拟合，
      生成 μ–|vy| 对比图像，通过 API 返回 Base64 供前端展示。
参数：φ = 25°，α = 45°
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit
import math
import io
import base64

# ==================== Flask 配置 ====================
app = Flask(__name__)
CORS(app)


# ==================== 中文字体配置 ====================
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
            continue
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


# ==================== 扩充后的候选共同函数库（10 种）====================
CANDIDATE_FUNCTIONS = {
    # --- 原有 6 种 ---
    "μ = a·exp(-b·|vy|) + c": {
        "func": lambda x, a, b, c: a * np.exp(-b * x) + c,
        "p0": [0.2, 100, 0.05], "num_params": 3
    },
    "μ = a·|vy|^(-b) + c": {
        "func": lambda x, a, b, c: a * np.power(np.maximum(x, 1e-9), -b) + c,
        "p0": [0.005, 0.3, 0.05], "num_params": 3
    },
    "μ = a/(|vy|+b) + c": {
        "func": lambda x, a, b, c: a / (x + b) + c,
        "p0": [0.002, 0.01, 0.05], "num_params": 3
    },
    "μ = a - b·ln(|vy|)": {
        "func": lambda x, a, b: a - b * np.log(np.maximum(x, 1e-9)),
        "p0": [0.15, 0.01], "num_params": 2
    },
    "μ = a + b·exp(-c·|vy|)": {
        "func": lambda x, a, b, c: a + b * np.exp(-c * x),
        "p0": [0.15, 0.05, 100], "num_params": 3
    },
    "μ = a·|vy| + b": {
        "func": lambda x, a, b: a * x + b,
        "p0": [-1.5, 0.2], "num_params": 2
    },
    # --- 新增 4 种 ---
    "μ = (a·|vy| + b) / (|vy| + c)": {
        "func": lambda x, a, b, c: (a * x + b) / (x + c),
        "p0": [0.05, 0.01, 0.02], "num_params": 3
    },
    "μ = a - b·√|vy|": {
        "func": lambda x, a, b: a - b * np.sqrt(np.maximum(x, 1e-9)),
        "p0": [0.2, 0.8], "num_params": 2
    },
    "μ = a·exp(-b·|vy|^c) + d": {
        "func": lambda x, a, b, c, d: a * np.exp(-b * np.power(np.maximum(x, 1e-9), c)) + d,
        "p0": [0.2, 50, 0.5, 0.05], "num_params": 4
    },
    "μ = a/(|vy|^b + c) + d": {
        "func": lambda x, a, b, c, d: a / (np.power(np.maximum(x, 1e-9), b) + c) + d,
        "p0": [0.01, 0.5, 0.1, 0.05], "num_params": 4
    },
}


# ==================== 数据解析 ====================
def parse_from_string(content):
    vy_list, ay_list = [], []
    lines = content.split('\n')
    for line in lines:
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


# ==================== 工具函数 ====================
def format_params(popt, precision=4):
    return '[' + ', '.join(f"{float(p):.{precision}f}" for p in popt) + ']'


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


# ==================== 单组数据预处理（μ 严格限定在 [0, 0.2]）====================
def preprocess_one_group(vy_raw, ay_raw, phi=25, alpha=45):
    mask = ay_raw < 0
    vy_neg = vy_raw[mask]
    ay_neg = ay_raw[mask]
    if len(ay_neg) < 5:
        return None

    ay_abs = np.abs(ay_neg)
    vy_abs = np.abs(vy_neg)

    g = 9.8
    phi_rad, alpha_rad = math.radians(phi), math.radians(alpha)
    sin_phi, cos_phi = math.sin(phi_rad), math.cos(phi_rad)
    sin_alpha = math.sin(alpha_rad)

    mu_values = (g * sin_phi - ay_abs) * sin_alpha / (g * cos_phi)

    # 第一步：剔除 μ < 0
    mu_pos = mu_values >= 0
    vy_abs, ay_abs, mu_values = vy_abs[mu_pos], ay_abs[mu_pos], mu_values[mu_pos]
    if len(mu_values) < 5:
        return None

    # 第二步：剔除 μ > 0.2
    range_mask = mu_values <= 0.2
    vy_abs, ay_abs, mu_values = vy_abs[range_mask], ay_abs[range_mask], mu_values[range_mask]
    if len(mu_values) < 5:
        return None

    # 第三步：SG + MAD 去野
    if len(mu_values) >= 11:
        w = min(11, len(mu_values) // 2 * 2 + 1)
        if w % 2 == 0:
            w -= 1
        mu_sg = savgol_filter(mu_values, w, min(3, w - 1))
        inliers = np.abs(mu_values - mu_sg) < 5 * np.median(np.abs(mu_values - mu_sg))
    else:
        inliers = np.ones(len(mu_values), dtype=bool)

    vy_in, mu_in = vy_abs[inliers], mu_values[inliers]

    if len(mu_in) < 5:
        return None

    # 排序
    sort_idx = np.argsort(vy_in)
    vy_sorted, mu_sorted = vy_in[sort_idx], mu_in[sort_idx]

    # 去重
    unique_vy, idx = np.unique(vy_sorted, return_inverse=True)
    if len(unique_vy) < len(vy_sorted):
        mu_unique = np.array([np.mean(mu_sorted[idx == i]) for i in range(len(unique_vy))])
    else:
        unique_vy, mu_unique = vy_sorted, mu_sorted

    if len(unique_vy) < 3:
        return None

    # 保序回归（降序）
    try:
        ir = IsotonicRegression(increasing=False)
        mu_iso = ir.fit_transform(unique_vy, mu_unique)
    except Exception:
        mu_iso = mu_unique

    # SG 平滑
    if len(mu_iso) >= 7:
        mu_iso_smooth = savgol_filter(mu_iso, min(7, len(mu_iso) // 2 * 2 + 1), 2)
    else:
        mu_iso_smooth = mu_iso

    return {
        'vy_in': vy_in, 'mu_in': mu_in,
        'unique_vy': unique_vy, 'mu_iso_smooth': mu_iso_smooth,
    }


# ==================== 独立拟合引擎（回退用）====================
def fit_models_to_smoothed(x_smooth, y_smooth, x_dense):
    """遍历全部 10 种候选函数，选独立最优"""
    best_r2 = -np.inf
    best_result = None

    for name, fspec in CANDIDATE_FUNCTIONS.items():
        func = fspec['func']
        p0 = fspec['p0']
        try:
            popt, _ = curve_fit(func, x_smooth, y_smooth, p0=p0, maxfev=20000)
            y_pred = func(x_smooth, *popt)
            ss_res = np.sum((y_smooth - y_pred) ** 2)
            ss_tot = np.sum((y_smooth - np.mean(y_smooth)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'name': name, 'popt': popt, 'r2': r2, 'func': func,
                    'y_fit_dense': func(x_dense, *popt)
                }
        except Exception:
            continue

    return best_result


def encode_image(buf):
    if buf is None:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# ==================== 核心处理函数 ====================
def process_sliding_friction_multi(groups, phi=25, alpha=45):
    # ---------- 1. 预处理所有组 ----------
    pre_data = {}
    for i, g in enumerate(groups):
        name = g.get('name', f'数据组{i+1}')
        content = g.get('content', '')
        if not content:
            continue
        vy_raw, ay_raw = parse_from_string(content)
        pd = preprocess_one_group(vy_raw, ay_raw, phi=phi, alpha=alpha)
        if pd is not None:
            pre_data[name] = pd
            print(f"【{name}】预处理完成，控制点 {len(pd['unique_vy'])} 个")

    if len(pre_data) == 0:
        raise ValueError("所有数据组均不满足分析条件（μ∈[0,0.2] 内数据不足）")

    # ---------- 2. 共同函数筛选 ----------
    print("\n========== 共同函数筛选 ==========")
    best_common_name = None
    best_min_r2 = -1
    common_summary = []

    for fname, fspec in CANDIDATE_FUNCTIONS.items():
        func = fspec['func']
        p0 = fspec['p0']
        r2_list = {}
        for name, pd in pre_data.items():
            _, r2 = fit_with_function(pd['unique_vy'], pd['mu_iso_smooth'], func, p0)
            r2_list[name] = r2

        min_r2 = min(r2_list.values()) if r2_list else -1
        common_summary.append({
            "function": fname, "r2_by_group": r2_list, "min_r2": round(min_r2, 4)
        })

        if min_r2 >= 0.75:
            print(f"  ✅ {fname}: 最小 R² = {min_r2:.4f}")
            if min_r2 > best_min_r2:
                best_min_r2 = min_r2
                best_common_name = fname
        else:
            print(f"  ❌ {fname}: 最小 R² = {min_r2:.4f}")

    # ---------- 3. 选择函数并绘图 ----------
    if best_common_name is None:
        print("\n⚠ 无共同函数使所有组 R²≥0.75，回退独立最优拟合模式")
        use_common = False
    else:
        print(f"\n🎯 选定共同函数：{best_common_name}（最小 R² = {best_min_r2:.4f}）")
        use_common = True

    fig, ax = plt.subplots(figsize=(14, 8))
    all_fit_results = {}
    summary = []

    for i, g in enumerate(groups):
        name = g.get('name', f'数据组{i+1}')
        if name not in pre_data:
            summary.append({"name": name, "status": "skipped", "reason": "有效数据不足"})
            continue

        pd = pre_data[name]
        color = COLORS[i % len(COLORS)]
        vy_dense = np.linspace(pd['unique_vy'][0], pd['unique_vy'][-1], 300)

        if use_common:
            chosen_spec = CANDIDATE_FUNCTIONS[best_common_name]
            popt, r2 = fit_with_function(pd['unique_vy'], pd['mu_iso_smooth'],
                                         chosen_spec['func'], chosen_spec['p0'])
            if popt is None:
                r2 = -1
                mu_fit = np.zeros_like(vy_dense)
            else:
                mu_fit = chosen_spec['func'](vy_dense, *popt)
            model_name = best_common_name
        else:
            best = fit_models_to_smoothed(pd['unique_vy'], pd['mu_iso_smooth'], vy_dense)
            if best is None:
                summary.append({"name": name, "status": "skipped", "reason": "拟合失败"})
                continue
            popt = best['popt']
            r2 = best['r2']
            mu_fit = best['y_fit_dense']
            model_name = best['name']

        ax.scatter(pd['vy_in'], pd['mu_in'],
                   c=color, s=28, alpha=0.65,
                   edgecolors='white', linewidth=0.4,
                   label=f"{name} (n={len(pd['vy_in'])}, R²={r2:.3f})")
        ax.plot(vy_dense, mu_fit, color=color, linewidth=2.8,
                label=f"  └ {model_name}  参数: {format_params(popt)}")

        all_fit_results[name] = {'r2': r2, 'popt': [float(p) for p in popt] if popt is not None else []}
        summary.append({
            "name": name, "status": "success",
            "n_points": len(pd['vy_in']),
            "model": model_name, "r2": round(r2, 4),
            "params": [round(float(p), 6) for p in popt] if popt is not None else [],
            "common_function": use_common
        })

    if not all_fit_results:
        raise ValueError("所有数据组拟合失败")

    # 图表装饰
    title_suffix = f"共同函数：{best_common_name}" if use_common else "独立最优拟合（无共同函数达标）"
    ax.set_xlabel('速度 |vy| (m/s)', fontsize=14)
    ax.set_ylabel('摩擦系数 μ', fontsize=14)
    ax.set_title(f'多组实验 μ–|vy| 负相关拟合  (φ={phi}°, α={alpha}°)  [μ∈[0,0.2]]  [{title_suffix}]',
                 fontsize=15, fontweight='bold', color='#22d3ee')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.88,
              ncol=1, columnspacing=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.03, y=0.03)

    plt.tight_layout()
    plt.show()

    # 弹出 Figure1 窗口
    try:
        plt.ion()
        fig.canvas.manager.set_window_title(f"Figure1 - μ–|vy| {title_suffix}")
        fig.show()
        plt.draw()
        plt.pause(0.2)
        active_figures.append(fig)
    except Exception:
        pass

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)

    return buf, summary, best_common_name if use_common else None, common_summary


# ==================== API 端点 ====================
@app.route('/api/analyze', methods=['POST'])
def analyze_multi_group():
    try:
        data = request.json
        groups = data.get('groups', [])

        if not groups or len(groups) == 0:
            return jsonify({"status": "error", "error": "请至少上传一组实验数据。"}), 400

        phi = float(data.get('ramp_angle', 25))
        alpha = float(data.get('v_angle', 45))

        buf, summary, common_name, common_summary = process_sliding_friction_multi(
            groups, phi=phi, alpha=alpha
        )

        success_count = sum(1 for s in summary if s.get('status') == 'success')
        return jsonify({
            "status": "success",
            "image1": encode_image(buf),
            "summary": summary,
            "common_function": common_name,
            "common_summary": common_summary,
            "message": (
                f"成功分析 {success_count}/{len(groups)} 组数据。"
                + (f"共同函数：{common_name}" if common_name else "未找到共同函数，使用独立最优拟合")
            )
        })

    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ==================== 启动 ====================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5004, debug=True, use_reloader=False)