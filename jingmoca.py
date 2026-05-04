from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import io
import os
import base64
import math
from io import StringIO

app = Flask(__name__)
CORS(app)

# 中文字体配置：优先使用系统中的中文字体，避免 SimHei 不存在时汉字无法显示
def configure_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    zh_fonts = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'STHeiti']
    for path in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            name = font_manager.FontProperties(fname=path).get_name()
            if name in zh_fonts:
                font_manager.fontManager.addfont(path)
                plt.rcParams['font.sans-serif'] = [path]
                return
        except Exception:
            continue
    plt.rcParams['font.sans-serif'] = ['SimHei']

configure_chinese_font()
plt.rcParams['figure.facecolor'] = '#060b19'
plt.rcParams['axes.facecolor'] = '#060b19'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'

active_figures = []


def read_experiment_data(file_path=None, file_content=None, file_name=None):
    """读取上传或本地实验数据，返回 DataFrame"""
    if file_content is not None and file_name is not None:
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ['.csv', '.txt']:
            try:
                df = pd.read_csv(StringIO(file_content), encoding='utf-8', engine='python', header=None)
                if df.shape[1] == 1:
                    df.columns = ['斜面倾角(°)']
                    df['时间(s)'] = df.index
                elif df.shape[1] >= 2:
                    df.columns = ['时间(s)', '斜面倾角(°)'][:df.shape[1]]
                return df
            except Exception as e:
                raise ValueError(f"无法解析上传的文本文件：{e}")
        elif ext in ['.xls', '.xlsx']:
            try:
                df = pd.read_excel(io.BytesIO(file_content.encode('utf-8')))
                return df
            except Exception as e:
                raise ValueError(f"无法解析上传的 Excel 文件：{e}")
        else:
            raise ValueError("不支持的文件类型，请上传 CSV 或 TXT 文件")
    elif file_path is not None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, sheet_name="工作薄1")
        elif ext in ['.csv', '.txt']:
            df = pd.read_csv(file_path, encoding='utf-8', engine='python', header=None)
            if df.shape[1] == 1:
                df.columns = ['斜面倾角(°)']
                df['时间(s)'] = df.index
            elif df.shape[1] >= 2:
                df.columns = ['时间(s)', '斜面倾角(°)'][:df.shape[1]]
        else:
            raise ValueError("不支持的文件类型，请上传 CSV 或 TXT 文件")
    else:
        raise ValueError("未提供实验数据文件路径或文件内容")
    return df


def process_static_friction(file_path=None, alpha=45, m=48, file_content=None, file_name=None):
    """
    处理静摩擦数据，只生成「摩擦系数随速度变化」图像
    返回：buf1 (图1), None (占位), result 字典
    """
    df = read_experiment_data(file_path=file_path, file_content=file_content, file_name=file_name)

    if "斜面倾角(°)" not in df.columns or "时间(s)" not in df.columns:
        raise ValueError("Excel文件缺少必要列：斜面倾角(°) 或 时间(s)")

    # 剔除最大/最小值，计算平均倾角 φ2
    phi_data = df["斜面倾角(°)"].dropna().sort_values().reset_index(drop=True)
    if len(phi_data) <= 2:
        raise ValueError("斜面倾角数据量过少，无法剔除最大/最小值")
    phi_filtered = phi_data.iloc[1:-1]
    phi2 = phi_filtered.mean()

    # 摩擦系数 μ = tan(φ2) × sin(α)
    alpha_rad = math.radians(alpha)
    phi2_rad = math.radians(phi2)
    mu = math.tan(phi2_rad) * math.sin(alpha_rad)

    # ========== 仅生成图1：摩擦系数随速度变化（静摩擦时速度=0，μ为定值） ==========
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.axhline(mu, color='#22d3ee', linewidth=2, label=f'μ = {mu:.4f}')
    ax1.set_title("摩擦系数随速度变化", fontsize=14, fontweight='bold', color='#22d3ee')
    ax1.set_xlabel("速度 (m/s)")
    ax1.set_ylabel("摩擦系数 μ")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(mu * 0.9, mu * 1.1)
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["0"])
    ax1.legend()

    plt.tight_layout()
    plt.show()

    # 弹出窗口
    try:
        plt.ion()
        fig1.canvas.manager.set_window_title("Figure1 - 摩擦系数随速度变化")
        fig1.show()
        plt.draw()
        plt.pause(0.2)
        active_figures.append(fig1)
    except Exception:
        pass

    # 保存图像到 BytesIO
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
    buf1.seek(0)

    # 计算结果
    result = {
        "phi2": round(phi2, 2),
        "mu": round(mu, 4),
        "alpha": alpha,
        "slider_weight_kg": m / 1000,
        "max_sin_phi": round(np.sin(np.radians(phi2)), 4)
    }

    return buf1, None, result


def encode_image(buf):
    if buf is None:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# ================= API 端点 =================

@app.route('/static-friction/calc', methods=['POST'])
def static_friction_calc():
    try:
        data = request.json
        file_path = data.get("file_path", "工作薄1.xlsx")
        alpha = float(data.get("v_angle", 45))
        m = float(data.get("slider_weight", 48))

        if not os.path.exists(file_path):
            return jsonify({"error": f"文件不存在：{file_path}"}), 400

        buf1, _, result = process_static_friction(file_path, alpha, m)
        return jsonify({
            "status": "success",
            "result": result,
            "message": "静摩擦数据处理完成"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/static-friction/process', methods=['POST'])
def process_static_friction_upload():
    try:
        data = request.json
        file_content = data.get('uploaded_file_content')
        file_name = data.get('uploaded_file_name')
        alpha = float(data.get('v_angle', 45))
        m = float(data.get('slider_weight', 48))

        if not file_content or not file_name:
            return jsonify({"status": "error", "error": "请先上传实验数据文件。"}), 400

        buf1, _, result = process_static_friction(
            file_content=file_content,
            file_name=file_name,
            alpha=alpha,
            m=m
        )

        return jsonify({
            "status": "success",
            "image1": encode_image(buf1),      # 只返回一张图
            "result": result,
            "message": "实验图像处理完成（仅生成摩擦系数-速度图像）"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/static-friction/image/coefficient', methods=['POST'])
def get_coefficient_image():
    try:
        data = request.json
        file_path = data.get("file_path", "工作薄1.xlsx")
        alpha = float(data.get("v_angle", 45))
        m = float(data.get("slider_weight", 48))
        buf1, _, _ = process_static_friction(file_path, alpha, m)
        return send_file(buf1, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/static-friction/image/force', methods=['POST'])
def get_force_image():
    # 该图像已废弃，返回提示
    return jsonify({"status": "removed", "message": "摩擦系数-压力图像已移除"}), 410


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)