from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
# 新增Excel处理依赖
import pandas as pd
import io

app = Flask(__name__)
CORS(app)  # 允许网页跨域调用

# 你的 DeepSeek API 信息
API_KEY = "sk-4f1e8d4ea40f4d1bb3a25c7fad21f6d1"
BASE_URL = "https://api.deepseek.com"

# 新增：解析Excel文件内容
def parse_excel_content(file_content):
    try:
        # 将base64编码的文件内容转为二进制流
        import base64
        file_data = base64.b64decode(file_content)
        # 读取Excel文件（支持.xlsx和.xls）
        df = pd.read_excel(io.BytesIO(file_data))
        # 转换为易读的文本格式
        excel_text = "Excel文件解析结果：\n"
        excel_text += f"工作表名称：{df.columns.tolist()}\n"
        excel_text += f"数据行数：{len(df)}\n"
        excel_text += "前10行数据：\n"
        excel_text += df.head(10).to_string(index=False)
        return excel_text
    except Exception as e:
        return f"Excel文件解析失败：{str(e)}"

# 核心接口：网页调用这里
@app.route('/ai-analyze', methods=['POST'])
def analyze():
    data = request.json

    # 接收前端参数
    friction_type = data.get("friction_type")
    metal_type = data.get("metal_type")
    slider_weight = data.get("slider_weight")
    v_angle = data.get("v_angle")
    ramp_angle = data.get("ramp_angle")
    prompt = data.get("prompt")
    # 新增：接收上传的文件数据
    uploaded_file_name = data.get("uploaded_file_name")
    uploaded_file_content = data.get("uploaded_file_content")
    uploaded_file_type = data.get("uploaded_file_type")  # 新增：接收文件类型

    # 构造文件信息文本
    file_info = ""
    if uploaded_file_name and uploaded_file_content:
        # 判断文件类型并解析
        if uploaded_file_type in ["xlsx", "xls"]:
            # 解析Excel文件
            file_info = parse_excel_content(uploaded_file_content)
        else:
            # 普通文本文件直接使用
            file_info = f"""
上传的实验数据文件：{uploaded_file_name}
文件内容：
{uploaded_file_content}
"""

    # 调用 DeepSeek AI
    openai.api_key = API_KEY
    openai.api_base = BASE_URL

    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"""
你是冰与金属摩擦实验专业分析师。

实验参数：
摩擦类型：{friction_type}
金属类型：{metal_type}
滑块重量：{slider_weight} g
V型槽张角：{v_angle} °
斜面倾角：{ramp_angle} °

{file_info}

用户需求：{prompt}

请结合实验参数和上传的文件数据，给出专业、简洁、科学的分析结果。
"""
        }]
    )

    result = response.choices[0].message['content']
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)