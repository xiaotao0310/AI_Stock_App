# ==================== 强制禁用代理 ====================
import os
for var in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if var in os.environ:
        try:
            del os.environ[var]
        except:
            pass

# ==================== 导入库 ====================
import html
import streamlit as st
import time
import re
import json
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from openai import OpenAI
import baostock as bs
import akshare as ak

# ==================== 页面配置 ====================
st.set_page_config(page_title="金韬短线实战系统", layout="wide")

# ==================== 安全读取 API Key ====================
try:
    MY_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    st.error("❌ 未找到 API Key，请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(api_key=MY_API_KEY, base_url="https://api.deepseek.com")


# ==================== baostock 会话管理 ====================

class BaostockSession:
    """统一管理 baostock 登录/登出，避免重复连接"""
    def __enter__(self):
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        bs.logout()


def _bs_code(code: str) -> str:
    """将6位股票代码转换为 baostock 格式（仅支持沪深，不含北交所）"""
    if code.startswith(('600', '601', '603', '605', '688')):
        return f"sh.{code}"
    return f"sz.{code}"


# ==================== 辅助函数 ====================

def get_stock_info_baostock(code: str, session: BaostockSession):
    """返回股票名称，失败返回 None"""
    try:
        rs = bs.query_stock_basic(code=_bs_code(code))
        if rs.error_code != '0':
            return None
        rows = []
        while rs.next():
            row = rs.get_row_data()
            if row:
                rows.append(row)
        return rows[0][1] if rows else None
    except Exception as e:
        print(f"baostock 获取名称失败: {e}")
        return None


def compute_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    dif = exp1 - exp2
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif, dea, macd


def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    low_list = low.rolling(n).min()
    high_list = high.rolling(n).max()
    rsv = (close - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(span=m1, adjust=False).mean()
    d = k.ewm(span=m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def get_technical_baostock(code: str, session: BaostockSession, days: int = 60) -> dict:
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            code=_bs_code(code),
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        if rs.error_code != '0':
            return {}
        data = []
        while rs.next():
            row = rs.get_row_data()
            if row:
                data.append(row)
        if not data or len(data) < 20:
            return {}
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df = df.sort_values('date')
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1) * 100).fillna(0)

        dif, dea, macd = compute_macd(df['close'])
        df['DIF'] = dif; df['DEA'] = dea; df['MACD'] = macd

        k, d, j = compute_kdj(df['high'], df['low'], df['close'])
        df['K'] = k; df['D'] = d; df['J'] = j

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        vol_change = ((latest['volume'] - prev['volume']) / prev['volume'] * 100 if prev['volume'] > 0 else 0)
        last5 = df.tail(5)
        avg_volume_5 = last5['volume'].mean()
        avg_amount_5 = avg_volume_5 * latest['close'] / 10000
        avg_amplitude_5 = last5['amplitude'].mean()
        today_amount = latest['volume'] * latest['close'] / 10000
        today_amplitude = latest['amplitude']
        avg_vol_5 = last5['volume'].mean()
        vol_ratio = latest['volume'] / avg_vol_5 if avg_vol_5 > 0 else 1.0

        return {
            'latest_price': latest['close'],
            'ma5': latest['MA5'],
            'ma20': latest['MA20'],
            'volume_change': vol_change,
            'avg_amount_5_yi': avg_amount_5 / 1e4,
            'avg_amplitude_5': avg_amplitude_5,
            'today_amplitude': today_amplitude,
            'today_amount_yi': today_amount / 1e4,
            'vol_ratio_vs_5': vol_ratio,
            'dif': latest['DIF'], 'dea': latest['DEA'], 'macd': latest['MACD'],
            'k': latest['K'], 'd': latest['D'], 'j': latest['J'],
        }
    except Exception as e:
        print(f"技术指标失败: {e}")
        return {}


def get_stock_news(code: str) -> list:
    """东方财富个股专属新闻（仅沪深）"""
    market = 1 if code.startswith(('600', '601', '603', '605', '688')) else 0
    url = f"https://np-listapi.eastmoney.com/comm/wap/getListInfo?client=wap&type=1&mTypeAndCode={market}.{code}&pageSize=10&pageIndex=1&_={int(time.time() * 1000)}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://wap.eastmoney.com/"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        items = data.get('data', {}).get('list', [])
        results = []
        for item in items:
            title = item.get('Art_Title', '').strip()
            media = item.get('Art_MediaName', '').strip()
            show_time = item.get('Art_ShowTime', '').strip()
            art_url = item.get('Art_Url', '').strip()
            time_part = show_time[11:16] if len(show_time) >= 16 else ''
            if title:
                label = f"[{media} {time_part}] {title}" if media else title
                results.append({'label': label, 'title': title, 'url': art_url})
        return results[:10]
    except Exception:
        return []


# ==================== 数据源函数（含换手率，仅支持沪深） ====================

def get_price_tencent(code: str) -> dict:
    """腾讯财经实时行情（含换手率）"""
    prefix = "sh" if code.startswith(('600', '601', '603', '605', '688')) else "sz"
    url = f"https://qt.gtimg.cn/q={prefix}{code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        text = resp.text
        if '~' not in text:
            return None
        parts = text.split('="')[1].split('"')[0].split('~')
        if len(parts) < 40:
            return None
        latest = float(parts[3])
        yesterday_close = float(parts[4])
        open_price = float(parts[5])
        volume = int(float(parts[6])) * 100
        high = float(parts[33])
        low = float(parts[34])
        amount = float(parts[37]) * 1e4
        change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
        # 换手率（腾讯接口索引38是换手率，百分比值，如 2.35）
        turnover = float(parts[38]) if len(parts) > 38 and parts[38] else None

        return {
            'price': latest,
            'change_pct': round(change_pct, 2) if change_pct is not None else None,
            'open': open_price, 'high': high, 'low': low,
            'yesterday_close': yesterday_close,
            'volume': volume, 'amount': amount, 'amount_yi': amount / 1e8,
            'turnover': turnover,  # 换手率（%）
        }
    except Exception:
        return None


def get_price_sina(code: str) -> dict:
    """新浪财经（备用，无换手率，保持兼容）"""
    prefix = "sh" if code.startswith(('600', '601', '603', '605', '688')) else "sz"
    url = f"https://hq.sinajs.cn/list={prefix}{code}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        text = resp.text
        if '="' in text:
            parts = text.split('="')[1].split('"')[0].split(',')
            if len(parts) >= 10:
                latest = float(parts[3])
                yesterday_close = float(parts[2])
                change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
                return {
                    'price': latest,
                    'change_pct': round(change_pct, 2) if change_pct is not None else None,
                    'open': float(parts[1]), 'high': float(parts[4]), 'low': float(parts[5]),
                    'yesterday_close': yesterday_close,
                    'volume': int(float(parts[6])) if parts[6] else 0,
                    'amount': float(parts[7]) if parts[7] else 0.0,
                    'amount_yi': float(parts[7]) / 1e8 if parts[7] else 0.0,
                    'turnover': None,  # 新浪此接口无换手率
                }
    except Exception:
        pass
    return None


def get_price_akshare_single(code: str) -> dict:
    """AkShare 买卖盘（无换手率）"""
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        info = dict(zip(df['item'], df['value']))
        latest = float(info.get('最新', 0))
        yesterday_close = float(info.get('昨收', 0))
        if not latest:
            return None
        change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
        return {
            'price': latest,
            'change_pct': round(change_pct, 2) if change_pct is not None else None,
            'open': float(info.get('今开', 0)) or None,
            'high': float(info.get('最高', 0)) or None,
            'low': float(info.get('最低', 0)) or None,
            'yesterday_close': yesterday_close or None,
            'volume': int(float(info.get('总手', 0))) or None,
            'amount': None, 'amount_yi': None,
            'turnover': None,
        }
    except Exception:
        return None


def get_price_eastmoney_direct(code: str) -> dict:
    """东方财富直连（含换手率 f168，仅沪深）"""
    market = 1 if code.startswith(('600', '601', '603', '605', '688')) else 0
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f60,f170,f168"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = resp.json().get('data')
        if not data or not data.get('f43'):
            return None
        return {
            'price': data['f43'] / 100,
            'change_pct': data.get('f170'),
            'open': data['f46'] / 100,
            'high': data['f44'] / 100,
            'low': data['f45'] / 100,
            'yesterday_close': data['f60'] / 100,
            'volume': data['f47'] * 100,
            'amount': data['f48'],
            'amount_yi': data['f48'] / 1e8,
            'turnover': data.get('f168'),  # 换手率（%）——缩放规则尚未权威确认，建议实测核对
        }
    except Exception:
        return None


def get_realtime_price(code: str) -> dict:
    """多源容错（腾讯优先，因为含换手率）—— 只负责4个外部HTTP源，不碰baostock"""
    result = get_price_tencent(code)
    if result and result.get('price') is not None:
        return result

    result = get_price_sina(code)
    if result and result.get('price') is not None:
        return result

    result = get_price_akshare_single(code)
    if result and result.get('price') is not None:
        return result

    result = get_price_eastmoney_direct(code)
    if result and result.get('price') is not None:
        return result

    return {}  # 4个外部源都失败，交给调用方决定是否用baostock兜底


def get_baostock_fallback_price(code: str, session: BaostockSession) -> dict:
    """baostock 兜底 —— 必须传入外层已登录的 session，绝不自己开新连接（避免嵌套登出把外层连接搞断）"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            code=_bs_code(code),
            fields="date,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        data = []
        while rs.next():
            row = rs.get_row_data()
            if row:
                data.append(row)
        if data:
            last = data[-1]
            return {
                'price': float(last[1]),
                'change_pct': None,
                'open': None, 'high': None, 'low': None,
                'yesterday_close': None,
                'volume': int(float(last[2])) if last[2] else None,
                'amount': None, 'amount_yi': None,
                'turnover': None,
            }
    except Exception as e:
        print(f"baostock 兜底失败: {e}")
    return {}


def render_news_html(news_items: list) -> str:
    if not news_items:
        return """<div style="max-height:340px; overflow-y:auto; background:#1e1e2f; padding:12px; border-radius:12px;"><p style="color:#ffaa00; text-align:center;">暂无相关资讯</p></div>"""
    rows = []
    for item in news_items:
        label = html.escape(item['label'])
        url = html.escape(item['url'])
        if url:
            row = f'<p style="border-bottom:1px solid #2a2a3f; padding:7px 0; margin:0; line-height:1.5;">• <a href="{url}" target="_blank" style="color:#00ff9d; text-decoration:none;">{label}</a></p>'
        else:
            row = f'<p style="color:#00ff9d; border-bottom:1px solid #2a2a3f; padding:7px 0; margin:0; line-height:1.5;">• {label}</p>'
        rows.append(row)
    return f"""<div style="max-height:340px; overflow-y:auto; background:#1e1e2f; padding:14px; border-radius:12px; font-family:monospace; font-size:13px;">{''.join(rows)}</div>"""


# ==================== 界面 ====================
st.title("🚀 金韬短线实战系统2.0")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 输入个股")
    stock_code = st.text_input("请输入股票代码（6位数字）", value="000001", help="例如: 000001, 600519, 603318")
    analyze_btn = st.button("开始 AI 智能决策分析", type="primary", use_container_width=True)
    st.info("""
    **实战提示：**
    本系统已接入金韬独家策略库：
    - 金韬团队专属
    - 版权所有，侵权必究
    - 本系统仅用于个人学习与研究，不构成投资建议
    """)


if analyze_btn:
    if not stock_code.isdigit() or len(stock_code) != 6:
        st.error("请输入正确的6位股票代码（例如 000001）")
        st.stop()

    with col2:
        progress_area = st.empty()
        progress_bar = st.progress(0)

    try:
        progress_area.markdown("🔍 **步骤1/5**：正在识别股票信息...")
        progress_bar.progress(10)
        stock_name = stock_code
        tech = {}
        with BaostockSession() as session:
            name = get_stock_info_baostock(stock_code, session)
            if name:
                stock_name = name
                progress_area.markdown(f"✅ **步骤1/5**：{stock_name} ({stock_code})")
            else:
                progress_area.markdown(f"⚠️ **步骤1/5**：名称获取失败，使用代码 {stock_code}")
            progress_bar.progress(20)

            progress_area.markdown("💰 **步骤2/5**：正在获取实时股价...")
            progress_bar.progress(30)
            price_info = get_realtime_price(stock_code)
            if not price_info or price_info.get('price') is None:
                # 四个外部源都失败，用 baostock 兜底，复用同一个已登录的 session
                price_info = get_baostock_fallback_price(stock_code, session)
            price_text = f"{price_info['price']} 元 ({price_info['change_pct']:+.2f}%)" if price_info and price_info.get('price') is not None and price_info.get('change_pct') is not None else (
                f"{price_info['price']} 元" if price_info and price_info.get('price') is not None else "获取失败"
            )
            progress_bar.progress(40)
            progress_area.markdown(f"✅ **步骤2/5**：当前股价 {price_text}")
            tech = get_technical_baostock(stock_code, session)

        progress_area.markdown("📰 **步骤3/5**：正在获取个股新闻...")
        progress_bar.progress(50)
        news_items = get_stock_news(stock_code)
        with col2:
            st.markdown("### 📡 个股实时新闻（可点击）")
            st.markdown(render_news_html(news_items), unsafe_allow_html=True)
        progress_bar.progress(70)
        progress_area.markdown(f"✅ **步骤3/5**：已加载 {len(news_items)} 条个股新闻")

        progress_bar.progress(80)
        if tech:
            tech_text = f"最新{tech['latest_price']:.2f}, MA5={tech['ma5']:.2f}, MA20={tech['ma20']:.2f}, 量比{tech['volume_change']:+.1f}%"
        else:
            tech_text = "计算失败"
        progress_area.markdown(f"✅ **步骤4/5**：{tech_text}")
        progress_bar.progress(85)

        progress_area.markdown("🧠 **步骤5/5**：AI 策略推演...")
        progress_bar.progress(90)
        now_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

        price_data = price_info if price_info else {}
        tech_data = tech if tech else {}

        # 补全数据
        if not price_data.get('price') and tech_data.get('latest_price'):
            price_data['price'] = tech_data['latest_price']
        if not price_data.get('amount_yi') and tech_data.get('today_amount_yi'):
            price_data['amount_yi'] = tech_data['today_amount_yi']

        # 提取字段（含换手率）
        open_price = price_data.get('open')
        high_price = price_data.get('high')
        low_price = price_data.get('low')
        yesterday_close = price_data.get('yesterday_close')
        current_price = price_data.get('price')
        amount_yi = price_data.get('amount_yi')
        turnover = price_data.get('turnover')  # 换手率（%）

        today_amplitude = tech_data.get('today_amplitude', 0)
        if today_amplitude == 0 and high_price is not None and low_price is not None and yesterday_close is not None:
            if yesterday_close != 0:
                today_amplitude = (high_price - low_price) / yesterday_close * 100

        def safe_float_str(value, fmt=".2f"):
            if value is None or pd.isna(value):
                return "暂无"
            try:
                return f"{value:{fmt}}"
            except:
                return str(value)

        turnover_str = safe_float_str(turnover, ".2f")
        yesterday_close_str = safe_float_str(yesterday_close)
        open_price_str = safe_float_str(open_price)
        high_price_str = safe_float_str(high_price)
        low_price_str = safe_float_str(low_price)
        current_price_str = safe_float_str(current_price)
        amount_yi_str = safe_float_str(amount_yi, ".2f")
        today_amplitude_str = safe_float_str(today_amplitude, ".2f")
        ma5_str = safe_float_str(tech_data.get('ma5'), ".2f")
        ma20_str = safe_float_str(tech_data.get('ma20'), ".2f")

        current_vs_ma = ''
        if current_price is not None:
            ma5_val = tech_data.get('ma5')
            ma20_val = tech_data.get('ma20')
            conditions = []
            if ma5_val is not None and not pd.isna(ma5_val):
                conditions.append('跌破5日线' if current_price < ma5_val else '站上5日线')
            else:
                conditions.append('5日线数据缺失')
            if ma20_val is not None and not pd.isna(ma20_val):
                conditions.append('跌破20日线' if current_price < ma20_val else '站上20日线')
            else:
                conditions.append('20日线数据缺失')
            current_vs_ma = '，'.join(conditions)
        else:
            current_vs_ma = '现价数据缺失'

        avg_amount_5_yi_str = safe_float_str(tech_data.get('avg_amount_5_yi'), ".2f")
        avg_amplitude_5_str = safe_float_str(tech_data.get('avg_amplitude_5'), ".2f")
        vol_ratio_str = safe_float_str(tech_data.get('vol_ratio_vs_5'), ".2f")
        dif_str = safe_float_str(tech_data.get('dif'), ".3f")
        dea_str = safe_float_str(tech_data.get('dea'), ".3f")
        macd_str = safe_float_str(tech_data.get('macd'), ".3f")
        k_str = safe_float_str(tech_data.get('k'), ".2f")
        d_str = safe_float_str(tech_data.get('d'), ".2f")
        j_str = safe_float_str(tech_data.get('j'), ".2f")

        data_summary = f"""
### 实时行情（{now_str}）
- 昨日收盘：{yesterday_close_str}元
- 今日开盘：{open_price_str}元
- 最高/最低/现价：{high_price_str} / {low_price_str} / {current_price_str}元
- 今日振幅：{today_amplitude_str}%
- 今日成交额：{amount_yi_str}亿元
- 今日换手率：{turnover_str}%

### 技术指标（日线）
- 5日均线：{ma5_str}元
- 20日均线：{ma20_str}元
- 当前股价相对均线：{current_vs_ma}
- 近5日日均成交额：{avg_amount_5_yi_str}亿元
- 近5日日均振幅：{avg_amplitude_5_str}%
- 今日成交量较5日均量倍数：{vol_ratio_str}倍

### MACD指标（日线）
- DIF：{dif_str}
- DEA：{dea_str}
- MACD柱：{macd_str}

### KDJ指标（日线）
- K值：{k_str}
- D值：{d_str}
- J值：{j_str}

### 近期个股新闻标题
{json.dumps([item['title'] for item in news_items], ensure_ascii=False, indent=2)}
"""

        system_prompt = f"""你是一位专业短线交易员，擅长技术分析。请严格按照以下步骤和格式，对用户提供的股票代码进行分析。

**当前日期为：{now_str.split()[0]}**。

输出格式要求（必须按顺序执行，不可跳过）：

## 第一步：定性分析（判断是否适合短线）
逐项核实以下四条，用表格形式输出：
| 核对项 | 标准要求 | 该股票实际情况（基于上述数据） | 结论（✅/❌） |
|--------|----------|--------------------------------|----------------|
| 流动性 | 近5日日均成交额 ≥ 1亿元 | [填写实际数据] | |
| 波动性 | 近5日日均振幅 ≥ 4% | [填写实际数据] | |
| 筹码活跃度 | 今日换手率 ≥ 2% 或 近5日日均换手率 ≥ 2% | [填写实际数据] | |
| 题材热度 | 是否属于近期市场热点板块（需列明具体概念） | [根据新闻标题和常识推测板块及催化剂] | |

若四项均为✅，继续后续步骤；若有任何一项❌，直接输出“不建议短线操作”并终止。

**风险前置提示**（若有）：列出该股已知的重大风险（如立案调查、业绩暴雷、异常波动公告、高负债等），每条用⚠️标注。

## 第二步：看“势”——判断当前位置
基于上述数据，输出以下内容：
- 昨日收盘价：xx元
- 今日开盘/最高/最低/现价：xx / xx / xx / xx元
- 今日振幅：xx%
- 今日成交额：xx亿元
- 今日换手率：xx%
- 5日均线位置：xx元；20日均线位置：xx元
- 当前股价相对于均线的状态（如：跌破5日线、仍在20日线上方等）
- K线形态描述（如：高开低走收阴线/低开高走反包阳线/墓碑线/长下影线等）

用一段话总结：目前处于【上涨中继/回调企稳/破位反抽/高位震荡/其他】阶段。**在判断时，请结合换手率的大小评判筹码锁定程度（换手率低代表惜售，高代表分歧大）。**

## 第三步：技术指标共振
以表格形式输出以下六个指标的日线级别信号：

| 指标 | 当日信号（需量化） | 简要解读 |
|------|------------------|----------|
| 均线系统 | （依据均线位置） | |
| 成交量 | （较5日均量倍数及放缩量情况） | |
| 换手率 | （实际数值及与近期对比） | （结合价格判断：放量换手是吸筹还是出货） |
| MACD | （DIF/DEA位置、金叉死叉、红绿柱变化） | |
| KDJ | （K/D/J数值、超买超卖、金叉死叉） | |
| 支撑/压力参考 | 支撑位：xx元；压力位：xx元 | |

综合小结：指标当前是否形成共振？偏多/偏空/分歧？

## 第四步：买入建议

### （一）：基于实战分析标准

根据以上分析，输出以下内容（若判断不应买入，则直接输出“⚠️ 不建议买入”，并列出理由）：

- **买入建议**：（建议买入/不建议买入/建议有限尝试）
- **买入时间**：（如：今日午后14:20-14:50/明日开盘后30分钟后再择机/等待回踩xx元后右侧确认等）
- **参考买入价格区间**：xx元 - xx元
- **计划仓位**：占总资金的百分比（如：不超过5%）
- **买入前必须确认的条件**（列出2-3条硬性条件，例如：换手率回落至某个区间等）
- **放弃买入的情形**（列出2-3条）

---

### （二）：如果非要买入

> ⚠️ 此部分为放宽条件的操作方案，风险显著高于标准操作，仅适用于愿意承担更高回撤的投资者，不构成标准建议。

- **买入时间**：（例如：今日午后14:50-14:57 / 明日开盘后30分钟观察 / 等待回踩xx元后等）
- **参考买入价格区间**：xx元 - xx元
- **计划仓位**：占总资金的百分比（必须显著低于标准仓位，如≤3%-5%）
- **买入前必须确认的条件**：列出2-3条硬性条件
- **放弃买入的情形**：列出2-3条触发放弃的具体信号
- **特别风控**：买入后的特殊止损纪律（例如：次日开盘30分钟内无条件止损；不得补仓等）

## 第五步：卖出建议
分别针对三类持仓者给出建议（若当前无持仓，可仅输出“若已持有则适用”）：

### 情形A：成本价 > 近期压力位（高位接盘者）
| 处理节点 | 操作动作 | 条件明细 |
|----------|----------|----------|
| | | |

### 情形B：成本价在中位区间（有利润垫或浅套）
| 处理节点 | 操作动作 | 条件明细 |
|----------|----------|----------|
| | | |

### 情形C：成本价很低（底仓持有者）
简述处理建议。

**通用硬止损线**：跌破xx元（如20日均线）无条件清仓。
**持仓时间边界**：若买入后x个交易日内未突破xx元，减仓或离场。

## 综合结论表
| 维度 | 结论 |
|------|------|
| 买入建议 | （一句话总结） |
| 卖出建议 | （一句话总结） |

请确保回答客观、冷静、直白，不添加情绪化词汇，不承诺收益。
"""

        user_message = f"""请分析以下股票：
股票代码：{stock_code}
股票名称：{stock_name}
{data_summary}
请按照上述格式输出。"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            timeout=60
        )
        raw = response.choices[0].message.content

        progress_bar.progress(98)
        progress_area.markdown("✨ 生成报告...")
        progress_bar.empty()
        progress_area.empty()

        st.subheader(f"📊 {stock_name}({stock_code}) 决策报告")
        st.write(f"**⏰ {now_str}**  &nbsp;&nbsp; **💰 {price_text}**")
        st.success("AI 分析完成")
        st.markdown(raw)
        st.warning("⚠️ 仅供参考，投资需谨慎。")

    except RuntimeError as e:
        st.error(f"baostock 连接失败：{e}")
    except Exception as e:
        st.error(f"分析异常: {e}")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Powered by baostock + 东方财富 + AkShare | 含换手率分析</p>", unsafe_allow_html=True)
