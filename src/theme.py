"""src/theme.py — 配色常量与工具函数（纯数据，无副作用）"""

# 配色色板
PRIMARY   = "#4F46E5"
PURPLE    = "#7C3AED"
CYAN      = "#06B6D4"
SUCCESS   = "#10B981"
WARNING   = "#F59E0B"
DANGER    = "#EF4444"
ORANGE    = "#F97316"
TEAL      = "#06B6D4"

BG         = "#F8FAFC"
CARD       = "#FFFFFF"
BORDER     = "#E2E8F0"
CHART_GRID = "#F1F5F9"
TEXT       = "#1E293B"
MUTED      = "#64748B"
LIGHT      = "#94A3B8"


def score_color(s: float) -> str:
    if s >= 80: return SUCCESS
    if s >= 60: return WARNING
    if s >= 40: return ORANGE
    return DANGER


def score_label(s: float) -> str:
    if s >= 80: return "优秀"
    if s >= 60: return "良好"
    if s >= 40: return "一般"
    return "较低"
