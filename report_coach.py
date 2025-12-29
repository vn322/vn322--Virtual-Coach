# report_coach.py — финальная версия: корректные русские названия, 3 лучших графика, 3 персонализированные рекомендации

import os
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastdtw import fastdtw


# === 1. Шрифт с fallback ===
FONT_NAME = "Helvetica"
try:
    font_path = 'DejaVuSans.ttf'
    if not os.path.exists(font_path):
        import sys
        if hasattr(sys, '_MEIPASS'):
            font_path = os.path.join(sys._MEIPASS, 'DejaVuSans.ttf')
    pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
    FONT_NAME = 'DejaVuSans'
except Exception as e:
    print("⚠️ DejaVuSans.ttf не найден:", e)


def safe_text(s):
    """Фолбэк для Helvetica"""
    if isinstance(s, str) and FONT_NAME == 'Helvetica':
        return s.encode('utf-8', errors='replace').decode('latin1', errors='replace')
    return str(s)


def _clean_interp(a, b, n_target=60):
    """Очистка от NaN и интерполяция к одинаковой длине"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return None, None
    n = max(len(a), len(b), n_target)
    ta = np.linspace(0, 1, len(a))
    tb = np.linspace(0, 1, len(b))
    a_i = np.interp(np.linspace(0, 1, n), ta, a)
    b_i = np.interp(np.linspace(0, 1, n), tb, b)
    return a_i, b_i


def _calc_dtw_norm(model_series, user_series):
    """Нормированное DTW: DTW / (σ_model * N) → безразмерная мера отклонения"""
    a_i, b_i = _clean_interp(model_series, user_series)
    if a_i is None:
        return np.nan
    try:
        dtw_dist, _ = fastdtw(a_i, b_i)
        sigma_m = np.std(model_series.dropna()) or 1e-3
        return dtw_dist / (sigma_m * len(a_i))
    except:
        return np.nan


def _plot_single(ax, model_series, user_series, title):
    a_i, b_i = _clean_interp(model_series, user_series)
    if a_i is None:
        return False
    ax.plot(a_i, label="Модель", color="#2196F3", linewidth=2.0)
    ax.plot(b_i, label="Спортсмен", color="#FF5722", linestyle="--", linewidth=1.8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.4, linestyle=':')
    return True


def localize_param_name(col):
    """Преобразует 'shoulder_left_angle' → 'Левое плечо (угол)' и т.д."""
    # Словари перевода
    sides = {"left": "Левое", "right": "Правое"}
    joints = {
        "shoulder": "плечо",
        "elbow": "локоть",
        "hip": "бедро",
        "knee": "колено",
        "x_factor": "X-фактор"
    }
    metrics = {
        "_angle": "величина угла",
        "_velocity": "скорость",
        "_acceleration": "ускорение",
        "_entropy": "энтропия"
    }

    # Обработка X-фактора отдельно
    if col.startswith("x_factor"):
        base = "X-фактор"
    else:
        for joint_key in ["shoulder", "elbow", "hip", "knee"]:
            if joint_key in col:
                side = "left" if "_left_" in col or col.endswith("_left") else "right"
                base = f"{sides[side]} {joints[joint_key]}"
                break
        else:
            return col.replace("_", " ").title()

    # Определяем тип метрики
    for suffix, name in metrics.items():
        if suffix in col:
            return f"{base} ({name})"

    return base


def save_coach_report(base_path, model_df, user_df, score, details, user_frames=None):
    os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)
    
    # --- CSV ---
    user_df.to_csv(f"{base_path}.csv", index=False, float_format="%.3f")

    # --- PDF ---
    pdf_path = f"{base_path}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', fontName=FONT_NAME, fontSize=20, spaceAfter=14, alignment=1)
    normal_style = ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=12, spaceAfter=8)
    small_style = ParagraphStyle('Small', fontName=FONT_NAME, fontSize=10, spaceAfter=4)

    story = []
    story.append(Paragraph(safe_text("Отчёт Virtual Coach"), title_style))
    color = "#4CAF50" if score >= 80 else "#FF9800" if score >= 60 else "#f44336"
    story.append(Paragraph(safe_text(f"<b style='color:{color}'>Общий результат: {score:.1f}%</b> совпадения с моделью"), normal_style))
    story.append(Spacer(1, 12))

    # --- Таблица по группам ---
    group_names = {
        "key_angles": "Ключевые углы (плечи, X-фактор)",
        "other_angles": "Остальные углы",
        "velocities": "Угловые скорости",
        "accelerations": "Угловые ускорения",
        "entropies": "Энтропии (стабильность)"
    }
    data = [[safe_text("Группа"), "Оценка", "Вес"]]
    for g, (s, w) in details.items():
        data.append([safe_text(group_names[g]), f"{s*100:.1f}%", f"{w:.2f}"])
    table = Table(data, colWidths=[3.2*inch, 1.0*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    # --- Графики: 6 ключевых параметров (без дублирования) ---
    plot_specs = [
        ("x_factor_angle", "X-фактор (угол поворота туловища)"),
        ("shoulder_left_angle", "Левое плечо (угол)"),
        ("shoulder_right_angle", "Правое плечо (угол)"),
        ("x_factor_velocity", "X-фактор (скорость)"),
        ("shoulder_left_velocity", "Левое плечо (скорость)"),
        ("shoulder_right_velocity", "Правое плечо (скорость)"),
    ]

    for col, title in plot_specs:
        if col in model_df.columns and col in user_df.columns:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
                if _plot_single(ax, model_df[col], user_df[col], title):
                    plt.tight_layout()
                    path = f"{base_path}_{col}.png"
                    plt.savefig(path, dpi=150, bbox_inches='tight')
                    plt.close()
                    story.append(Image(path, width=6*inch, height=2.0*inch))
                    story.append(Spacer(1, 8))
            except Exception:
                continue

    # === ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ: ВСЕГДА ТОП-3 ПО DTW ===
    story.append(Paragraph("Наибольшие отклонения", ParagraphStyle('H2', parent=styles['Heading2'], fontName=FONT_NAME)))
    
    deviations = []
    for col in model_df.columns:
        if col.endswith('_angle') or col.endswith('_velocity') or col.endswith('_acceleration') or col.endswith('_entropy'):
            if col in user_df.columns:
                dtw_norm = _calc_dtw_norm(model_df[col], user_df[col])
                if not np.isnan(dtw_norm):
                    deviations.append((col, dtw_norm))

    # Сортируем по убыванию отклонения
    deviations.sort(key=lambda x: x[1], reverse=True)
    top3 = deviations[:3]

    recs = []
    advice_map = {
        "_angle": "Обратите внимание на амплитуду и форму движения в ключевые фазы.",
        "_velocity": "Скорость выполнения не совпадает с моделью: возможен слишком резкий старт или «просадка» в середине фазы.",
        "_acceleration": "Динамика ускорения нарушена: движение недостаточно плавное или, наоборот, излишне резкое.",
        "_entropy": "Движение нестабильно: повторяемость низкая — уделите внимание контролю."
    }

    for i, (col, dtw_val) in enumerate(top3, 1):
        name_ru = localize_param_name(col)
        advice = next((v for k, v in advice_map.items() if k in col), "Проверьте технику выполнения этого параметра.")
        recs.append(f"🔹 <b>{i}. {name_ru}</b>: {advice}")

    # Если данных мало — заглушка
    if not recs:
        recs.append("🔹 Недостаточно данных для анализа отклонений.")

    for r in recs:
        story.append(Paragraph(safe_text(r), normal_style))
    story.append(Spacer(1, 10))



    doc.build(story)