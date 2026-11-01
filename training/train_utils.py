import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

def compute_metrics(y_true, y_pred, y_prob=None, pos_label=1):
    """Compute comprehensive evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
        )
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        
    return metrics

def draw_confusion_matrix(cm, class_names, title, out_path):
    """Render and save a professional confusion matrix image using PIL."""
    width, height = 640, 520
    img = Image.new("RGB", (width, height), color=(248, 249, 250))
    draw = ImageDraw.Draw(img)
    
    # Draw header bar
    draw.rectangle([0, 0, width, 55], fill=(30, 41, 59))
    draw.text((25, 18), title, fill=(255, 255, 255))
    
    # Subtitle / axis labels
    draw.text((25, 70), "True Label (Rows) vs Predicted Label (Columns)", fill=(71, 85, 105))
    
    # Box parameters
    start_x, start_y = 160, 140
    cell_size = 180
    
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    
    matrix_cells = [
        (0, 0, tn, f"TN: {tn}", f"({tn/total*100:.1f}%)", (220, 252, 231), (22, 101, 52)),
        (0, 1, fp, f"FP: {fp}", f"({fp/total*100:.1f}%)", (254, 226, 226), (153, 27, 27)),
        (1, 0, fn, f"FN: {fn}", f"({fn/total*100:.1f}%)", (254, 243, 199), (146, 64, 14)),
        (1, 1, tp, f"TP: {tp}", f"({tp/total*100:.1f}%)", (219, 234, 254), (30, 64, 175)),
    ]
    
    # Column Headers (Predicted)
    draw.text((start_x + cell_size // 2 - 40, start_y - 30), f"Pred: {class_names[0]}", fill=(15, 23, 42))
    draw.text((start_x + cell_size + cell_size // 2 - 40, start_y - 30), f"Pred: {class_names[1]}", fill=(15, 23, 42))
    
    # Row Headers (True)
    draw.text((start_x - 130, start_y + cell_size // 2 - 10), f"True: {class_names[0]}", fill=(15, 23, 42))
    draw.text((start_x - 130, start_y + cell_size + cell_size // 2 - 10), f"True: {class_names[1]}", fill=(15, 23, 42))
    
    for r, c, val, label_text, pct_text, bg_col, text_col in matrix_cells:
        x1 = start_x + c * cell_size
        y1 = start_y + r * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        
        draw.rectangle([x1, y1, x2, y2], fill=bg_col, outline=(148, 163, 184), width=2)
        draw.text((x1 + cell_size // 2 - 35, y1 + cell_size // 2 - 18), label_text, fill=text_col)
        draw.text((x1 + cell_size // 2 - 25, y1 + cell_size // 2 + 5), pct_text, fill=text_col)
        
    # Summary footer
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    footer_text = f"Total Samples: {total} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
    draw.rectangle([0, height - 40, width, height], fill=(241, 245, 249))
    draw.text((25, height - 28), footer_text, fill=(30, 41, 59))
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
