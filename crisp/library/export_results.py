import os
import shutil


def copy_file_and_paste(source_path, destination_path):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)


def prep_metrics_workbook(destination_path, wb, ws):
    col = ws.max_column + 1
    ws.cell(row=1, column=col, value="Accuracy")
    ws.cell(row=1, column=col + 1, value="Precision")
    ws.cell(row=1, column=col + 2, value="Recall")
    ws.cell(row=1, column=col + 3, value="F1")
    wb.save(destination_path)


def export_metrics(wb, destination_path, ws, row, col, accuracy, precision, recall, f1):
    ws.cell(row=row, column=col, value=round(accuracy, 2))
    ws.cell(row=row, column=col + 1, value=round(precision, 2))
    ws.cell(row=row, column=col + 2, value=round(recall, 2))
    ws.cell(row=row, column=col + 3, value=round(f1, 2))
    wb.save(destination_path)
