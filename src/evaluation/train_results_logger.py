import hashlib
import logging
import os
import re
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

import pandas as pd

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


class ResultsLogger:

    @staticmethod
    def generate_run_identifier(config: ExperimentConfig) -> str:
        """Generates a short, readable identifier from SETUP_ parameters."""
        config_dict = asdict(config)
        parts = []

        acronyms = {
            'SETUP_LANDMARK_DETECTION_METHOD': 'ldm',
            'SETUP_FACE_DETECTION_METHOD': 'fdm',
            'SETUP_FEATURES_TO_USE': 'feats',
            'SETUP_VALIDATION_STRATEGY': 'val',
            'SETUP_HYPERPARAMETER_TUNING': 'tune'
        }

        for key, value in config_dict.items():
            if key.startswith('SETUP_'):
                # use acronym or derive from key
                prefix = acronyms.get(key, key.replace('SETUP_', '').lower())

                if isinstance(value, Enum):
                    val_str = value.name
                elif isinstance(value, list):
                    val_str = '-'.join(str(v) for v in value)
                elif isinstance(value, bool):
                    val_str = 'T' if value else 'F'
                else:
                    val_str = str(value)

                parts.append(f"{prefix}-{val_str}")

        return "_".join(parts)


    @staticmethod
    def generate_safe_filename_from_config(config: ExperimentConfig, extension: str) -> str:
        """Creates a short, safe filename using a hash of the full config identifier."""
        full_identifier = ResultsLogger.generate_run_identifier(config)

        short_hash = hashlib.md5(full_identifier.encode('utf-8')).hexdigest()

        return f"run_{short_hash}.{extension}"


    @staticmethod
    def _extract_model_specific_config(model_name: str, config: ExperimentConfig) -> str:
        """Extracts parameters relevant to a model, preserving type visualization."""

        model_name = model_name.lower()

        if 'svm' in model_name and 'deep' in model_name:
            base_model_name = 'svm_deep'
        elif 'cnn' in model_name:
            base_model_name = 'cnn'
        elif 'lstm' in model_name:
            base_model_name = 'lstm'
        else:
            base_model_name = model_name.split('_')[0]

        prefix = 'SVM_' if base_model_name == 'svm_deep' else f'{base_model_name.upper()}_'
        config_dict = asdict(config)
        model_params = {k: v for k, v in config_dict.items() if k.startswith(prefix)}

        return str(model_params)


    @staticmethod
    def sanitize_sheet_name(name: str) -> str:
        return re.sub(r'[\\/*?:\[\]]', '_', name)[:31]


    @staticmethod
    def _log_header_config_to_sheet(writer: pd.ExcelWriter, sheet_name: str, config: ExperimentConfig, validation_split_data: Optional[Dict] = None) -> int:
        """Logs all SETUP_ parameters to the sheet header."""
        config_dict = asdict(config)

        # find all SETUP_ parameters
        header_data = {key: value for key, value in config_dict.items() if key.startswith('SETUP_')}

        log_data = {
            "Parameter": ["First Run Timestamp"] + list(header_data.keys()),
            "Value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + [str(v) for v in header_data.values()]
        }

        if validation_split_data:
            log_data["Parameter"].append("")  # Separator
            log_data["Value"].append("")

            log_data["Parameter"].append(f"Validation Strategy: {validation_split_data['strategy']}")
            log_data["Value"].append("")

            # Handle single split (Holdout)
            if 'Train Subjects' in validation_split_data:
                log_data["Parameter"].append("  Train Subjects")
                log_data["Value"].append(validation_split_data['Train Subjects'])
                log_data["Parameter"].append("  Test Subjects")
                log_data["Value"].append(validation_split_data['Test Subjects'])

            # Handle multiple folds (K-Fold, LOSO)
            elif 'Folds' in validation_split_data:
                for fold in validation_split_data['Folds']:
                    log_data["Parameter"].append(f"  Fold {fold['Fold']} Train")
                    log_data["Value"].append(fold['Train Subjects'])
                    log_data["Parameter"].append(f"  Fold {fold['Fold']} Test")
                    log_data["Value"].append(fold['Test Subjects'])

        config_df = pd.DataFrame(log_data)

        def style_config(df):
            style_df = pd.DataFrame('', index=df.index, columns=df.columns)
            highlight = df['Parameter'].astype(str).str.contains("Validation Strategy:", na=False)
            for idx in df.index[highlight]:
                style_df.loc[idx, :] = 'font-weight: bold;'
            return style_df

        styled_config = config_df.style.apply(style_config, axis=None)
        styled_config.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)

        return len(config_df) + 2

    @staticmethod
    def log_results_to_excel(results_list: List[Dict], config: ExperimentConfig, validation_split_data: Optional[Dict] = None):
        if not results_list: return

        base_output_dir = os.path.join(config.RESULTS_EXCEL_FILE, config.SUITE_NAME)
        safe_suite_name = re.sub(r'[\\/*?:\\[\\]]', '_', config.SUITE_NAME)
        excel_filename = f"{safe_suite_name}_results.xlsx"
        filepath = os.path.join(base_output_dir, excel_filename)
        os.makedirs(base_output_dir, exist_ok=True)

        run_identifier = ResultsLogger.generate_run_identifier(config)
        short_hash = hashlib.md5(run_identifier.encode('utf-8')).hexdigest()[:6]
        sheet_name = ResultsLogger.sanitize_sheet_name(f"{config.NAME}_{short_hash}")

        # Only add Config to final row of each group
        model_type_regex = r'_(OptimalF1|Balanced|OptimalAccuracy|OptimalRecall)(?:_Fold_\d+|_CV_Mean)?$'
        for result in results_list:
            model = result.get('Model', '') or ''
            result_type = re.search(model_type_regex, model)

            if result_type:
                if 'CV_Mean' in model:
                    result['Config'] = ResultsLogger._extract_model_specific_config(model, config)
                elif 'Fold_' in model:
                    result['Config'] = ''  # skip fold rows
                else:
                    result['Config'] = ''  # default
            else:
                result['Config'] = ''

        df_new = pd.DataFrame(results_list)
        df_new["Log Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if 'Config' in df_new and 'Model' in df_new:
            cols = df_new.columns.tolist()
            cols.insert(cols.index('Model') + 1, cols.pop(cols.index('Config')))
            df_new = df_new[cols]

        def highlight_types(df):
            type_regex = r'_(OptimalF1|OptimalAccuracy|OptimalRecall|Balanced|Other)$'
            shades = ['#A8CCE4', '#BBDCF1', '#CDE6FB', '#D7E8FD', '#E6F1FF']
            matchers = df['Model'].str.extract(type_regex)[0].fillna('Other')
            color_map = {typ: shades[i % len(shades)] for i, typ in
                         enumerate(['OptimalF1', 'OptimalAccuracy', 'OptimalRecall', 'Balanced', 'Other'])}
            style_df = pd.DataFrame('', index=df.index, columns=df.columns)
            for idx, typ in matchers.items():
                color = color_map.get(typ, '')  # '' for no color
                if color:
                    style_df.loc[idx, :] = f'background-color: {color};'
            return style_df

        def highlight_summary_rows(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            model_col = df['Model'].astype(str)

            bold_rows = model_col.str.contains(r'CV_Mean', case=False, na=False)
            dark_rows = model_col.str.contains(r'Validation|Strategy', case=False, na=False)

            for idx in df.index[bold_rows]:
                styles.loc[idx, :] += 'font-weight: bold;'
            for idx in df.index[dark_rows]:
                styles.loc[idx, :] = 'background-color: #404040; color: white; font-weight: bold;'
            return styles

        styled_df = df_new.style \
            .apply(highlight_types, axis=None) \
            .apply(highlight_summary_rows, axis=None) \
            .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#222F3E'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('border', '1px solid #444'), ('padding', '4px')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
        ])

        try:
            mode = 'a' if os.path.exists(filepath) else 'w'
            with pd.ExcelWriter(filepath, engine='openpyxl', mode=mode,
                                if_sheet_exists='overlay' if mode == 'a' else None) as writer:
                if mode == 'a' and sheet_name in writer.book.sheetnames:
                    sheet = writer.book[sheet_name]
                    start_row = sheet.max_row

                    black_row = pd.DataFrame([[''] * len(df_new.columns)], columns=df_new.columns)
                    black_style = pd.DataFrame('background-color: black;', index=black_row.index, columns=black_row.columns)
                    black_styled = black_row.style.apply(lambda _: black_style, axis=None)

                    black_styled.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=start_row)
                    styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=start_row + 1)
                else:
                    config_rows = ResultsLogger._log_header_config_to_sheet(writer, sheet_name, config, validation_split_data)
                    styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=True, startrow=config_rows)
            logger.info(f"Results logged to '{filepath}', Sheet: '{sheet_name}'")
        except Exception as e:
            logger.error(f"Error logging results to Excel: {e}", exc_info=True)

