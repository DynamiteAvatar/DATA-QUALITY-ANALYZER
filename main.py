import flet as ft
from flet import Icons, Colors
import pandas as pd
import flet.fastapi as flet_fastapi


from data_analyser import UnifiedDataQualityAnalyzer
from data_viz import (
    generate_missing_data_heatmap,
    generate_feature_correlation_clustermap,
    generate_numerical_distribution_plot,
    generate_column_quality_heatmap
)

ACTIVE_VERSION = "current_data"


def main(page: ft.Page):
    page.title = "Unified Data Quality Analyzer"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.bgcolor = "#F7F9FC"

    analyzer = UnifiedDataQualityAnalyzer()

    # ---------------- FILE PICKERS ----------------
    upload_picker = ft.FilePicker()
    save_picker = ft.FilePicker()
    page.overlay.extend([upload_picker, save_picker])

    # ---------------- HEADER ----------------
    title = ft.Text("Unified Data Quality Analyzer", size=22, weight="bold")

    dataset_info = ft.Text("No dataset loaded", italic=True)

    health_text = ft.Text()
    completeness_text = ft.Text()
    duplicate_text = ft.Text()

    # ---------------- INSPECTOR PANEL ----------------
    inspector_title = ft.Text("Issue Inspector", weight="bold")
    inspector_body = ft.Text(
        "Click any row in the tables to see details here.",
        selectable=True
    )

    inspector_panel = ft.Container(
        width=320,
        padding=15,
        border_radius=8,
        bgcolor=Colors.WHITE,
        border=ft.border.all(1, Colors.GREY_300),
        content=ft.Column([inspector_title, ft.Divider(), inspector_body])
    )

    def update_inspector(title, text):
        inspector_title.value = title
        inspector_body.value = text
        page.update()

    # ---------------- SAFE TABLE INIT ----------------
    def table_card(title_text):
        return ft.Container(
            expand=True,
            border=ft.border.all(1, Colors.GREY_300),
            border_radius=8,
            padding=10,
            content=ft.Column([
                ft.Text(title_text, weight="bold"),
                ft.DataTable(columns=[ft.DataColumn(ft.Text("—"))], rows=[])
            ])
        )

    missing_card = table_card("Missing Data")
    consistency_card = table_card("Consistency Issues")
    dup_rows_card = table_card("Duplicate Rows")
    dup_cols_card = table_card("Duplicate Columns")

    # ---------------- VISUAL PLACEHOLDERS ----------------
    def viz_card(title_text):
        return ft.Container(
            expand=True,
            border=ft.border.all(1, Colors.GREY_300),
            border_radius=8,
            padding=10,
            content=ft.Column([
                ft.Text(title_text, weight="bold"),
                ft.Container(
                    height=260,
                    alignment=ft.alignment.center,
                    content=ft.Text("Run analysis to view", italic=True)
                )
            ])
        )

    missing_viz = viz_card("Missing Data Heatmap")
    corr_viz = viz_card("Correlation Matrix")
    dist_viz = viz_card("Distribution Plot")
    quality_viz = viz_card("Column Quality Risk Matrix")

    # ---------------- EXPLANATION DIALOGS (FIXED CLOSE BUTTON) ----------------
    def create_dialog(title, text):
        # We create the dialog instance first
        dlg = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(text),
        )
        # Now we assign the action using the direct reference to 'dlg'
        dlg.actions = [
            ft.TextButton("Close", on_click=lambda _: page.close(dlg))
        ]
        return dlg

    dist_dialog = create_dialog(
        "Distribution Explained",
        "This curve shows how numeric values are distributed.\n\n"
        "• Right skew → large outliers\n"
        "• Left skew → compressed upper range\n"
        "• Symmetric → normal-like distribution"
    )

    missing_dialog = create_dialog(
        "Missing Data Heatmap Explained",
        "Visualizes where data is missing across the dataset.\n\n"
        "• Dark patches → Missing values\n"
        "• Patterns → May indicate systematic collection errors."
    )

    corr_dialog = create_dialog(
        "Correlation Matrix Explained",
        "Shows relationships between numerical features.\n\n"
        "• Near 1.0 → Strong positive link\n"
        "• Near -1.0 → Strong inverse link\n"
        "• Near 0 → No linear relationship."
    )

    quality_dialog = create_dialog(
        "Quality Risk Matrix Explained",
        "A composite view of column health.\n\n"
        "• Red zones → High missingness or low cardinality issues.\n"
        "• Green zones → Healthy, reliable columns."
    )

    # ---------------- LOAD CSV ----------------
    def on_file_selected(e):
        if not e.files:
            return
        df = pd.read_csv(e.files[0].path)
        analyzer.data_versions[ACTIVE_VERSION] = df
        dataset_info.value = (
            f"Loaded: {e.files[0].name} | Rows: {len(df)} | Columns: {len(df.columns)}"
        )
        analyze_btn.disabled = False
        page.update()

    upload_picker.on_result = on_file_selected

    # ---------------- ANALYSIS ----------------
    def run_analysis(e):
        analyzer.run_full_scan(performance_mode=False)
        analyzer.prepare_data(ACTIVE_VERSION)

        report = analyzer.quality_reports[ACTIVE_VERSION]
        df = analyzer.data_versions[ACTIVE_VERSION]

        health_text.value = f"Health %: {report['dimensions']['overall_quality_score']}"
        completeness_text.value = f"Completeness %: {report['dimensions']['completeness_score']}"
        duplicate_text.value = f"Duplicate Rows: {report['duplicates_exact']['duplicate_rows_count']}"

        # ---------- Missing Data Table ----------
        missing_card.content.controls[1] = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Column")),
                ft.DataColumn(ft.Text("Missing Count")),
                ft.DataColumn(ft.Text("Missing %"))
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(m["column"])),
                        ft.DataCell(ft.Text(str(m["missing_count"]))),
                        ft.DataCell(ft.Text(str(m["missing_percent"])))
                    ],
                    on_select_changed=lambda e, m=m: update_inspector(
                        "Missing Data",
                        f"Column: {m['column']}\n"
                        f"Missing: {m['missing_count']} ({m['missing_percent']}%)\n\n"
                        "Recommendation: Impute or drop if sparse."
                    )
                )
                for m in report["missing"]["missing_summary"]
            ]
        )

        # ---------- Consistency Table ----------
        consistency_card.content.controls[1] = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Column")),
                ft.DataColumn(ft.Text("Check")),
                ft.DataColumn(ft.Text("Invalid Count"))
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(i["column"])),
                        ft.DataCell(ft.Text(i["check"])),
                        ft.DataCell(ft.Text(str(i["invalid_count"])))
                    ],
                    on_select_changed=lambda e, i=i: update_inspector(
                        "Consistency Issue",
                        f"Column: {i['column']}\n"
                        f"Rule: {i['check']}\n"
                        f"Invalid rows: {i['invalid_count']}"
                    )
                )
                for i in report["consistency"]["issues"]
            ]
        )

        # ---------- Duplicate Rows Table ----------
        dup_rows_card.content.controls[1] = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Row Index"))],
            rows=[
                ft.DataRow(
                    cells=[ft.DataCell(ft.Text(str(idx)))],
                    on_select_changed=lambda e, idx=idx: update_inspector(
                        "Duplicate Row",
                        f"Row index {idx} is duplicated.\n"
                        "Recommendation: Safe to remove."
                    )
                )
                for idx in report["duplicates_exact"]["duplicate_row_indices"]
            ]
        )

        # ---------- Duplicate Columns Table ----------
        dup_cols_card.content.controls[1] = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Type")),
                ft.DataColumn(ft.Text("Columns"))
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text("Value")),
                        ft.DataCell(ft.Text(f"{a} vs {b}"))
                    ],
                    on_select_changed=lambda e, a=a, b=b: update_inspector(
                        "Duplicate Columns",
                        f"Columns {a} and {b} contain identical data.\n"
                        "Recommendation: Drop one."
                    )
                )
                for a, b in report["duplicate_columns"]["duplicate_value_columns"]
            ]
        )

        # ---------- VISUALS ----------
        missing_viz.content.controls[1] = ft.Column([
            ft.Image(src_base64=generate_missing_data_heatmap(df), width=520),
            ft.TextButton("Explain Missing Data", icon=Icons.INFO_OUTLINE, on_click=lambda _: page.open(missing_dialog))
        ])

        corr_viz.content.controls[1] = ft.Column([
            ft.Image(src_base64=generate_feature_correlation_clustermap(df), width=520),
            ft.TextButton("Explain Correlation", icon=Icons.INFO_OUTLINE, on_click=lambda _: page.open(corr_dialog))
        ])

        dist_viz.content.controls[1] = ft.Column([
            ft.Image(src_base64=generate_numerical_distribution_plot(df), width=520),
            ft.TextButton("Explain Distribution", icon=Icons.INFO_OUTLINE, on_click=lambda _: page.open(dist_dialog))
        ])

        quality_viz.content.controls[1] = ft.Column([
            ft.Image(src_base64=generate_column_quality_heatmap(df), width=520),
            ft.TextButton("Explain Quality Risk", icon=Icons.INFO_OUTLINE, on_click=lambda _: page.open(quality_dialog))
        ])

        download_btn.disabled = False
        page.update()

    # ---------------- DOWNLOAD ----------------
    def save_csv(e):
        if e.path:
            analyzer.data_versions[ACTIVE_VERSION].to_csv(e.path, index=False)

    save_picker.on_result = save_csv

    # ---------------- RESET ----------------
    def reset_app(e):
        page.clean()
        main(page)

    # ---------------- BUTTONS ----------------
    upload_btn = ft.ElevatedButton(
        "Upload CSV", icon=Icons.UPLOAD_FILE,
        on_click=lambda _: upload_picker.pick_files(allowed_extensions=["csv"])
    )
    analyze_btn = ft.ElevatedButton(
        "Analyze", icon=Icons.SCIENCE,
        disabled=True, on_click=run_analysis
    )
    download_btn = ft.ElevatedButton(
        "Download Cleaned CSV", icon=Icons.DOWNLOAD,
        disabled=True,
        on_click=lambda _: save_picker.save_file(file_name="cleaned.csv")
    )
    reset_btn = ft.TextButton("Reset", on_click=reset_app)

    # ---------------- LAYOUT ----------------
    page.add(
        ft.Row([title, upload_btn, analyze_btn, download_btn, reset_btn], spacing=15),
        ft.Divider(),
        dataset_info,
        ft.Row([health_text, completeness_text, duplicate_text], spacing=30),
        ft.Divider(),

        # Data Quality Section with Inspector
        ft.Row([
            ft.Column([
                ft.Text("Data Quality", size=18, weight="bold"),
                ft.Row([missing_card, consistency_card], spacing=20),
                ft.Row([dup_rows_card, dup_cols_card], spacing=20),
            ], expand=True),
            inspector_panel
        ], vertical_alignment=ft.CrossAxisAlignment.START),

        ft.Divider(),

        # Visual Analysis Section
        ft.Column([
            ft.Text("Visual Analysis", size=18, weight="bold"),
            ft.Row([missing_viz, corr_viz], spacing=20),
            ft.Row([dist_viz, quality_viz], spacing=20),
        ])
    )


ft.app(target=main, view=ft.AppView.WEB_BROWSER)
