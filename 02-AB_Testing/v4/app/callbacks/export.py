from __future__ import annotations

from datetime import datetime, timezone

from dash import Input, Output, State, no_update

from app.export_utils import build_summary_df, generate_text_report


def register_export_callbacks(app):
    @app.callback(
        Output("export-download", "data"),
        Input("download-txt-btn", "n_clicks"),
        Input("download-csv-btn", "n_clicks"),
        State("store-analysis", "data"),
        prevent_initial_call=True,
    )
    def download_report(txt_clicks, csv_clicks, R):
        if not R:
            return no_update
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if triggered_id == "download-txt-btn":
            return dict(content=generate_text_report(R),
                        filename=f"ab_testing_report_{ts}.txt", type="text/plain")
        elif triggered_id == "download-csv-btn":
            return dict(content=build_summary_df(R).to_csv(index=False),
                        filename=f"ab_test_summary_{ts}.csv", type="text/csv")
        return no_update
