import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QGroupBox, QHeaderView, QComboBox, QFileDialog, QSplitter, QTextEdit,
    QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PDF Export dependencies
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==========================================
# 1. Synthetic Data Generation Engine
# ==========================================
def generate_synthetic_sports_data():
    """Generates synthetic performance data for 20 athletes across 12 weeks."""
    np.random.seed(42)
    
    athletes = [f"Athlete {i:02d}" for i in range(1, 21)]
    weeks = list(range(1, 13))
    phases = ["General Prep", "Hypertrophy", "Max Strength", "Power & Taper"]
    
    records = []
    base_date = datetime(2026, 1, 5)
    
    for athlete_idx, athlete in enumerate(athletes):
        # Base athletic capability parameters per athlete
        base_sprint = np.random.uniform(4.5, 5.2)  # 40m sprint (s)
        base_jump = np.random.uniform(40, 65)       # Jump height (cm)
        base_strength = np.random.uniform(100, 180) # 1RM Squat (kg)
        base_hrv = np.random.uniform(55, 85)        # Baseline HRV (ms)
        
        for wk in weeks:
            phase = phases[(wk - 1) // 3]
            sessions_per_week = np.random.choice([3, 4, 5])
            
            for s in range(1, sessions_per_week + 1):
                session_date = base_date + timedelta(weeks=wk - 1, days=(s - 1) * 2)
                
                # Training Load dynamics
                rpe = np.random.uniform(5, 9.5)
                duration = np.random.choice([60, 75, 90, 105])
                training_load = rpe * duration
                
                # Overreach/Fatigue injection for specific weeks
                fatigue_spike = 1.3 if wk in [4, 8] and athlete_idx % 3 == 0 else 1.0
                
                # Physiological Responses
                recovery = max(20, min(100, 100 - (rpe * 8 * fatigue_spike) + np.random.normal(0, 5)))
                fatigue_score = max(1, min(10, (100 - recovery) / 10.0 + np.random.normal(0, 0.5)))
                
                hr_avg = int(120 + rpe * 7 + np.random.normal(0, 4))
                hrv = max(30, min(110, base_hrv - (fatigue_score * 4) + np.random.normal(0, 3)))
                
                # Performance metrics influenced by fatigue & progression
                progression_factor = 1.0 - (wk * 0.005)  # Slight improvement over time
                sprint_time = round(base_sprint * progression_factor + (fatigue_score * 0.02) + np.random.normal(0, 0.02), 2)
                jump_height = round(base_jump * (2.0 - progression_factor) - (fatigue_score * 0.4) + np.random.normal(0, 0.5), 1)
                strength = round(base_strength + (wk * 1.2) - (fatigue_score * 0.8) + np.random.normal(0, 1.0), 1)
                
                records.append({
                    "Date": session_date.strftime("%Y-%m-%d"),
                    "Athlete": athlete,
                    "Week": wk,
                    "Training_Phase": phase,
                    "Session": s,
                    "Training_Load": round(training_load, 1),
                    "RPE": round(rpe, 1),
                    "Duration_Min": duration,
                    "Sprint_Time_s": sprint_time,
                    "Jump_Height_cm": jump_height,
                    "Strength_kg": strength,
                    "Recovery_Pct": round(recovery, 1),
                    "HR_Avg_bpm": hr_avg,
                    "HRV_ms": round(hrv, 1),
                    "Fatigue_Score": round(fatigue_score, 1)
                })
                
    return pd.DataFrame(records)


# ==========================================
# 2. Analytics & Pattern Flagging Engine
# ==========================================
class AnalyticsEngine:
    @staticmethod
    def detect_flags(df):
        """Identifies automated flags for workload spikes, high fatigue, and trend shifts."""
        flags = []
        if df.empty:
            return flags
            
        athlete_groups = df.groupby("Athlete")
        
        for athlete, group in athlete_groups:
            group = group.sort_values("Date")
            
            # High Fatigue Detection
            recent_fatigue = group["Fatigue_Score"].tail(3).mean()
            if recent_fatigue > 7.5:
                flags.append(f"⚠️ {athlete}: Sustained High Fatigue (Avg {recent_fatigue:.1f}/10)")
                
            # Training Load Spike (Acute:Chronic Workload Ratio approximation)
            if len(group) >= 5:
                recent_load = group["Training_Load"].iloc[-1]
                avg_load = group["Training_Load"].mean()
                if recent_load > avg_load * 1.4:
                    flags.append(f"📈 {athlete}: Training Load Spike (+{((recent_load/avg_load)-1)*100:.0f}% vs Avg)")
                    
            # Performance Improvement/Decline
            if len(group) >= 6:
                early_sprint = group["Sprint_Time_s"].head(3).mean()
                late_sprint = group["Sprint_Time_s"].tail(3).mean()
                if late_sprint < early_sprint * 0.97:
                    flags.append(f"🚀 {athlete}: Sprint Performance Improvement (-{((early_sprint-late_sprint)/early_sprint)*100:.1f}s)")
                elif late_sprint > early_sprint * 1.03:
                    flags.append(f"🔻 {athlete}: Sprint Performance Decline (+{((late_sprint-early_sprint)/early_sprint)*100:.1f}s)")
                    
            # Plateau Detection
            if len(group) >= 8:
                jump_std = group["Jump_Height_cm"].tail(6).std()
                if jump_std < 0.4:
                    flags.append(f"⏸️ {athlete}: Jump Height Performance Plateau Detected")
                    
        return flags[:8]  # Limit top flags for display

    @staticmethod
    def generate_rankings(df):
        """Generates a weighted performance composite score ranking."""
        if df.empty:
            return pd.DataFrame()
            
        agg = df.groupby("Athlete").agg({
            "Sprint_Time_s": "mean",
            "Jump_Height_cm": "mean",
            "Strength_kg": "mean",
            "Recovery_Pct": "mean",
            "Fatigue_Score": "mean"
        }).reset_index()
        
        # Min-max normalization for composite index
        sprint_score = (agg["Sprint_Time_s"].max() - agg["Sprint_Time_s"]) / (agg["Sprint_Time_s"].max() - agg["Sprint_Time_s"].min() + 1e-5)
        jump_score = (agg["Jump_Height_cm"] - agg["Jump_Height_cm"].min()) / (agg["Jump_Height_cm"].max() - agg["Jump_Height_cm"].min() + 1e-5)
        strength_score = (agg["Strength_kg"] - agg["Strength_kg"].min()) / (agg["Strength_kg"].max() - agg["Strength_kg"].min() + 1e-5)
        rec_score = agg["Recovery_Pct"] / 100.0
        
        agg["Composite_Score"] = ((sprint_score * 0.3) + (jump_score * 0.3) + (strength_score * 0.25) + (rec_score * 0.15)) * 100
        agg = agg.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
        agg.index += 1
        agg = agg.reset_index().rename(columns={"index": "Rank"})
        return agg


# ==========================================
# 3. Main GUI Application
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Performance Intelligence Dashboard")
        self.setGeometry(30, 30, 1550, 950)
        
        # Generate initial synthetic dataset
        self.raw_df = generate_synthetic_sports_data()
        self.filtered_df = self.raw_df.copy()
        
        self.init_ui()
        self.apply_filters()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Control Bar (Filters & Export Actions)
        filter_group = QGroupBox("Filter Panel & Actions")
        filter_layout = QHBoxLayout(filter_group)

        # Filters
        self.cmb_athlete = QComboBox()
        self.cmb_athlete.addItem("All Athletes")
        self.cmb_athlete.addItems(sorted(self.raw_df["Athlete"].unique()))
        self.cmb_athlete.currentTextChanged.connect(self.apply_filters)

        self.cmb_week = QComboBox()
        self.cmb_week.addItem("All Weeks")
        self.cmb_week.addItems([f"Week {w}" for w in sorted(self.raw_df["Week"].unique())])
        self.cmb_week.currentTextChanged.connect(self.apply_filters)

        self.cmb_phase = QComboBox()
        self.cmb_phase.addItem("All Phases")
        self.cmb_phase.addItems(list(self.raw_df["Training_Phase"].unique()))
        self.cmb_phase.currentTextChanged.connect(self.apply_filters)

        filter_layout.addWidget(QLabel("Athlete:"))
        filter_layout.addWidget(self.cmb_athlete)
        filter_layout.addWidget(QLabel("Week:"))
        filter_layout.addWidget(self.cmb_week)
        filter_layout.addWidget(QLabel("Phase:"))
        filter_layout.addWidget(self.cmb_phase)

        filter_layout.addStretch()

        # Action Buttons
        btn_csv = QPushButton("Export CSV")
        btn_csv.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 6px 12px;")
        btn_csv.clicked.connect(self.export_csv)

        btn_pdf = QPushButton("Export PDF Executive Report")
        btn_pdf.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 12px;")
        btn_pdf.clicked.connect(self.export_pdf_report)

        filter_layout.addWidget(btn_csv)
        filter_layout.addWidget(btn_pdf)

        main_layout.addWidget(filter_group)

        # Main Layout Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left Column: Visual Analytics Canvas
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        self.fig = Figure(figsize=(8, 9))
        self.canvas = FigureCanvas(self.fig)
        viz_layout.addWidget(self.canvas)
        splitter.addWidget(viz_widget)

        # Right Column: Dashboard Controls, Rankings & Insights
        dash_widget = QWidget()
        dash_layout = QVBoxLayout(dash_widget)
        self.tabs = QTabWidget()

        # Tab 1: Key Insights & Automated Flags
        tab_insights = QWidget()
        lay_insights = QVBoxLayout(tab_insights)
        
        lay_insights.addWidget(QLabel("<b>Automated Risk & Trend Flags:</b>"))
        self.txt_flags = QTextEdit()
        self.txt_flags.setReadOnly(True)
        lay_insights.addWidget(self.txt_flags)

        lay_insights.addWidget(QLabel("<b>Key Executive Observations:</b>"))
        self.txt_observations = QTextEdit()
        self.txt_observations.setReadOnly(True)
        lay_insights.addWidget(self.txt_observations)

        self.tabs.addTab(tab_insights, "Insights & Flags")

        # Tab 2: Synthetic Athlete Ranking Table
        tab_ranking = QWidget()
        lay_ranking = QVBoxLayout(tab_ranking)
        self.table_ranking = QTableWidget()
        lay_ranking.addWidget(self.table_ranking)
        self.tabs.addTab(tab_ranking, "Athlete Rankings")

        # Tab 3: Raw Data View
        tab_data = QWidget()
        lay_data = QVBoxLayout(tab_data)
        self.table_raw = QTableWidget()
        lay_data.addWidget(self.table_raw)
        self.tabs.addTab(tab_data, "Filtered Raw Data")

        dash_layout.addWidget(self.tabs)
        splitter.addWidget(dash_widget)

        splitter.setSizes([950, 600])
        main_layout.addWidget(splitter)

    def apply_filters(self):
        df = self.raw_df.copy()

        # Filter Athlete
        ath = self.cmb_athlete.currentText()
        if ath != "All Athletes":
            df = df[df["Athlete"] == ath]

        # Filter Week
        wk = self.cmb_week.currentText()
        if wk != "All Weeks":
            wk_num = int(wk.replace("Week ", ""))
            df = df[df["Week"] == wk_num]

        # Filter Phase
        ph = self.cmb_phase.currentText()
        if ph != "All Phases":
            df = df[df["Training_Phase"] == ph]

        self.filtered_df = df
        self.update_dashboard()

    def update_dashboard(self):
        self.render_charts()
        self.update_rankings()
        self.update_insights_and_flags()
        self.update_raw_table()

    def render_charts(self):
        self.fig.clear()
        df = self.filtered_df

        if df.empty:
            self.canvas.draw()
            return

        # 2x2 Plot Grid layout
        ax1 = self.fig.add_subplot(221)  # Training Load & Fatigue Trend
        ax2 = self.fig.add_subplot(222)  # Sprint Time vs Jump Height
        ax3 = self.fig.add_subplot(223)  # Recovery vs HRV
        ax4 = self.fig.add_subplot(224)  # Strength Distribution / Progression

        # Chart 1: Weekly Load & Fatigue
        weekly = df.groupby("Week").agg({"Training_Load": "mean", "Fatigue_Score": "mean"}).reset_index()
        ax1.plot(weekly["Week"], weekly["Training_Load"], marker='o', color="#1976D2", label="Avg Load")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(weekly["Week"], weekly["Fatigue_Score"], marker='s', color="#D32F2F", linestyle="--", label="Avg Fatigue")
        ax1.set_title("Weekly Load & Fatigue Trend", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Week")
        ax1.set_ylabel("Training Load", color="#1976D2")
        ax1_twin.set_ylabel("Fatigue (1-10)", color="#D32F2F")
        ax1.grid(True, alpha=0.3)

        # Chart 2: Performance Metrics (Sprint vs Jump)
        ax2.scatter(df["Jump_Height_cm"], df["Sprint_Time_s"], c=df["Fatigue_Score"], cmap="plasma", alpha=0.8)
        ax2.set_title("Jump Height vs Sprint Time (Color = Fatigue)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Jump Height (cm)")
        ax2.set_ylabel("Sprint Time (s)")
        ax2.grid(True, alpha=0.3)

        # Chart 3: Recovery vs HRV
        ax3.scatter(df["Recovery_Pct"], df["HRV_ms"], color="#388E3C", alpha=0.6)
        ax3.set_title("Recovery Pct vs HRV (ms)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Recovery (%)")
        ax3.set_ylabel("HRV (ms)")
        ax3.grid(True, alpha=0.3)

        # Chart 4: Strength Progression
        str_weekly = df.groupby("Week")["Strength_kg"].mean().reset_index()
        ax4.bar(str_weekly["Week"], str_weekly["Strength_kg"], color="#7B1FA2", alpha=0.7)
        ax4.set_title("Average Strength Progression (kg)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Week")
        ax4.set_ylabel("Strength (kg)")
        ax4.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_rankings(self):
        rankings = AnalyticsEngine.generate_rankings(self.filtered_df)
        self.table_ranking.clear()

        if rankings.empty:
            return

        self.table_ranking.setRowCount(len(rankings))
        self.table_ranking.setColumnCount(6)
        self.table_ranking.setHorizontalHeaderLabels(["Rank", "Athlete", "Composite Index", "Sprint Avg (s)", "Jump Avg (cm)", "Recovery Avg (%)"])

        for i, row in rankings.iterrows():
            idx = i - 1
            self.table_ranking.setItem(idx, 0, QTableWidgetItem(str(row["Rank"])))
            self.table_ranking.setItem(idx, 1, QTableWidgetItem(row["Athlete"]))
            self.table_ranking.setItem(idx, 2, QTableWidgetItem(f"{row['Composite_Score']:.1f}"))
            self.table_ranking.setItem(idx, 3, QTableWidgetItem(f"{row['Sprint_Time_s']:.2f}"))
            self.table_ranking.setItem(idx, 4, QTableWidgetItem(f"{row['Jump_Height_cm']:.1f}"))
            self.table_ranking.setItem(idx, 5, QTableWidgetItem(f"{row['Recovery_Pct']:.1f}%"))

        self.table_ranking.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def update_insights_and_flags(self):
        # Automated Risk Flags
        flags = AnalyticsEngine.detect_flags(self.filtered_df)
        self.txt_flags.clear()
        if flags:
            self.txt_flags.setText("\n".join(flags))
        else:
            self.txt_flags.setText("✅ No critical automated performance flags detected for current filter selection.")

        # Key Observations
        df = self.filtered_df
        if df.empty:
            self.txt_observations.setText("No data available.")
            return

        avg_load = df["Training_Load"].mean()
        avg_rec = df["Recovery_Pct"].mean()
        top_sprint = df.groupby("Athlete")["Sprint_Time_s"].mean().idxmin()
        top_jump = df.groupby("Athlete")["Jump_Height_cm"].mean().idxmax()

        obs = [
            f"1. **Load/Recovery Balance**: Team average training load is **{avg_load:.1f} AU** with a mean recovery rate of **{avg_rec:.1f}%**.",
            f"2. **Speed Benchmark**: **{top_sprint}** leads sprint performance across selected conditions.",
            f"3. **Power Benchmark**: **{top_jump}** exhibits the highest average vertical jump displacement.",
            f"4. **Cardiovascular Stress**: Mean session Heart Rate is **{df['HR_Avg_bpm'].mean():.0f} bpm** with an average HRV of **{df['HRV_ms'].mean():.1f} ms**.",
            f"5. **Fatigue Index**: Overall fatigue score averages **{df['Fatigue_Score'].mean():.1f}/10** across {len(df)} recorded sessions."
        ]
        self.txt_observations.setMarkdown("\n\n".join(obs))

    def update_raw_table(self):
        df = self.filtered_df
        self.table_raw.clear()
        if df.empty:
            return

        self.table_raw.setRowCount(min(100, len(df)))  # Limit display for performance
        self.table_raw.setColumnCount(len(df.columns))
        self.table_raw.setHorizontalHeaderLabels(df.columns)

        for r_idx, (_, row) in enumerate(df.head(100).iterrows()):
            for c_idx, val in enumerate(row):
                self.table_raw.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))

        self.table_raw.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Filtered Data CSV", "", "CSV Files (*.csv)")
        if path:
            self.filtered_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export Successful", f"Filtered dataset exported successfully to:\n{path}")

    def export_pdf_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Executive PDF Report", "", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            story = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#1976D2'),
                spaceAfter=12
            )

            story.append(Paragraph("Sports Performance Intelligence Executive Report", title_style))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Filter: {self.cmb_athlete.currentText()}", styles['Normal']))
            story.append(Spacer(1, 15))

            # Executive Insights Section
            story.append(Paragraph("Key Data-Driven Observations", styles['Heading2']))
            insights_text = self.txt_observations.toPlainText()
            for line in insights_text.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 4))

            story.append(Spacer(1, 15))

            # Performance Flags Section
            story.append(Paragraph("Automated Risk & Trend Flags", styles['Heading2']))
            flags_text = self.txt_flags.toPlainText()
            for line in flags_text.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 3))

            story.append(Spacer(1, 15))

            # Rankings Summary Table
            story.append(Paragraph("Top Athlete Composite Rankings", styles['Heading2']))
            rankings = AnalyticsEngine.generate_rankings(self.filtered_df).head(10)

            if not rankings.empty:
                table_data = [["Rank", "Athlete", "Composite Index", "Sprint (s)", "Jump (cm)"]]
                for _, r in rankings.iterrows():
                    table_data.append([
                        str(r["Rank"]),
                        str(r["Athlete"]),
                        f"{r['Composite_Score']:.1f}",
                        f"{r['Sprint_Time_s']:.2f}",
                        f"{r['Jump_Height_cm']:.1f}"
                    ])

                t = Table(table_data, colWidths=[50, 150, 120, 100, 100])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(t)

            doc.build(story)
            QMessageBox.information(self, "PDF Export Successful", f"Executive PDF report generated successfully at:\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to generate PDF report:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())