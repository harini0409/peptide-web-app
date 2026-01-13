import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr
import warnings
import io
warnings.filterwarnings('ignore')

# Amino Acid Scales
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

BOMAN = {
    'A': 0.3, 'R': 1.0, 'N': 0.2, 'D': -0.1, 'C': 0.9,
    'Q': 0.2, 'E': -0.1, 'G': 0.0, 'H': 0.5, 'I': 1.8,
    'L': 1.8, 'K': 0.7, 'M': 1.3, 'F': 2.5, 'P': 0.0,
    'S': 0.1, 'T': 0.3, 'W': 2.6, 'Y': 2.2, 'V': 1.5
}

POSITIVE_AA = {'K', 'R', 'H'}
NEGATIVE_AA = {'D', 'E'}
POLAR_AA = {'R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T', 'Y'}

class PeptidePropertyCalculator:
    def calculate_net_charge(self, sequence):
        positive = sum(1 for aa in sequence if aa in POSITIVE_AA)
        negative = sum(1 for aa in sequence if aa in NEGATIVE_AA)
        return positive - negative
    
    def calculate_hydrophobicity(self, sequence):
        if not sequence:
            return 0
        values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
        return sum(values) / len(values)
    
    def calculate_boman_index(self, sequence):
        if not sequence:
            return 0
        values = [BOMAN.get(aa, 0) for aa in sequence]
        return sum(values) / len(values)
    
    def calculate_hydrophobic_moment(self, sequence, angle=100):
        if not sequence:
            return 0
        n = len(sequence)
        angle_rad = np.radians(angle)
        sum_sin = sum(HYDROPHOBICITY.get(sequence[i], 0) * np.sin(i * angle_rad) for i in range(n))
        sum_cos = sum(HYDROPHOBICITY.get(sequence[i], 0) * np.cos(i * angle_rad) for i in range(n))
        return np.sqrt(sum_sin**2 + sum_cos**2) / n
    
    def calculate_rw_propensity(self, sequence):
        if len(sequence) < 2:
            return 0
        rw_count = sum(1 for i in range(len(sequence)-1)
                      if (sequence[i:i+2] == 'RW' or sequence[i:i+2] == 'WR'))
        return rw_count / (len(sequence) - 1)
    
    def calculate_solubility_index(self, sequence):
        if not sequence:
            return 0
        polar_count = sum(1 for aa in sequence if aa in POLAR_AA)
        return polar_count / len(sequence)
    
    def calculate_instability_index(self, sequence):
        if not sequence:
            return 0
        unstable_aa = {'D', 'E', 'P', 'G', 'S'}
        unstable_count = sum(1 for aa in sequence if aa in unstable_aa)
        return 40 * (unstable_count / len(sequence))
    
    def calculate_all_properties(self, sequence):
        sequence = sequence.upper().strip()
        props = {
            'sequence': sequence,
            'length': len(sequence),
            'net_charge': self.calculate_net_charge(sequence),
            'hydrophobicity': self.calculate_hydrophobicity(sequence),
            'boman_index': self.calculate_boman_index(sequence),
            'hydrophobic_moment': self.calculate_hydrophobic_moment(sequence),
            'gravy_score': self.calculate_hydrophobicity(sequence),
            'instability_index': self.calculate_instability_index(sequence),
            'rw_propensity': self.calculate_rw_propensity(sequence),
            'solubility_index': self.calculate_solubility_index(sequence)
        }
        return props

class PeptideFileParser:
    @staticmethod
    def parse_csv(content):
        df = pd.read_csv(io.StringIO(content))
        seq_columns = ['sequence', 'peptide', 'seq', 'Sequence', 'Peptide']
        seq_col = None
        for col in seq_columns:
            if col in df.columns:
                seq_col = col
                break
        if seq_col is None:
            seq_col = df.columns[0]
        id_columns = ['id', 'ID', 'name', 'Name', 'peptide_id', 'seq_id']
        id_col = None
        for col in id_columns:
            if col in df.columns:
                id_col = col
                break
        sequences = []
        seq_ids = []
        for idx, row in df.iterrows():
            seq = str(row[seq_col]).strip()
            if seq and seq != 'nan':
                sequences.append(seq)
                if id_col:
                    seq_ids.append(str(row[id_col]))
                else:
                    seq_ids.append(f"SEQ_{idx+1}")
        return sequences, seq_ids
    
    @staticmethod
    def parse_txt(content):
        sequences = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        seq_ids = [f"SEQ_{i+1}" for i in range(len(sequences))]
        return sequences, seq_ids
    
    @staticmethod
    def parse_fasta(content):
        sequences = []
        seq_ids = []
        current_seq = []
        current_id = None
        seq_counter = 1
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    seq_ids.append(current_id if current_id else f"SEQ_{seq_counter}")
                    seq_counter += 1
                    current_seq = []
                current_id = line[1:].split()[0] if len(line) > 1 else f"SEQ_{seq_counter}"
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq))
            seq_ids.append(current_id if current_id else f"SEQ_{seq_counter}")
        return sequences, seq_ids

def generate_training_data(n_samples=2000):
    calculator = PeptidePropertyCalculator()
    amino_acids = list(HYDROPHOBICITY.keys())
    data = []
    for _ in range(n_samples):
        length = np.random.randint(5, 51)
        sequence = ''.join(np.random.choice(amino_acids, length))
        props = calculator.calculate_all_properties(sequence)
        score = (
            0.15 * props['boman_index'] +
            0.15 * props['hydrophobic_moment'] +
            0.10 * abs(props['net_charge']) +
            0.15 * props['solubility_index'] +
            0.15 * props['rw_propensity'] +
            0.10 * (1 - props['instability_index']/100) +
            0.10 * (props['hydrophobicity'] + 5) / 10 +
            0.10 * (props['length'] / 50)
        )
        props['consolidated_score'] = score
        data.append(props)
    return pd.DataFrame(data)

class PeptideScreeningModel:
    def __init__(self):
        self.calculator = PeptidePropertyCalculator()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'length', 'net_charge', 'hydrophobicity', 'boman_index',
            'hydrophobic_moment', 'gravy_score', 'instability_index',
            'rw_propensity', 'solubility_index'
        ]
    
    def train(self, n_samples=2000):
        training_data = generate_training_data(n_samples)
        X = training_data[self.feature_columns]
        y = training_data['consolidated_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse
    
    def predict_and_rank(self, sequences, seq_ids, top_n=50):
        results = []
        for seq, seq_id in zip(sequences, seq_ids):
            try:
                props = self.calculator.calculate_all_properties(seq)
                features = [props[col] for col in self.feature_columns]
                features_scaled = self.scaler.transform([features])
                score = self.model.predict(features_scaled)[0]
                props['predicted_score'] = score
                props['sequence_id'] = seq_id
                results.append(props)
            except:
                continue
        if not results:
            return pd.DataFrame()
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('predicted_score', ascending=False)
        df_results['rank'] = range(1, len(df_results) + 1)
        return df_results.head(top_n)

def create_plots(results_df):
    if len(results_df) == 0:
        return None, None, None, None, None
    n_show = min(20, len(results_df))
    top_n = results_df.head(n_show)
    fig1 = go.Figure(data=[go.Bar(
        x=top_n['predicted_score'], y=top_n['sequence_id'], orientation='h',
        marker=dict(color=top_n['predicted_score'], colorscale='Viridis', showscale=True, colorbar=dict(title="Score")),
        text=top_n['predicted_score'].round(4), textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<br>Rank: %{customdata}<extra></extra>',
        customdata=top_n['rank']
    )])
    fig1.update_layout(
        title=dict(text=f"<b>Top {n_show} Ranked Peptides</b><br><sub>Higher scores = better potential</sub>", x=0.5, xanchor='center'),
        xaxis_title="Predicted Score", yaxis_title="Sequence ID", height=600, yaxis=dict(autorange="reversed")
    )
    fig2 = go.Figure(data=[go.Scatter3d(
        x=results_df['hydrophobicity'], y=results_df['boman_index'], z=results_df['solubility_index'],
        mode='markers', marker=dict(size=6, color=results_df['predicted_score'], colorscale='Plasma', showscale=True,
        colorbar=dict(title="Score"), line=dict(width=0.5, color='white')),
        text=results_df['sequence_id'],
        hovertemplate='<b>%{text}</b><br>Hydrophobicity: %{x:.2f}<br>Boman: %{y:.2f}<br>Solubility: %{z:.2f}<extra></extra>'
    )])
    fig2.update_layout(
        title=dict(text="<b>3D Property Space</b><br><sub>Hydrophobicity, Boman Index, Solubility</sub>", x=0.5, xanchor='center'),
        scene=dict(xaxis_title="Hydrophobicity", yaxis_title="Boman Index", zaxis_title="Solubility"), height=600
    )
    n_radar = min(5, len(results_df))
    top_5 = results_df.head(n_radar)
    categories = ['Net Charge', 'Hydrophobicity', 'Boman', 'Moment', 'Solubility', 'RW']
    fig3 = go.Figure()
    for idx, row in top_5.iterrows():
        values = [abs(row['net_charge'])/5, (row['hydrophobicity']+5)/10, row['boman_index']/3, 
                  row['hydrophobic_moment']/2, row['solubility_index'], row['rw_propensity']*10]
        fig3.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', 
                                       name=f"{row['sequence_id']} (Rank {int(row['rank'])})"))
    fig3.update_layout(
        title=dict(text=f"<b>Top {n_radar} Property Profiles</b><br><sub>Normalized 0-1 scale</sub>", x=0.5, xanchor='center'),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=600
    )
    props = ['net_charge', 'hydrophobicity', 'boman_index', 'hydrophobic_moment', 
             'instability_index', 'rw_propensity', 'solubility_index', 'predicted_score']
    prop_labels = ['Net Charge', 'Hydrophobicity', 'Boman', 'Moment', 'Instability', 'RW', 'Solubility', 'Score']
    corr = results_df[props].corr()
    fig4 = go.Figure(data=go.Heatmap(
        z=corr.values, x=prop_labels, y=prop_labels, colorscale='RdBu', zmid=0,
        text=corr.values.round(2), texttemplate='%{text}', textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig4.update_layout(
        title=dict(text="<b>Property Correlations</b><br><sub>Blue=positive, Red=negative</sub>", x=0.5, xanchor='center'),
        height=600, xaxis=dict(tickangle=-45)
    )
    fig5 = go.Figure(data=[go.Scatter(
        x=results_df['length'], y=results_df['predicted_score'], mode='markers',
        marker=dict(size=8, color=results_df['predicted_score'], colorscale='Turbo', showscale=True,
                   colorbar=dict(title="Score"), line=dict(width=0.5, color='white')),
        text=results_df['sequence_id'],
        hovertemplate='<b>%{text}</b><br>Length: %{x} aa<br>Score: %{y:.4f}<extra></extra>'
    )])
    fig5.update_layout(
        title=dict(text="<b>Length vs Score</b><br><sub>Optimal length analysis</sub>", x=0.5, xanchor='center'),
        xaxis_title="Length (amino acids)", yaxis_title="Predicted Score", height=600
    )
    return fig1, fig2, fig3, fig4, fig5

print("Training model...")
screening_model = PeptideScreeningModel()
r2, rmse = screening_model.train(n_samples=2000)
print(f"Model trained: R2={r2:.4f}, RMSE={rmse:.4f}")

def process_file(file, top_n):
    try:
        if file is None:
            return "Please upload a file", None, None, None, None, None, None, None
        content = file.decode('utf-8') if isinstance(file, bytes) else open(file.name, 'r').read()
        if 'csv' in str(file):
            sequences, seq_ids = PeptideFileParser.parse_csv(content)
        elif content.strip().startswith('>'):
            sequences, seq_ids = PeptideFileParser.parse_fasta(content)
        else:
            sequences, seq_ids = PeptideFileParser.parse_txt(content)
        if not sequences:
            return "No sequences found", None, None, None, None, None, None, None
        results = screening_model.predict_and_rank(sequences, seq_ids, top_n=int(top_n))
        if len(results) == 0:
            return "Could not process sequences", None, None, None, None, None, None, None
        fig1, fig2, fig3, fig4, fig5 = create_plots(results)
        table = results[['rank', 'sequence_id', 'sequence', 'predicted_score', 'length', 'net_charge', 
                        'hydrophobicity', 'boman_index', 'solubility_index', 'hydrophobic_moment']].copy()
        for col in ['predicted_score', 'net_charge', 'hydrophobicity', 'boman_index', 'solubility_index', 'hydrophobic_moment']:
            table[col] = table[col].round(4)
        table = table.rename(columns={'rank': 'Rank', 'sequence_id': 'Sequence ID', 'sequence': 'Peptide Sequence',
            'predicted_score': 'Score', 'length': 'Length (aa)', 'net_charge': 'Net Charge',
            'hydrophobicity': 'Hydrophobicity', 'boman_index': 'Boman Index',
            'solubility_index': 'Solubility', 'hydrophobic_moment': 'Hydrophobic Moment'})
        csv_path = 'peptide_screening_results.csv'
        results.to_csv(csv_path, index=False)
        top_peptide = results.iloc[0]
        summary = f"""
PEPTIDE SCREENING ANALYSIS COMPLETE

DATASET OVERVIEW:
• Total Sequences Analyzed: {len(sequences)}
• Successfully Processed: {len(results)}
• Top Peptides Displayed: {min(len(results), int(top_n))}

TOP-RANKED PEPTIDE:
• Sequence ID: {top_peptide['sequence_id']}
• Sequence: {top_peptide['sequence']}
• Predicted Score: {top_peptide['predicted_score']:.4f}
• Length: {top_peptide['length']} amino acids

KEY PROPERTIES:
• Net Charge: {top_peptide['net_charge']:.2f}
• Hydrophobicity: {top_peptide['hydrophobicity']:.4f}
• Boman Index: {top_peptide['boman_index']:.4f}
• Solubility: {top_peptide['solubility_index']:.4f}
• Hydrophobic Moment: {top_peptide['hydrophobic_moment']:.4f}

STATISTICS (Top {min(len(results), int(top_n))}):
• Mean Score: {results['predicted_score'].mean():.4f}
• Median Score: {results['predicted_score'].median():.4f}
• Score Range: {results['predicted_score'].min():.4f} - {results['predicted_score'].max():.4f}

INTERPRETATION:
• Higher Score = Better therapeutic potential
• Net Charge: Membrane interaction
• Hydrophobicity: Membrane affinity
• Boman Index: Protein binding
• Solubility: Water compatibility

Results saved to: {csv_path}
"""
        return summary, table, fig1, fig2, fig3, fig4, fig5, csv_path
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None, None, None, None

with gr.Blocks(title="Peptide Screening ML", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Peptide Screening ML Platform\n### Machine Learning-Powered Peptide Analysis\nSupports CSV, TXT, and FASTA formats")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Configuration")
            file_input = gr.File(label="Upload Sequences", file_types=['.csv', '.txt', '.fasta', '.fa'], type="binary")
            gr.Markdown("**Formats:** CSV (with 'sequence' column), FASTA (with headers), TXT (one per line)")
            top_n = gr.Slider(10, 100, value=50, step=10, label="Top Peptides to Display")
            btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
        with gr.Column(scale=2):
            gr.Markdown("### Analysis Summary")
            summary = gr.Textbox(label="Results", lines=25, show_label=False)
    gr.Markdown("## Results Table")
    table = gr.Dataframe(label="Top Peptides with Sequence IDs", wrap=True)
    gr.Markdown("## Interactive Visualizations")
    with gr.Tabs():
        with gr.Tab("🏆 Top Scores"):
            gr.Markdown("Bar chart of top-ranked peptides with Sequence IDs. Hover for details.")
            plot1 = gr.Plot()
        with gr.Tab("🎯 3D Space"):
            gr.Markdown("3D visualization of hydrophobicity, Boman index, and solubility. Each point is labeled with Sequence ID.")
            plot2 = gr.Plot()
        with gr.Tab("⭐ Profiles"):
            gr.Markdown("Radar chart comparing top 5 peptides across normalized properties. Larger shapes indicate balanced properties.")
            plot3 = gr.Plot()
        with gr.Tab("🔗 Correlations"):
            gr.Markdown("Heatmap showing property relationships. Blue=positive, Red=negative correlation.")
            plot4 = gr.Plot()
        with gr.Tab("📏 Length"):
            gr.Markdown("Scatter plot of sequence length vs predicted score. Hover to see Sequence IDs.")
            plot5 = gr.Plot()
    download = gr.File(label="Download Full Results (CSV)")
    btn.click(process_file, inputs=[file_input, top_n], outputs=[summary, table, plot1, plot2, plot3, plot4, plot5, download])

if __name__ == "__main__":
    demo.launch(share=True, ssr_mode=False)