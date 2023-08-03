import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from itertools import cycle

# Create a cycle of colors
color_cycle = cycle(px.colors.qualitative.Plotly)
color_map = {}

def plot_dataframe(df, column, legend_name, row, col, line_color):
    if col == 1:
        subplot_title = 'Train Accuracy' if row == 1 else 'Validation Accuracy'
    elif col == 2:
        subplot_title = 'Train Loss' if row == 1 else 'Validation Loss'
    else:
        subplot_title = 'Train Perplexity' if row == 1 else 'Validation Perplexity'

    showlegend = True if row == 1 and col == 1 else False

    fig.add_trace(go.Scatter(x=df['step'], y=df[column], mode='lines', name=f'{legend_name}',
                             line=dict(width=3, color=line_color), showlegend=showlegend), row=row, col=col)

# Get the current directory
current_dir = os.getcwd()
folder = os.path.join(current_dir)

# Create a subplot for the figure with 2 rows and 3 columns
fig = make_subplots(rows=2, cols=3, subplot_titles=('Train Accuracy', 'Train Loss', 'Train Perplexity', 'Validation Accuracy', 'Validation Loss', 'Validation Perplexity'), vertical_spacing=0.1, horizontal_spacing=0.05)
# Get existing annotations (subplot titles)
annotations = list(fig['layout']['annotations'])

# Add new annotations for column labels
annotations += [
    dict(x=0.15, y=-0.1, xref="paper", yref="paper", text="(a)", showarrow=False, font=dict(size=16)),
    dict(x=0.5, y=-0.1, xref="paper", yref="paper", text="(b)", showarrow=False, font=dict(size=16)),
    dict(x=0.85, y=-0.1, xref="paper", yref="paper", text="(c)", showarrow=False, font=dict(size=16))
]

# List of selected files

#=================================+>
# for 16:

selected_files=['scalars_512_16_4_2_0.1.csv',
                'scalars_512_16_4_4_0.1.csv',
                'scalars_512_16_8_2_0.5.csv',
                'scalars_512_16_8_2_0.1.csv']

# selected_files=[]
# not showing since, did not make sense
# selected_files=['scalars_512_16_4_4_0.5.csv']  # not plotted. 

# selected_files=['scalars_512_16_4_2_0.5.csv']  #unstable and overfitting  # combined below.. with 512_32_4_2_0.5
#===============================+>
# for 32

selected_files = ['scalars_512_32_8_2_0.1.csv','scalars_512_32_4_4_0.1.csv',
                  'scalars_512_32_4_2_0.1.csv']

# selected_files = ['scalars_512_32_4_4_0.5.csv']  # do not need this do not plot
selected_files=['scalars_512_32_4_2_0.5.csv','scalars_512_16_4_2_0.5.csv','scalars_512_32_8_2_0.5.csv']   # overfitting and unstable

# selected_files=['scalars_512_32_8_2_0.5.csv']  # overfitting and unstable  # plotted abvoe. 
#===============================+>
# for 64

selected_files = ['scalars_512_64_4_8_0.4couldbeincomplete.csv',
'scalars_512_64_4_8_0.5.csv',
'scalars_512_64_8_8_0.5unstable so stopped.csv',
'scalars_512_64_16_4_0.5.csv']

selected_files=['scalars_512_64_4_2_0.4.csv',
'scalars_512_64_4_2_0.3.csv',
'scalars_512_64_4_4_0.4.csv',
'scalars_512_64_4_4_0.3.csv']

selected_files=['scalars_512_64_4_2_0.5.csv']
#==============================>
# For 128
selected_files=['scalars_512_128_4_2_0.5.csv','scalars_512_128_8_2_0.5.csv']

# with more number of points.. ==> have not yet included in overleaf document.

selected_files=['scalars_512_128_4_4_0.1new.csv',
                'scalars_512_512_4_2_0.1.csv',
                'scalars_512_64_4_4_0.csv','scalars_512_128_4_4_0.csv']

# selected_files=[
# 'scalars_512_128_8_4_0.5.csv',
# 'scalars_512_128_4_8_0.5.csv',
# 'scalars_512_128_4_4_0.5.csv']



#=======================+>
# for 256
# selected_files=[

# 'scalars_512_256_4_4_0.4.csv',

# 'scalars_512_256_4_8_0.5_70epochs mostly.csv',

# 'scalars_512_256_4_16_0.5.csv',
# 'scalars_512_256_8_16_0.5.csv']

# selected_files = [
#     'scalars_512_256_16_8_0.5.csv','scalars_512_256_4_16_0.5.csv']
# selected_files = ['scalars_512_256_4_16_0.5.csv']   # THIS IS IN UPPER ONE. DO NOT PLOT IT SEPARATELR

# selected_files=['scalars_512_256_4_2_0.4.csv',
# 'scalars_512_256_4_2_0.3.csv',
# 'scalars_512_256_4_2_0.5.csv']

#=====================>
# for 512

# selected_files =['scalars_512_512_4_2_0.5.csv','scalars_512_512_4_2_0.4.csv',
#                  'scalars_512_512_4_2_0.3incomplete.csv']
# selected_files =['scalars_512_512_4_4_0.5stopped because of high perplexity.csv',
#                  'scalars_512_512_4_8_0.5stopped_because_of_high error.csv',
#                  'scalars_512_512_8_8_0.5.csv','scalars_512_512_32_16_0.5.csv']


#============+>
# for 0.1 dropout
# selected_files = ['scalars_512_128_4_2_0.1.csv','scalars_512_128_4_4_0.1.csv',
#                   'scalars_512_256_4_2_0.1.csv','scalars_512_256_4_4_0.1.csv',
#                   'scalars_512_128_4_2_0.5.csv','scalars_512_128_4_4_0.5.csv',
#                   'scalars_512_256_4_2_0.5.csv','scalars_512_256_4_4_0.4.csv']

# selected_files = ['scalars_512_128_4_2_0.1.csv','scalars_512_128_4_4_0.1.csv',
#                   'scalars_512_256_4_2_0.1.csv','scalars_512_256_4_4_0.1.csv',
#                   'scalars_512_256_4_2_0.5.csv']

# Iterate over selected csv files
for filename in selected_files:
    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        # legend_name = filename.split('.')[0]
        legend_name = os.path.splitext(filename)[0]

        if legend_name not in color_map:
            color_map[legend_name] = next(color_cycle)

        line_color = color_map[legend_name]

        # Training plots
        plot_dataframe(df, 'Train Accuracy', legend_name, 1, 1, line_color)
        plot_dataframe(df, 'Train Loss', legend_name, 1, 2, line_color)
        plot_dataframe(df, 'Train Perplexity', legend_name, 1, 3, line_color)
        
        # Validation plots
        plot_dataframe(df, 'Validation Accuracy', legend_name, 2, 1, line_color)
        plot_dataframe(df, 'Validation Loss', legend_name, 2, 2, line_color)
        plot_dataframe(df, 'Validation Perplexity', legend_name, 2, 3, line_color)
    else:
        print(f"{filename} does not exist in the directory")

# Update layout properties
fig.update_layout(showlegend=True,
                  legend=dict(
                      yanchor="top",
                      y=1.35,
                      xanchor="center",
                      x=1.05,
                      font=dict(size=16),
                      orientation="v"
                  ),
                  autosize=False,
                  width=1600,
                  height=800,
                  margin=dict(
                      l=50,
                      r=50,
                      b=100,
                      t=100,
                      pad=4
                  ),
                  paper_bgcolor="White",
                  annotations=annotations
)
fig.update_xaxes(tickfont=dict(size=14),title_font=dict(size=18))  # Adjust size as needed
fig.update_yaxes(tickfont=dict(size=14),title_font=dict(size=18)) 
# Show the plot
fig.show()

# Separate plot for Epoch Duration vs. Step
fig_epoch_duration = go.Figure()
for filename in selected_files:
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        # Get the legend name by using os.path.splitext
        legend_name = os.path.splitext(filename)[0]
        line_color = color_map[legend_name]

        # Make sure the 'Epoch Duration' column exists in the dataframe
        if 'Epoch Duration' in df.columns:
            # Convert the epoch duration from seconds to minutes
            df['Epoch Duration'] = df['Epoch Duration'] / 60
            # Calculate the accumulated time for each step
            df['Accumulated Time'] = df['Epoch Duration'].cumsum()
            fig_epoch_duration.add_trace(go.Scatter(x=df['step'], y=df['Accumulated Time'], mode='lines', name=f'{legend_name}',
                                                    line=dict(width=2, color=line_color)))
        else:
            print(f"'Epoch Duration' column not found in {filename}")

# Update layout properties for Epoch Duration vs. Step
fig_epoch_duration.update_layout(title="Accumulated Time vs. Step",
                                 xaxis_title="Epochs",
                                 yaxis_title="Accumulated Time (minutes)",
                                 autosize=False,
                                 width=1200,
                                 height=500,
                                 legend=dict(
                                     font=dict(size=16), # Increase the font size
                                     orientation="v"    # Set orientation to vertical
                                 ),
                                 paper_bgcolor="White")  # Increase the size of subplot titles)
# Increase the font size of the x-axis and y-axis labels
fig_epoch_duration.update_xaxes(tickfont=dict(size=14),title_font=dict(size=18))  # Adjust size as needed
fig_epoch_duration.update_yaxes(tickfont=dict(size=14),title_font=dict(size=18))  # Adjust size as needed
# Add these lines after the rest of your code for setting up the layout
# for i, title in enumerate(fig.layout.annotations):
#     title.font.size = 20  # Increase the font size to 16

fig_epoch_duration.show()