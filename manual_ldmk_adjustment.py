import json
import glob
import os

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import cv2
import numpy as np

app = dash.Dash(__name__)

all_ldmk_data = {}
all_images = {}

for file_path in glob.glob('extracted/ortho_landmarks/*.json'):
    file_name = os.path.basename(file_path)
    key = "-".join(file_name.split("-")[:2])
    dyn = file_name.split("-")[3].split("_")
    
    with open(file_path, 'r') as json_file:
        all_ldmk_data[key] = json.load(json_file)

for file_path in glob.glob('extracted/orthogonalized/*.png'):
    file_name = os.path.basename(file_path)
    key = "-".join(file_name.split("-")[:2])
    all_images[key] = "extracted/orthogonalized/"+file_name

# Load initial data
idx = "13-11"
image_path = all_images[idx]
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image_rgb.shape

app.layout = html.Div([
    dcc.Graph(
        id='image-graph',
        figure={
            'data': [
                go.Image(z=image_rgb),
                go.Scatter(
                    x=[p[0] for p in all_ldmk_data[idx]],
                    y=[p[1] for p in all_ldmk_data[idx]],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    name='Landmarks'
                )
            ],
            'layout': go.Layout(
                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'scaleanchor': "x", 'scaleratio': 1},
                margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                height=800,
                width=800,
                clickmode='event+select'
            )
        },
        style={'height': '90vh', 'width': '100%'}
    ),
    html.Button("Save Landmarks", id="save-button"),
    html.Div(id='debug-info'),
    dcc.Download(id="download-landmarks"),
    dcc.Store(id='landmarks-store', data=all_ldmk_data[idx]),
    dcc.Store(id='selected-landmark-index', data=None)
])

@app.callback(
    Output('debug-info', 'children'),
    [Input('selected-landmark-index', 'data')]
)
def update_debug_info(selected_index):
    return f"Selected Landmark Index: {selected_index}"

@app.callback(
    [Output('image-graph', 'figure'), Output('selected-landmark-index', 'data')],
    [Input('image-graph', 'clickData')],
    [State('landmarks-store', 'data'), State('selected-landmark-index', 'data'), State('image-graph', 'figure')]
)
def update_landmarks(clickData, landmarks, selected_index, fig):
    if clickData:

        curve_idx = clickData['points'][0]['curveNumber']
        data_type = fig['data'][curve_idx]['type']

        x, y = clickData['points'][0]['x'], clickData['points'][0]['y']
        distances = [(x - px) ** 2 + (y - py) ** 2 for px, py in landmarks]
        closest_idx = np.argmin(distances)

        if data_type == 'scatter':
            fig['data'][1]['x'] = [p[0] for p in landmarks]
            fig['data'][1]['y'] = [p[1] for p in landmarks]
            selected_index = closest_idx
        else:
            landmarks[selected_index] = (x, y)
            fig['data'][1]['x'] = [p[0] for p in landmarks]
            fig['data'][1]['y'] = [p[1] for p in landmarks]
            selected_index = None
        print(selected_index, (x, y))
        return fig, selected_index
    return dash.no_update, dash.no_update

@app.callback(
    Output('landmarks-store', 'data'),
    [Input('image-graph', 'figure')],
    [State('landmarks-store', 'data')]
)
def store_landmarks(fig, landmarks):
    landmarks = [(px, py) for px, py in zip(fig['data'][1]['x'], fig['data'][1]['y'])]
    return landmarks

@app.callback(
    Output('download-landmarks', 'data'),
    [Input('save-button', 'n_clicks')],
    [State('landmarks-store', 'data')]
)
def save_landmarks(n_clicks, landmarks):
    if n_clicks:
        path = 'landmarks.json'
        with open(path, 'w') as f:
            json.dump(landmarks, f)
        return dcc.send_file(path)
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
