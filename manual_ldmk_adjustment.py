import json
import glob
import os

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import cv2
import numpy as np

app = dash.Dash(__name__)

all_ldmk_data = {}
all_images = {}

for file_path in glob.glob(f'extracted/ortho_landmarks/*.json'):
    file_name = os.path.basename(file_path)
    key1 = "-".join(file_name.split("-")[:2])
    key2 = file_name.split("-")[-1].split("_")[1]
    key = key1 + "-" + key2
    dyn = file_name.split("-")[3].split("_")
    
    with open(file_path, 'r') as json_file:
        all_ldmk_data[key] = json.load(json_file)

for file_path in glob.glob(f'extracted/orthogonalized/*.png'):
    file_name = os.path.basename(file_path)
    key1 = "-".join(file_name.split("-")[:2])
    key2 = file_name.split("-")[-1].split("_")[1]
    key = key1 + "-" + key2
    all_images[key] = "extracted/orthogonalized/"+file_name

# Load initial data
# idx = "13-11"
# image_path = all_images[idx]
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# height, width, _ = image_rgb.shape

app.layout = html.Div([
    html.Div([  # Main container
        dcc.Graph(
            id='image-graph',
            figure={
                'data': [],
                'layout': go.Layout(
                    xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                    yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'scaleanchor': "x", 'scaleratio': 1},
                    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                    height=800,
                    # width='60vh',  
                    clickmode='event+select'
                )
            },
            style={'height': '90vh', 'width': '100%'}
        ),
        html.Div([  # Side panel for input and button
            dcc.Input(id='input-index', type='text', placeholder='Enter Image Index',
                      style={'margin': '5px', 'width': '250px', 'height': '40px'}),  
            html.Button('Load Image', id='load-button',
                        style={'margin': '5px', 'width': '150px', 'height': '45px'}),  
            html.Button('Save Updated Landmarks', id='save-updated-button',  
                style={'margin': '5px', 'width': '250px', 'height': '45px'}),
        ], style={
            'display': 'flex',
            'flexDirection': 'column',  
            'justifyContent': 'center',
            'alignItems': 'left',
            'width': '50%',  
            'height': '90vh'
        }),
    ], style={
        'display': 'flex',
        'alignItems': 'stretch',
        'width': '100%'  # Ensures the outer div takes full width of its parent
    }),
    html.Div(id='debug-info', style={
        'position': 'absolute',  # Absolute position
        'top': '10px',  
        'right': '10px',  
        'background': 'white',
        'padding': '10px',
        'border': '1px solid #ccc',
        'border-radius': '5px'
    }),
    dcc.Download(id="download-landmarks"),
    dcc.Store(id='landmarks-store'),
    dcc.Store(id='selected-landmark-index', data=None)
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh'})

@app.callback(
    [Output('image-graph', 'figure'),
     Output('landmarks-store', 'data'),
     Output('selected-landmark-index', 'data')],
    [Input('load-button', 'n_clicks'),
     Input('image-graph', 'clickData')],
    [State('input-index', 'value'),
     State('landmarks-store', 'data'),
     State('selected-landmark-index', 'data'),
     State('image-graph', 'figure')]
)
def update_content(load_clicks, clickData, input_idx, landmarks, selected_index, fig):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'load-button':
        if input_idx in all_images and input_idx in all_ldmk_data:
            image_path = all_images[input_idx]
            name = os.path.basename(image_path).split('.')[0]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = all_ldmk_data[input_idx]

            fig = {
                'data': [
                    go.Image(z=image_rgb),
                    go.Scatter(
                        x=[p[0] for p in landmarks],
                        y=[p[1] for p in landmarks],
                        mode='markers+text',
                        marker=dict(color='red', size=12),
                        text=[str(i) for i in range(len(landmarks))],
                        textposition="top center",
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
            }
            return fig, landmarks, dash.no_update  # Reset selected index only if needed
        else:
            raise PreventUpdate

    elif trigger_id == 'image-graph' and clickData:
        point_data = clickData['points'][0]
        curve_idx = point_data['curveNumber']
        x, y = point_data['x'], point_data['y']

        if curve_idx > 0:
            selected_index = int(point_data['text'])
        else:
            if selected_index:
                landmarks[selected_index] = (x, y)
                fig['data'][1]['x'] = [p[0] for p in landmarks]
                fig['data'][1]['y'] = [p[1] for p in landmarks]
            selected_index = None
        return fig, landmarks, selected_index

    raise PreventUpdate

@app.callback(
    Output('debug-info', 'children'),
    [Input('selected-landmark-index', 'data')]
)
def update_debug_info(selected_index):
    return f"Selected Landmark Index: {selected_index}"

@app.callback(
    Output('download-landmarks', 'data'),
    [Input('save-updated-button', 'n_clicks')],
    [State('input-index', 'value'),
     State('landmarks-store', 'data')]
)
def save_updated_landmarks(n_clicks, input_idx, landmarks):
    if n_clicks and landmarks:
        if input_idx in all_images:
            original_filename = os.path.basename(all_images[input_idx]).split('.')[0]
            updated_path = f'extracted/updated_landmarks/{original_filename}.json'

            # Ensure the directory exists
            os.makedirs(os.path.dirname(updated_path), exist_ok=True)

            # Save the updated landmarks to the specified file
            with open(updated_path, 'w') as f:
                json.dump(landmarks, f)
            
            # Optionally, if you want to allow downloading the saved file:
            return dcc.send_file(updated_path)
        else:
            return dash.no_update
    return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)