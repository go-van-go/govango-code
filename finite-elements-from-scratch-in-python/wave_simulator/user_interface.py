import os
import glob
import pickle
import toml
import panel as pn
import matplotlib.pyplot as plt
import pyvista as pv

from pathlib import Path
from wave_simulator.visualizer import Visualizer

pn.extension('vtk')

class UserInterface:
    def __init__(self, outputs_dir='./outputs'):
        self.outputs_dir = outputs_dir
        self.sim_folders = self._get_sim_folders()

        # Add a blank option at the start
        sim_options = [""] + self.sim_folders


        self.selected_folder = None
        self.data_files = []
        self.visualizer = None

        self.sim_selector = pn.widgets.Select(name='Simulation Run', options=sim_options, value="")
        #self.frame_slider = pn.widgets.IntSlider(name='Time', start=0, end=0, step=1)
        self.frame_selector = pn.widgets.Select(name='Timestep', options=[])
        self.refresh_button = pn.widgets.Button(name='Load Data', button_type='primary')
        self.status_text = pn.pane.HTML("", height=20)

        self.sim_selector.param.watch(self._update_folder, 'value')
        self.refresh_button.on_click(self._load_frame)

        self.parameters_pane = pn.pane.HTML("<i>No parameters loaded.</i>", width=300)
        self.runtime_pane = pn.pane.HTML("", width=300)

        self.show_3d_button = pn.widgets.Button(name='Show 3D', button_type='success')
        self.show_3d_button.on_click(self._show_3d)

        self.content = pn.Row()
        self.panel = pn.Row(
            pn.Column(
                pn.pane.HTML("<h2>Simulation Viewer</h2>"),
                self.sim_selector,
                self.frame_selector,
                self.refresh_button,
                self.show_3d_button,
                self.status_text,
                pn.pane.HTML("<h3>Simulation Parameters</h3>"),
                self.parameters_pane,
                pn.pane.HTML("<h3>Runtime</h3>"),
                self.runtime_pane,
            ),
            pn.Column(
                self.content
            )
        )


    def _get_sim_folders(self):
        return sorted([
            f for f in os.listdir(self.outputs_dir)
            if os.path.isdir(os.path.join(self.outputs_dir, f))
        ])

    def _update_folder(self, event):
        if not event.new:
            # Nothing selected, clear data
            self.selected_folder = None
            self.data_files = []
            self.status_text.object = "<span style='color:gray'>Select a simulation run.</span>"
            self.frame_selector.options = []
            self.frame_selector.value = None
            self.refresh_button.disabled = True
            return
        self.selected_folder = os.path.join(self.outputs_dir, event.new)
        data_dir = os.path.join(self.selected_folder, "data")
        self.data_files = sorted(glob.glob(f"{data_dir}/*.pkl"))

        if not self.data_files:
            self.status_text.object = f"<span style='color:red'>⚠️ No .pkl files found in: {data_dir}</span>"
            self.frame_selector.options = []
            self.frame_selector.value = None
            self.refresh_button.disabled = True
        else:
            self.status_text.object = f"<span style='color:green'>✅ Found {len(self.data_files)} data files.</span>"
            # Build a mapping of display label → file path
            self.timestep_map = {}
            options = []
            for path in self.data_files:
                time_str = os.path.basename(path).split("_t")[-1].split(".")[0]
                label = f"t = {time_str}"
                self.timestep_map[label] = path
                options.append(label)
            
            self.frame_selector.options = options
            self.frame_selector.value = options[0]

            self.refresh_button.disabled = False

        param_file = os.path.join(self.selected_folder, "parameters.toml")
        if os.path.exists(param_file):
            html = self._format_parameters(param_file)
            self.parameters_pane.object = html
        else:
            self.parameters_pane.object = "<i>⚠️ No parameters.toml found.</i>"



    def _format_parameters(self, file_path):
        try:
            parameters = toml.load(file_path)
        except Exception as e:
            return f"<b>❌ Failed to load parameters.toml:</b> {e}"
    
        html = ["<div style='font-family: monospace; font-size: 12px;'>"]
        for section, values in parameters.items():
            html.append(f"<h4>[{section}]</h4><ul style='margin-top: 0;'>")
            for key, value in values.items():
                html.append(f"<li><b>{key}</b>: {value}</li>")
            html.append("</ul>")
        html.append("</div>")
        return "\n".join(html)

    def _show_3d(self, event=None):
        if self.visualizer is None:
            self.status_text.object = "<span style='color:red'>⚠️ Load data before showing 3D.</span>"
            return
        try:
            self.visualizer.add_nodes_3d("p")
            self.visualizer.add_inclusion_boundary()
            self.visualizer.show()
            self.status_text.object = "<span style='color:green'>✅ 3D view launched.</span>"
        except Exception as e:
            self.status_text.object = f"<span style='color:red'>❌ Error in 3D view: {e}</span>"

    def _load_frame(self, event=None):
        if not self.data_files:
            self.status_text.object = "<span style='color:red'>⚠️ No data loaded.</span>"
            return
    
        try:
            data_path = self.timestep_map[self.frame_selector.value]
    
            # Load timestep-specific data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
    
            # Load mesh data
            mesh_data = data['mesh_directory'] / "mesh.pkl"
            if mesh_data.exists():
                with open(mesh_data, 'rb') as mf:
                    mesh_data = pickle.load(mf)
            else:
                self.status_text.object = "<span style='color:red'>❌ mesh.pkl not found.</span>"
                return
    
            self.visualizer = Visualizer(mesh_data, data)
            tracked_fig = self.visualizer.plot_tracked_points()
            energy_fig = self.visualizer.plot_energy()
            # Try to load runtime from the first available data file
            # Append runtime info to parameters pane
            runtime_sec = data.get("runtime", None)
            if runtime_sec is not None:
                hours = int(runtime_sec // 3600)
                minutes = int((runtime_sec % 3600) // 60)
                seconds = int(runtime_sec % 60)
                runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                self.runtime_pane.object = f"""
                <div style='font-family: monospace; font-size: 12px;'>
                <b>Elapsed Time</b>: {runtime_str} (hh:mm:ss)
                </div>
                """
            else:
                self.runtime_pane.object = "<i>⚠️ Runtime not recorded.</i>"


            # Get image corresponding to current timestep
            time_step = int(os.path.basename(data_path).split("_t")[-1].split(".")[0])
            image_path = os.path.join(self.selected_folder, "images", f"t_{time_step:08d}.png")
    
            image_pane = (
                pn.pane.PNG(image_path, width=550)
                if os.path.exists(image_path)
                else pn.pane.Markdown("**⚠️ No image found for this timestep.**")
            )
    
            layout = pn.Row(
                pn.Column(
                    pn.pane.HTML("<b>Receivers</b>"),
                    pn.pane.Matplotlib(tracked_fig, tight=True, height=900),
                ),
                pn.Column(
                    pn.pane.HTML("<b>Energy Plot</b>"),
                    pn.pane.Matplotlib(energy_fig, tight=True, width=550),
                    pn.pane.HTML("<b>Snapshot</b>"),
                    image_pane,
                )
            )
    
            self.content.objects = [layout]
            self.status_text.object = f"<span style='color:green'>✅ Loaded frame: {os.path.basename(data_path)}</span>"
        except Exception as e:
            self.status_text.object = f"<span style='color:red'>❌ Error loading frame: {e}</span>"

    def show(self):
        return self.panel
