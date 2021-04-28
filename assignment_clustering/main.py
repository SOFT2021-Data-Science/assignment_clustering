from flask import Flask, render_template_string, send_from_directory, send_file
import os
from task import make_scatterplot, cluster_stuff, scatter_elbow, scatterplot_all_clustered, make_scatterplot_with_boundaries


app = Flask(__name__, static_url_path='/resources')

app.config['RESOURCE_FOLDER'] = "resources"

@app.route('/')
def index():
    return render_template_string( """
        <p>localhost:5000/all</p>
        <p>localhost:5000/elbow</p>
        <p>localhost:5000/scatter</p>
        <p>localhost:5000/boundaries</p>
        <p>localhost:5000/clusters</p>
    """)

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_from_directory(app.config['RESOURCE_FOLDER'], filename, as_attachment=False)
    except FileNotFoundError as ex:
        return "404" + ex
    

@app.route('/all')
def all():
    scatterplot_all_clustered()
    try:
        return send_file(app.config['RESOURCE_FOLDER']+"/scatterplot_with_all_clusters.jpeg", as_attachment=False)
    except FileNotFoundError as ex:
        return "404" + ex

@app.route('/elbow')
def elbow():
    scatter_elbow()
    try:
        return send_file(app.config['RESOURCE_FOLDER']+"/scatterplot_elbow.jpeg", as_attachment=False)
    except FileNotFoundError as ex:
        return "404" + ex

@app.route('/scatter')
def stuff():
    make_scatterplot()
    try:
        return send_file(app.config['RESOURCE_FOLDER']+"/scatterplot_1.jpeg", as_attachment=False)
    except FileNotFoundError as ex:
        return "404" + ex

@app.route('/boundaries')
def boundaries():
    make_scatterplot_with_boundaries()
    try:
        return send_file(app.config['RESOURCE_FOLDER']+"/scatterplot_with_boundaries.jpeg", as_attachment=False)
    except FileNotFoundError as ex:
        return "404" + ex

@app.route('/clusters')
def trash():
    cluster_stuff()

    html = """
        <img alt="" src="{{ url_for('download_file', filename="scatterplot_cluster_0.jpeg") }}"> 
        <img alt="" src="{{ url_for('download_file', filename="scatterplot_cluster_1.jpeg") }}"> 
        <img alt="" src="{{ url_for('download_file', filename="scatterplot_cluster_2.jpeg") }}"> 
        <img alt="" src="{{ url_for('download_file', filename="scatterplot_cluster_3.jpeg") }}"> 
    """

    return render_template_string(html)


if __name__ == "__main__":
    app.run()

# <img src="..resources/out_of_window.gif">
