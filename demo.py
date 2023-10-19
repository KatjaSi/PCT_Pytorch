import plotly.express as px
from agumentation import random_volume_crop_pc
from data import load_data

def main():

    train_points, _ = load_data("train")

    pc = train_points[100]
    
    rc = random_volume_crop_pc(pc, 0.3)
    fig = px.scatter_3d(x = rc[:,0], y = rc[:,1], z = rc[:,2])
    fig.write_html('rc.html')

if __name__ == '__main__':
    main()