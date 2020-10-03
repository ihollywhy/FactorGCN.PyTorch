import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dgl


def vis_graph(g, title="", save_name=None):
    if isinstance(g, nx.Graph):
        pass
    elif isinstance(g, np.ndarray):
        g = nx.DiGraph(g)
    elif isinstance(g, dgl.DGLGraph):
        g = g.to_networkx()
    else:
        raise NameError('unknow format of input graph')

    g = nx.Graph(g)
    g = nx.DiGraph(g)
    g = nx.to_numpy_matrix(g)
    np.fill_diagonal(g, 0.0)
    g = nx.DiGraph(g)
    # g.remove_edges_from(g.selfloop_edges())
    g.remove_nodes_from(list(nx.isolates(g)))

    nx.draw_networkx(g, arrows=False, with_labels=False, 
                        node_color="#fbb034", 
                        node_size=450,
                        width=4.5)  # networkx draw()
    plt.draw()  # pyplot draw()
    plt.title(title)
    plt.axis('off')
    if save_name is not None:
        plt.savefig(f"{save_name}.png", dpi=1000, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot(x=None, y=None):
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    plt.show()
    plt.close()