import argparse

def set_arg():
    parser = argparse.ArgumentParser(description='Hyperparameter',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--latent_dim', type=int, default=50, help='default dim of latent embedding')
    parser.add_argument('--hidden_dim1', type=int, default=50, help='first layer')
    parser.add_argument('--hidden_dim2', type=int, default=50, help='second layer')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='default weight decay of optimizer')
    parser.add_argument('--tol', type=int, default=0.)
    parser.add_argument('--init',type=str,default='mclust',help="Methods for initializing cluster centroids(kmeans or mclust)")
    parser.add_argument('--radius', type=int, default=50, help='radius KNN, stereo=50, Slide-seq=40')
    parser.add_argument('--knn', type=int, default=6, help='num of nearest neighbors')
    parser.add_argument('--n_domains', type=int, default=0, help='number of spatial domains')
    parser.add_argument('--mode_knn', type=str, default='knn', help='radius, knn')
    parser.add_argument('--update_interval',type=int,default='10')
    parser.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')

    return parser