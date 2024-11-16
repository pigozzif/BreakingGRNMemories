import io

import jax
import numpy as np
import pygame
from matplotlib import pyplot as plt

from grn import GeneRegulatoryNetwork
from utils import parse_args, set_seed


def create_window(title, width=900, height=600):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    screen.fill((255, 255, 255))
    pygame.display.flip()
    return screen


def paint_trajectories(screen, data):
    plt.plot(data.T)
    plot_stream = io.BytesIO()
    plt.savefig(plot_stream)
    plot_stream.seek(0)
    plot_surface = pygame.image.load(plot_stream, 'PNG')
    screen.fill(0)
    screen.blit(plot_surface, (0, 0))


def run(screen, grn, key, step=10):
    running = True
    grn.set_time(n_secs=1)
    output, _ = grn(key=key)
    t0 = 1.0
    grn.set_time(n_secs=step)
    data = np.array(output.ys)
    state = 1
    while running:
        if state:
            output, _ = grn(key=key,
                            y0=output.ys[:, -1],
                            w0=output.ws[:, -1],
                            t0=t0)
            data = np.hstack([data, output.ys])
            t0 += step
        paint_trajectories(screen=screen,
                           data=data)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    state = 0
                else:
                    state = 1
        pygame.display.flip()
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    plt.gca().set_xscale("linear")
    window = create_window(title="biomodel-{}".format(args.task))
    k = jax.random.PRNGKey(args.seed)
    g = GeneRegulatoryNetwork.create(biomodel_idx=int(args.task.split("-")[0]))
    run(screen=window, grn=g, key=k)
