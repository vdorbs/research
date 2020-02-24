class Visual:
    def boundary(self, N):
        raise NotImplementedError

    def plot_boundary(self, ax, N, color='k', linewidth=3, label=''):
        return ax.plot(*self.boundary(N).T, color, linewidth=linewidth, label=label)
